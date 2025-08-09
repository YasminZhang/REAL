# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import torch
import numpy as np

sys.path.append('/home/aiscuser/jepo/recipe/jepo')

from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from jepo_core_algos import compute_jepo_advantage

__all__ = ["JEPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


class JEPOActor(DataParallelPPOActor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initialized JEPO Actor")

    def update_policy(self, data: DataProto):
        self.actor_module.train()
        
        temperature = data.meta_info["temperature"]
        
        # Compute response mask and JEPO advantages for the whole batch first
        data.batch["response_mask"] = compute_response_mask(data)
        
        uids = data.non_tensor_batch.get("uid", [])
        if len(uids) > 0:
            uid_index = np.array(uids)
            token_level_rewards = data.batch.get("token_level_scores", torch.zeros_like(data.batch["responses"], dtype=torch.float))
            response_mask = data.batch["response_mask"]
            advantages, _ = compute_jepo_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask, 
                index=uid_index
            )
            data.batch["jepo_advantages"] = advantages
        else:
            # Fallback if no UIDs
            # data.batch["jepo_advantages"] = torch.zeros_like(data.batch["responses"], dtype=torch.float)
            raise ValueError("JEPO requires UIDs in the data batch for advantage computation.")
        
        select_keys = [
            "responses", 
            "response_mask",
            "input_ids",
            "attention_mask", 
            "position_ids",
            "old_log_probs",
            "jepo_advantages"
        ]
        
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        
        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = self._process_jepo_micro_batch(micro_batch, temperature)
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
                
        self.actor_optimizer.zero_grad()
        return metrics

    def _process_jepo_micro_batch(self, micro_batch, temperature):
        """Process a single JEPO micro batch and return metrics"""
        micro_batch_metrics = {}
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        
        response_mask = model_inputs["response_mask"]
        old_log_prob = model_inputs["old_log_probs"]
        jepo_advantages = model_inputs["jepo_advantages"]
        
        entropy_coeff = self.config.entropy_coeff
        loss_agg_mode = self.config.loss_agg_mode
        
        if self.config.use_dynamic_bsz:
            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
        else:
            loss_scale_factor = 1 / self.gradient_accumulation
        
        # Calculate entropy if needed
        calculate_entropy = entropy_coeff != 0
        entropy, log_prob = self._forward_micro_batch(
            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
        )
        
        # Compute JEPO policy loss using advantages (similar to policy gradient loss)
        pg_losses = -jepo_advantages * log_prob
        from verl.trainer.ppo.core_algos import agg_loss
        pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        
        # Add entropy loss if configured
        if entropy_coeff != 0:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            jepo_loss = pg_loss - entropy_loss * entropy_coeff
            micro_batch_metrics["actor/entropy_loss"] = entropy_loss.detach().item() * loss_scale_factor
        else:
            jepo_loss = pg_loss
        
        # Add KL loss if configured
        if self.config.use_kl_loss:
            ref_log_prob = model_inputs["ref_log_prob"]
            from verl.trainer.ppo.core_algos import kl_penalty
            kld = kl_penalty(
                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
            )
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            jepo_loss = jepo_loss + kl_loss * self.config.kl_loss_coef
            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
        
        loss = jepo_loss * loss_scale_factor
        loss.backward()
        
        micro_batch_metrics.update({
            "actor/jepo_loss": jepo_loss.detach().item() * loss_scale_factor,
            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
        })
        
        return micro_batch_metrics