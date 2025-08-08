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
from verl.workers.actor.dp_actor import DataParallelPPOActor
from jepo_core_algos import compute_jepo_advantages_batched, jepo_loss_batched

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
        
        jepo_config = data.meta_info.get("jepo_config", {})

        data.batch["response_mask"] = compute_response_mask(data)
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        
        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                metrics.update(self._process_mini_batch(mini_batch, jepo_config))
                
        self.actor_optimizer.zero_grad()
        return metrics

    def _process_mini_batch(self, mini_batch, jepo_config):
        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            from verl.utils.seqlen_balancing import prepare_dynamic_batch
            micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
        else:
            self.gradient_accumulation = (
                self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            )
            micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

        self.actor_optimizer.zero_grad()
        metrics = {}
        
        for micro_batch in micro_batches:
            batch_metrics = self._process_micro_batch(micro_batch, jepo_config)
            append_to_dict(metrics, batch_metrics)
            
        grad_norm = self._optimizer_step()
        metrics.update({"jepo_actor/grad_norm": grad_norm.detach().item()})
        return metrics

    def _process_micro_batch(self, micro_batch, jepo_config):
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        response_mask = model_inputs["response_mask"]
        
        if self.config.use_dynamic_bsz:
            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
        else:
            loss_scale_factor = 1 / self.gradient_accumulation

        policy_loss = self._compute_jepo_loss(model_inputs, jepo_config)
        
        loss = policy_loss * loss_scale_factor
        loss.backward()
        
        return {"jepo_actor/policy_loss": policy_loss.detach().item() * loss_scale_factor}

    def _compute_jepo_loss(self, model_inputs, jepo_config):
        response_tokens_tensor = model_inputs.get("responses")
        uids = model_inputs.get("uid", [])
        
        if response_tokens_tensor is None or len(uids) == 0:
            return torch.tensor(0.0, device=self.actor_module.device, requires_grad=True)
        
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            logger.error("Tokenizer not found")
            return torch.tensor(0.0, device=self.actor_module.device, requires_grad=True)
        
        try:
            loss_components = self._compute_jepo_by_uid_groups(
                uids, response_tokens_tensor, model_inputs, tokenizer, jepo_config
            )
            return loss_components.get("total_loss", torch.tensor(0.0, device=self.actor_module.device, requires_grad=True))
            
        except Exception as e:
            logger.error(f"JEPO computation failed: {e}")
            return torch.tensor(0.0, device=self.actor_module.device, requires_grad=True)

    def _compute_jepo_by_uid_groups(self, uids, response_tokens_tensor, model_inputs, tokenizer, jepo_config):
        unique_uids = np.unique(uids)
        
        questions_response_tokens = []
        questions_prompts = []
        questions_ground_truths = []
        
        ground_truths = model_inputs.get("reward_model", {}).get("ground_truth", [])
        prompts = model_inputs.get("prompts", [])
        
        for uid in unique_uids:
            uid_mask = torch.tensor([u == uid for u in uids], dtype=torch.bool, device=response_tokens_tensor.device)
            uid_indices = torch.where(uid_mask)[0]
            
            if len(uid_indices) <= 1:
                continue
                
            uid_response_tokens = self._extract_response_tokens(
                response_tokens_tensor[uid_mask], tokenizer
            )
            
            uid_prompt = prompts[uid_indices[0]] if prompts else ""
            uid_ground_truth = ground_truths[uid_indices[0]] if ground_truths else ""
            
            questions_response_tokens.append(uid_response_tokens)
            questions_prompts.append(uid_prompt)
            questions_ground_truths.append(uid_ground_truth)
        
        if not questions_response_tokens:
            return {"total_loss": torch.tensor(0.0, device=self.actor_module.device, requires_grad=True)}
            
        return self._compute_batched_jepo_loss(
            questions_response_tokens, questions_prompts, questions_ground_truths, 
            tokenizer, jepo_config
        )

    def _extract_response_tokens(self, response_tensor, tokenizer):
        response_tokens = []
        for i in range(response_tensor.shape[0]):
            tokens = response_tensor[i].cpu().tolist()
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                tokens = [t for t in tokens if t != tokenizer.pad_token_id]
            response_tokens.append(tokens)
        return response_tokens

    def _compute_batched_jepo_loss(self, questions_response_tokens, questions_prompts, 
                                   questions_ground_truths, tokenizer, jepo_config):
        
        delimiter = jepo_config.get("delimiter", "\n\n")
        format_penalty = jepo_config.get("format_penalty", 1.0)
        beta_supp = jepo_config.get("beta_supp", 1.0)
        beta_kl = jepo_config.get("beta_kl", 0.1)
        
        tilde_A_i_list, tilde_A_i_ref_list, cot_log_probs_list, answer_log_probs_list = compute_jepo_advantages_batched(
            questions_response_tokens=questions_response_tokens,
            questions_prompts=questions_prompts,
            ground_truth_answers=questions_ground_truths,
            tokenizer=tokenizer,
            delimiter=delimiter,
            format_penalty=format_penalty,
            model=self.actor_module,
            device=self.actor_module.device
        )
        
        return jepo_loss_batched(
            questions_cot_log_probs=cot_log_probs_list,
            questions_answer_log_probs=answer_log_probs_list,
            questions_tilde_A_i=tilde_A_i_list,
            questions_tilde_A_i_ref=tilde_A_i_ref_list,
            beta_supp=beta_supp,
            beta_kl=beta_kl
        )

    def _get_tokenizer(self):
        return getattr(self.actor_module, 'tokenizer', None) or getattr(self.config, 'tokenizer', None)