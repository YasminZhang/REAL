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

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor
from tqdm import tqdm

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

from jepo_core_algos import compute_jepo_advantages, compute_jepo_advantages_from_logprobs, compute_jepo_from_logprobs_fast_with_grad_mean

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
        self._cached_tokenizer = None

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def update_policy(self, data: DataProto):
        self.actor_module.train()
        self.actor_optimizer.zero_grad()

        # Compute response mask and JEPO advantages for the whole batch first
        if "response_mask" not in data.batch:
            data.batch["response_mask"] = compute_response_mask(data)
        
        jepo_config = data.meta_info.get("jepo_config", {})
        model_inputs = {**data.batch, **data.non_tensor_batch}
        ground_truths = model_inputs.get("reward_model", {})
        ground_truths_tokens = np.array([self._cached_tokenizer.encode(gt.get("ground_truth", [])) for gt in ground_truths], dtype=object)
        delimiter = jepo_config.get("delimiter", "\n\n")
        print(f"Using delimiter: {delimiter}")
        format_penalty = jepo_config.get("format_penalty", 1.0)
        beta_supp = jepo_config.get("beta_supp", 1.0)
        beta_kl = jepo_config.get("beta_kl", 0.1)
        pad_token = self._cached_tokenizer.pad_token_id

        # Get data dictionaries for creating DataProto objects
        # data_dicts contains all questions.
        #breakpoint()
        data_dicts = compute_jepo_advantages(
            response_tokens=data.batch["responses"],
            prompt_tokens=data.batch["prompts"],
            ground_truth_answer_tokens=ground_truths_tokens,
            delimiter_str=delimiter,
            format_penalty=format_penalty,
            model=self.actor_module,
            device=self.actor_module.device,
            pad_token=pad_token,
            index=data.non_tensor_batch["uid"],
            tokenizer=self._cached_tokenizer
        )
        
        # Process each data dict to compute log probabilities using parent's _forward_micro_batch
        all_advantages = []
        all_cot_log_probs = []
        all_log_mean_answer_probs = []
        
        temperature = data.meta_info["temperature"]
        num_delimiter = 0
        for data_dict in tqdm(data_dicts):
            # Use the model directly with the optimized padded inputs
            # The _forward_micro_batch only gives us log_probs for specific tokens, 
            # but JEPO needs the full vocabulary distribution
            # for every single question
            num_delimiter += np.sum(data_dict['has_delimiter'])
            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                output = self.actor_module(
                    input_ids=data_dict['batch_input_ids'].detach(),
                    attention_mask=data_dict['attention_mask'],
                    position_ids=data_dict['position_ids'],
                    use_cache=False,
                )
                logits = output.logits
                logits.div_(temperature)
                log_probs_batch = torch.log_softmax(logits, dim=-1)
            # Compute JEPO advantages from log probabilities
            print("Finish computing log probabilities")
            jepo_advs, cot_log_probs, _, log_mean = compute_jepo_from_logprobs_fast_with_grad_mean(
                log_probs_batch, data_dict, format_penalty, data_dict['has_delimiter'])

            print("Finish computing JEPO advantages")
            all_advantages.append(jepo_advs)
            all_cot_log_probs.append(cot_log_probs) 
            all_log_mean_answer_probs.append(log_mean)
        print("number of responses has delimiter:", num_delimiter)
        # Concatenate results
        jepo_advs = torch.cat(all_advantages, dim=0)
        cot_log_probs = torch.cat(all_cot_log_probs, dim=0)
        log_mean_answer_probs = torch.stack(all_log_mean_answer_probs)

        # data.batch["jepo_advs"] = jepo_advs
        # data.batch["cot_log_probs"] = cot_log_probs
        # data.batch["log_mean_answer_probs"] = log_mean_answer_probs

        # Compute individual loss components
        jepo_loss = (jepo_advs * cot_log_probs).mean()
        supp_loss = (log_mean_answer_probs * beta_supp).mean()
        
        loss = jepo_loss + supp_loss
        print("Finish calculate JEPO loss:", loss.item())
        loss.backward()
        print("Finish backward JEPO loss")
        grad_norm = self._optimizer_step()
        print("Finish doing gradient descent")
        
        # Collect metrics
        metrics = {
            "jepo_actor/jepo_loss": jepo_loss.detach().item(),
            "jepo_actor/supp_loss": supp_loss.detach().item(),
            "jepo_actor/total_loss": loss.detach().item(),
            "jepo_actor/grad_norm": grad_norm.detach().item(),
            "jepo_actor/jepo_advs_mean": jepo_advs.mean().detach().item(),
            "jepo_actor/jepo_advs_std": jepo_advs.std().detach().item(),
            "jepo_actor/cot_log_probs_mean": cot_log_probs.mean().detach().item(),
            "jepo_actor/log_mean_answer_probs_mean": log_mean_answer_probs.mean().detach().item(),
            "jepo_actor/beta_supp": beta_supp,
            "jepo_actor/format_penalty": format_penalty,
        }
        
        self.actor_optimizer.zero_grad()
        return metrics