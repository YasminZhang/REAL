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

from jepo_core_algos import compute_jepo_advantages, compute_jepo_from_logits_sparse, jepo_two_pass_step_for_one_question, precompute_adv_for_dd, compute_advantages_with_dataproto, compute_jepo_adv_with_dataproto, compute_single_jepo_advantages, dummy_backward_fsdp_safe

__all__ = ["JEPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]




from contextlib import nullcontext
import math
import numpy as np
import torch

def _chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

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

        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="jepo actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()
        self.actor_optimizer.zero_grad()

        # -------- config --------
        jepo_cfg = data.meta_info.get("jepo_config", {}) or {}
        epochs = int(jepo_cfg.get("epochs", 1))
        mini_bs = int(jepo_cfg.get("mini_batch_size_per_gpu", 8))        # questions per optimizer step per rank
        micro_bs = int(jepo_cfg.get("micro_batch_size_per_gpu", 1))      # questions per backward call
        resp_micro_bs = int(jepo_cfg.get("responses_micro_batch_size", 8))  # responses per backward inside a question
        format_penalty = float(jepo_cfg.get("format_penalty", 0.0))
        beta_supp = float(jepo_cfg.get("beta_supp", 0.001))
        temperature = float(data.meta_info["temperature"])

        # -------- compute advantages and add to data --------
        data = compute_jepo_adv_with_dataproto(
            data=data,
            model=self.actor_module,
            jepo_cfg=jepo_cfg,
            cached_tokenizer=self._cached_tokenizer
        )
        
        # -------- filter data where has_delimiter is True --------
        has_delimiter_mask = data.batch["has_delimiter"]
        filtered_data = data[has_delimiter_mask]


        # -------- meters --------
        meters = dict(
            jepo_loss=0.0,
            supp_loss=0.0,
            total_loss=0.0,
            grad_norm=0.0,
            jepo_advs_mean=0.0,
            jepo_advs_std=0.0,
            cot_log_probs_mean=0.0,
            log_mean_answer_probs_mean=0.0,
        )
        meter_count = 0
        num_delim = 0

        # Get total number of filtered responses for scaling
        total_filtered = filtered_data.batch["responses"].shape[0]
        num_delim = total_filtered
        
        # Extract stored advantages and weights
        jepo_advantages = filtered_data.batch["jepo_adv"]  # [N_filtered]
        jepo_weights = filtered_data.batch["jepo_weights"]  # [N_filtered]
        
        # -------- training loop (mini/micro/accum) --------
        for _ in range(epochs):
            # Split filtered data into mini-batches
            filtered_indices = torch.arange(total_filtered, device=filtered_data.batch["responses"].device)
            
            for mini_start in range(0, total_filtered, mini_bs):
                mini_end = min(mini_start + mini_bs, total_filtered)
                mini_indices = filtered_indices[mini_start:mini_end]
                N_responses = len(mini_indices)  # number of responses in this optimizer step
                accum_steps = math.ceil(N_responses / micro_bs)
                self.actor_optimizer.zero_grad()

                for k in range(accum_steps):
                    micro_start = k * micro_bs
                    micro_end = min(micro_start + micro_bs, N_responses)
                    micro_indices = mini_indices[micro_start:micro_end]
                    
                    # FSDP: delay allreduce until last accum step
                    sync_ctx = (
                        self.actor_module.no_sync()
                        if isinstance(self.actor_module, (FSDP, FSDPModule)) and (k < accum_steps - 1)
                        else nullcontext()
                    )
                    with sync_ctx:
                        if len(micro_indices) == 0:
                            # Still do a dummy backward to keep FSDP sync aligned across ranks
                            dummy_backward_fsdp_safe(self.actor_module, scaler=None)
                            continue
                        # Get micro-batch data
                        micro_data = filtered_data[micro_indices]
                        B_micro = len(micro_indices)
                        
                        # Prepare data dict for compute_jepo_from_logits_sparse
                        micro_data_dict = {
                            'cot_start_positions': [micro_data.non_tensor_batch["cot_start_positions"][i] for i in range(B_micro)],
                            'answer_start_positions': [micro_data.non_tensor_batch["answer_start_positions"][i] for i in range(B_micro)],
                            'cot_tokens_list': [micro_data.non_tensor_batch["cot_tokens_list"][i] for i in range(B_micro)],
                            'ground_truth_answer_tokens': [micro_data.non_tensor_batch["ground_truth_answer_tokens"][i] for i in range(B_micro)]
                        }
                        
                        # Forward pass to get logits
                        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                            out = self.actor_module(
                                input_ids=micro_data.batch["batch_input_ids"],
                                attention_mask=micro_data.batch["batch_attention_mask"],
                                position_ids=micro_data.batch["batch_position_ids"],
                                use_cache=False,
                            )
                            logits = out.logits
                            logits.div_(temperature)
                        
                        # Compute CoT and answer log probabilities using the sparse function
                        _, cot_log_probs, answer_log_probs, _ = compute_jepo_from_logits_sparse(
                            logits=logits,
                            data_dict=micro_data_dict,
                            format_penalty=format_penalty,
                            has_delimiter=[True] * B_micro,  # All filtered data has delimiter
                            vocab_chunk=8192
                        )
                        
                        # Get advantages and weights for this micro-batch
                        micro_advantages = jepo_advantages[micro_indices]
                        micro_weights = jepo_weights[micro_indices]
                        
                        # Compute losses
                        jepo_loss_part = (micro_advantages * cot_log_probs).sum() / total_filtered
                        supp_loss_part = beta_supp * (micro_weights * answer_log_probs).sum() / total_filtered
                        
                        loss_chunk = (jepo_loss_part + supp_loss_part) / accum_steps
                        loss_chunk.backward()
                        
                        # Accumulate metrics
                        meters["total_loss"] += float(loss_chunk.detach())
                        meters["jepo_loss"] += float(jepo_loss_part.detach()) / accum_steps
                        meters["supp_loss"] += float(supp_loss_part.detach()) / accum_steps
                        meters["jepo_advs_mean"] += float(micro_advantages.mean().detach())
                        meters["jepo_advs_std"] += float(micro_advantages.std().detach())
                        meters["cot_log_probs_mean"] += float(cot_log_probs.mean().detach())
                        meters["log_mean_answer_probs_mean"] += float(answer_log_probs.mean().detach())
                        meter_count += 1
                        
                        del out, logits, cot_log_probs, answer_log_probs
                        torch.cuda.empty_cache()

                # one optimizer step per mini-batch
                grad_norm = self._optimizer_step()
                meters["grad_norm"] += float(grad_norm.detach())

        print("number of responses has delimiter:", num_delim)

        # average meters
        if meter_count > 0:
            for k in meters:
                meters[k] /= meter_count

        return {
            "jepo_actor/jepo_loss": meters["jepo_loss"],
            "jepo_actor/supp_loss": meters["supp_loss"],
            "jepo_actor/total_loss": meters["total_loss"],
            "jepo_actor/grad_norm": meters["grad_norm"],
            "jepo_actor/jepo_advs_mean": meters["jepo_advs_mean"],
            "jepo_actor/jepo_advs_std": meters["jepo_advs_std"],
            "jepo_actor/cot_log_probs_mean": meters["cot_log_probs_mean"],
            "jepo_actor/log_mean_answer_probs_mean": meters["log_mean_answer_probs_mean"],
            "jepo_actor/beta_supp": beta_supp,
            "jepo_actor/format_penalty": format_penalty,
        }