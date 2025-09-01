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

from jepo_core_algos import (
    attach_jepo_adv_to_dataproto,
    dummy_backward_fsdp_safe,
    compute_segment_logprobs_shifted as compute_jepo_from_logits_efficient,
    token_level_jepo_loss as compute_jepo_token_level_pg_loss_from_logits,
    _allreduce_sum_scalar,
)

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
        accum_steps = int(jepo_cfg.get("accum_steps", 4))               # fixed accumulate steps for consistent backwards
        format_penalty = float(jepo_cfg.get("format_penalty", 0.0))
        beta_supp = float(jepo_cfg.get("beta_supp", 0.001))
        beta_kl = float(jepo_cfg.get("beta_kl", 0.0))
        kl_loss_type = getattr(self.config, "kl_loss_type", "low_var_kl")
        temperature = float(data.meta_info["temperature"])
        n = int(jepo_cfg.get("num_response_per_question", 8))  # number of responses per question

        # -------- compute advantages and add to data --------
        data = attach_jepo_adv_to_dataproto(
            data=data,
            model=self.actor_module,
            jepo_cfg=jepo_cfg,
            cached_tokenizer=self._cached_tokenizer
        )
        
        # -------- do not drop samples; keep mask & counts --------
        has_delimiter_mask = data.batch["has_delimiter"]  # [N]


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
            kl_loss=0.0,
        )
        meter_count = 0
        num_delim = 0

        # Counts
        num_responses = int(data.batch["responses"].shape[0])
        num_delim = int(has_delimiter_mask.sum().item())
        # Extract stored advantages and weights (all responses)
        jepo_adv_raw_all = data.batch.get("jepo_adv_raw", data.batch["jepo_adv"])  # fallback for raw
        format_adv_all = data.batch.get("format_adv", torch.zeros_like(jepo_adv_raw_all))
        jepo_weights_all = data.batch["jepo_weights"]

        # Buffer-wide diagnostics for format advantage
        try:
            fmt_max = float(torch.max(format_adv_all).detach().item())
        except Exception:
            fmt_max = 0.0

        # Globalized delimiter fraction across FSDP ranks
        try:
            _dev_glob = has_delimiter_mask.device
            num_resp_glob = _allreduce_sum_scalar(num_responses, device=_dev_glob)
            num_delim_glob = _allreduce_sum_scalar(num_delim, device=_dev_glob)
            frac_delim_glob = float(num_delim_glob / max(num_resp_glob, 1.0))
        except Exception:
            frac_delim_glob = float(num_delim / max(num_responses, 1))

        # -------- training loop with fixed accumulate steps --------
        for _ in range(epochs):
            self.actor_optimizer.zero_grad()
            
            for k in range(accum_steps):
                step_start = k * micro_bs
                step_end = min(step_start + micro_bs, num_responses)
                
                # FSDP: delay allreduce until last accum step
                sync_ctx = (
                    self.actor_module.no_sync()
                    if isinstance(self.actor_module, (FSDP, FSDPModule)) and (k < accum_steps - 1)
                    else nullcontext()
                )
                with sync_ctx:
                    if step_start >= num_responses:
                        # Still do a dummy backward to keep FSDP sync aligned across ranks
                        dummy_backward_fsdp_safe(self.actor_module, scaler=None)
                    else:
                        # Get step data by slicing full batch
                        step_indices = torch.arange(step_start, step_end, device=data.batch["responses"].device)
                        step_data = data[step_indices]
                        B_step = len(step_indices)
                        
                        # Prepare data dict for compute_jepo_from_logits_efficient
                        # Convert tensors back to lists for the function interface
                        step_data_dict = {
                            'cot_start_positions': step_data.batch["cot_start_positions"].cpu().tolist(),
                            'answer_start_positions': step_data.batch["answer_start_positions"].cpu().tolist(),
                            'cot_tokens_list': [step_data.batch["cot_tokens"][i][step_data.batch["cot_tokens"][i] != self._cached_tokenizer.pad_token_id].cpu().tolist() for i in range(B_step)],
                            'ground_truth_answer_tokens': [step_data.batch["ground_truth_tokens"][i][step_data.batch["ground_truth_tokens"][i] != self._cached_tokenizer.pad_token_id].cpu().tolist() for i in range(B_step)],
                            # Provide teacher-forced input ids so delimiter span can be labeled
                            'batch_input_ids': step_data.batch["batch_input_ids"],
                        }
                        
                        # Forward pass to get logits (eval mode, gradients enabled)
                        was_training_actor = self.actor_module.training
                        self.actor_module.eval()
                        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                            out = self.actor_module(
                                input_ids=step_data.batch["batch_input_ids"],
                                attention_mask=step_data.batch["batch_attention_mask"],
                                position_ids=step_data.batch["batch_position_ids"],
                                use_cache=False,
                            )
                            logits = out.logits
                            logits.div_(temperature)
                        if was_training_actor:
                            self.actor_module.train()
                        
                        # Build GRPO-style token-level loss for JEPO
                        A_vec = jepo_adv_raw_all[step_indices] + format_adv_all[step_indices]
                        w_vec = jepo_weights_all[step_indices]
                        loss_parts = compute_jepo_token_level_pg_loss_from_logits(
                            logits=logits,
                            data_dict=step_data_dict,
                            A_vec=A_vec,
                            w_vec=w_vec,
                            beta_supp=float(beta_supp),
                            normalize_by="tokens",
                            vocab_chunk=8192,
                        )
                        # For monitoring and optional KL
                        jepo_loss_part = loss_parts["jepo_loss_part"]
                        supp_loss_part = loss_parts["supp_loss_part"]
                        cot_log_probs = loss_parts["cot_log_probs"]
                        answer_log_probs = loss_parts["answer_log_probs"]
                        denom_all = loss_parts["denom"]

                        # Optionally compute KL term against reference policy on the same tokens
                        kl_loss_part = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                        if beta_kl > 0 and hasattr(self, "_ref_module") and self._ref_module is not None:
                            was_training_ref = self._ref_module.training
                            self._ref_module.eval()
                            with torch.no_grad():
                                with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                                    out_ref = self._ref_module(
                                        input_ids=step_data.batch["batch_input_ids"],
                                        attention_mask=step_data.batch["batch_attention_mask"],
                                        position_ids=step_data.batch["batch_position_ids"],
                                        use_cache=False,
                                    )
                                    ref_logits = out_ref.logits
                                    ref_logits.div_(temperature)
                            if was_training_ref:
                                self._ref_module.train()
                            _, ref_answer_log_probs, _ = compute_jepo_from_logits_efficient(
                                logits=ref_logits,
                                data_dict=step_data_dict,
                                format_penalty=format_penalty,
                                has_delimiter=has_delimiter_mask[step_indices].tolist(),
                                vocab_chunk=8192,
                            )
                            # kld over answer sums per sample; normalize by contributing samples
                            kld = kl_penalty(
                                logprob=answer_log_probs, ref_logprob=ref_answer_log_probs, kl_penalty=kl_loss_type
                            )
                            # denom handled below
                        
                        # KL term already computed above if enabled. Normalize by token denom for consistency
                        kl_loss_part = (kld.sum() / denom_all) * beta_kl if 'kld' in locals() else torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                        
                        loss_chunk = (jepo_loss_part + supp_loss_part + kl_loss_part) / accum_steps
                        print("finish calculate jepo loss")
                        loss_chunk.backward()
                        
                        # Accumulate metrics
                        meters["total_loss"] += float(loss_chunk.detach())
                        meters["jepo_loss"] += float(jepo_loss_part.detach()) / accum_steps
                        meters["supp_loss"] += float(supp_loss_part.detach()) / accum_steps
                        with torch.no_grad():
                            meters["jepo_advs_mean"] += float(A_vec.mean().detach())
                            meters["jepo_advs_std"] += float(A_vec.std().detach())
                        meters["cot_log_probs_mean"] += float(cot_log_probs.mean().detach())
                        meters["log_mean_answer_probs_mean"] += float(answer_log_probs.mean().detach())
                        meters["kl_loss"] += float(kl_loss_part.detach()) / accum_steps
                        meter_count += 1
                        
                        del out, logits, cot_log_probs, answer_log_probs
                        torch.cuda.empty_cache()

            # one optimizer step per epoch
            grad_norm = self._optimizer_step()
            meters["grad_norm"] += float(grad_norm.detach())

        print("number of responses has delimiter for this rank:", num_delim)

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
            "jepo_actor/kl_loss": meters.get("kl_loss", 0.0),
            "jepo_actor/beta_kl": beta_kl,
            "jepo_buffer/num_has_delimiter": int(num_delim),
            "jepo_buffer/frac_has_delimiter": float(num_delim / max(num_responses, 1)),
            "jepo_buffer/frac_has_delimiter_global": frac_delim_glob,
            "jepo_buffer/format_adv_max": fmt_max,
            "jepo_actor/format_penalty": format_penalty,
        }
