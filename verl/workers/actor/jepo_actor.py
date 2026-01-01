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

import numpy as np
import torch

sys.path.append('/home/aiscuser/jepo/recipe/jepo')

import logging
import os

import ray
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (agg_loss, get_policy_loss_fn,
                                         kl_penalty)
from verl.utils.device import (get_device_name, is_cuda_available,
                               is_npu_available)
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (gather_outputs_and_unpad, ulysses_pad,
                                ulysses_pad_and_slice_inputs)
from verl.workers.actor import BasePPOActor
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import (index_first_axis, pad_input,
                                         rearrange, unpad_input)
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

from jepo_core_algos import (_allreduce_sum_scalar,
                             attach_jepo_adv_to_dataproto,
                             dummy_backward_fsdp_safe)

__all__ = ["JEPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


## Note: dynamic micro-batching uses verl.utils.seqlen_balancing.prepare_dynamic_batch
## No custom token bucketing helpers are used here to avoid divergence from internals.


import math
from contextlib import nullcontext

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

    # no wrapper for dynamic bsz in precompute; use prepare_dynamic_batch directly there

    @GPUMemoryLogger(role="jepo actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()
        self.actor_optimizer.zero_grad()

        # -------- config --------
        jepo_cfg = data.meta_info.get("jepo_config", {}) or {}
        epochs = int(jepo_cfg.get("epochs", 1))
        mini_bs = int(jepo_cfg.get("mini_batch_size_per_gpu", 8))        # questions per optimizer step per rank
        micro_bs = int(jepo_cfg.get("micro_batch_size_per_gpu", 4))      # questions per backward call
        format_penalty = float(jepo_cfg.get("format_penalty", 0.0))
        beta_supp = float(jepo_cfg.get("beta_supp", 0.001))
        beta_supp_extra = float(jepo_cfg.get("beta_supp_extra", 0.001))
        beta_kl = float(jepo_cfg.get("beta_kl", 0.0))
        kl_loss_type = getattr(self.config, "kl_loss_type", "low_var_kl")
        temperature = float(data.meta_info["temperature"])
        
        print('jepo_cfg:', jepo_cfg)

        assert mini_bs % micro_bs == 0, "Expected mini_bs to be multiple of micro_bs"

        # Entropy regularization (match dp_actor behavior)
        entropy_coeff = float(jepo_cfg.get("entropy_coeff", 0.0))
        loss_agg_mode = jepo_cfg.get("loss_agg_mode", "token-mean")

        # -------- build teacher-forced packs per question --------
        data = attach_jepo_adv_to_dataproto(
            data=data,
            model=self.actor_module,
            jepo_cfg=jepo_cfg,
            cached_tokenizer=self._cached_tokenizer
        )
        # Precompute A_raw, format_adv, and weights using teacher-forced batches per question
        # Note: jepo_data_dicts carries per-question grouping; it's only needed for precompute.
        if "jepo_data_dicts" in data.non_tensor_batch:
            # Drop grouping artifact to allow generic per-response slicing downstream
            try:
                del data.non_tensor_batch["jepo_data_dicts"]
            except Exception:
                pass
            data = self._precompute_adv_w_with_verl(data, temperature=temperature, format_penalty=format_penalty)
            
        
        # -------- do not drop samples; keep mask & counts --------
        has_delimiter_mask = data.batch["has_delimiter"]  # [N]


        # -------- meters --------
        meters = dict(
            extra_loss=0.0,
            raw_jepo_loss=0.0,
            raw_cot_loss=0.0,
            raw_log_likelihood_loss=0.0,
            raw_l2_loss=0.0,
            raw_supp_loss=0.0,
            raw_total_loss=0.0,
            raw_kl_loss=0.0,
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
        # Safely read JEPO advantages: prefer 'jepo_adv_raw' if present; fallback to 'jepo_adv' without
        # eagerly indexing a missing key (TensorDict.get default is evaluated before call)
        if "jepo_adv_raw" in set(data.batch.keys()):
            jepo_adv_raw_all = data.batch["jepo_adv_raw"]
        else:
            jepo_adv_raw_all = data.batch["jepo_adv"]
        format_adv_all = data.batch.get("format_adv", torch.zeros_like(jepo_adv_raw_all))
        
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

        use_dynamic_bsz = bool(jepo_cfg.get("use_dynamic_bsz", True))
        max_token_len = (
            jepo_cfg.get("ppo_max_token_len_per_gpu", 16384)
            * getattr(self, "ulysses_sequence_parallel_size", 1)
        )

        for _ in range(epochs):
            self.actor_optimizer.zero_grad()
            self.actor_module.train()

            # dp_actor-style mini-batch iterator
            mini_batches = data.split(mini_bs)

            for mini_batch in mini_batches:
                # dp_actor-style micro-batch iterator
                if use_dynamic_bsz:
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = max(1, mini_bs // max(micro_bs, 1))
                    micro_batches = mini_batch.split(micro_bs)

                # Sanity check: ensure equal micro-batch counts across ranks
                try:
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        t = torch.tensor([len(micro_batches)], device=get_device_name(), dtype=torch.int32)
                        t_min = t.clone()
                        t_max = t.clone()
                        torch.distributed.all_reduce(t_min, op=torch.distributed.ReduceOp.MIN)
                        torch.distributed.all_reduce(t_max, op=torch.distributed.ReduceOp.MAX)
                        assert (
                            int(t_min.item()) == int(t_max.item())
                        ), f"JEPO micro-batch count mismatch across ranks: min={int(t_min.item())}, max={int(t_max.item())}"
                except Exception:
                    raise RuntimeError("JEPO micro-batch count check failed")

                self.actor_optimizer.zero_grad()
                for micro_batch in micro_batches:
                    dev = micro_batch.batch["batch_input_ids"].device
                    pad_id = self._cached_tokenizer.pad_token_id if self._cached_tokenizer is not None else 0

                    ids_full = micro_batch.batch["batch_input_ids"]
                    attn_full = micro_batch.batch["attention_mask"]
                    pos_full = micro_batch.batch["position_ids"]
                    a_s = micro_batch.batch["answer_start_positions"].tolist()
                    c_s = micro_batch.batch["cot_start_positions"].tolist()

                    gt_tokens_nd = micro_batch.non_tensor_batch.get("ground_truth_answer_tokens", None)
                    if gt_tokens_nd is None:
                        gt_tokens = [None] * int(ids_full.size(0))
                    else:
                        try:
                            import numpy as _np
                            gt_tokens = gt_tokens_nd.tolist() if isinstance(gt_tokens_nd, _np.ndarray) else gt_tokens_nd
                        except Exception:
                            gt_tokens = gt_tokens_nd

                    if "jepo_adv_raw" in set(micro_batch.batch.keys()):
                        jepo_adv_raw_mb = micro_batch.batch["jepo_adv_raw"]
                    else:
                        jepo_adv_raw_mb = micro_batch.batch["jepo_adv"]
                    format_adv_mb = micro_batch.batch.get("format_adv", torch.zeros_like(jepo_adv_raw_mb))
                    
                     

                    # add a parameter to not use format_adv_mb
                    use_format_adv = bool(jepo_cfg.get("use_format_adv", True))
                    A_all = jepo_adv_raw_mb + format_adv_mb if use_format_adv else jepo_adv_raw_mb # TODO: ablate

                    w_all = micro_batch.batch.get("jepo_weights", torch.ones((ids_full.size(0),), device=dev))
                    w_all_extra = micro_batch.batch.get("jepo_extra_weights", torch.ones((ids_full.size(0),), device=dev))

                    gt_values = micro_batch.batch.get("gt_values", torch.zeros((ids_full.size(0),), device=dev))

                    prefix_lens = [int(c) for c in c_s]
                    cot_lens = [max(0, int(a) - int(c)) for a, c in zip(a_s, c_s)]
                    ans_lens = [
                        (int(len(t)) if (t is not None) else max(0, int(ids_full.size(1)) - int(a)))
                        for t, a in zip(gt_tokens, a_s)
                    ]

                    Bp = int(ids_full.size(0))
                    R_ls = [cot_lens[i] + ans_lens[i] for i in range(Bp)]
                    P_max = max(prefix_lens) if prefix_lens else 0
                    R_max = max(R_ls) if R_ls else 0

                    if P_max == 0 and R_max == 0:
                        dummy_backward_fsdp_safe(self.actor_module, scaler=None)
                        continue

                    ids_pack = torch.full((Bp, P_max + R_max), pad_id, dtype=ids_full.dtype, device=dev)
                    attn_pack = torch.zeros((Bp, P_max + R_max), dtype=attn_full.dtype, device=dev)
                    pos_pack = torch.zeros((Bp, P_max + R_max), dtype=pos_full.dtype, device=dev)
                    resp_pack = torch.full((Bp, R_max), pad_id, dtype=ids_full.dtype, device=dev)
                    mask_cot = torch.zeros((Bp, R_max), dtype=torch.float32, device=dev)
                    mask_ans = torch.zeros((Bp, R_max), dtype=torch.float32, device=dev)
                    A_pack = torch.zeros((Bp, R_max), dtype=torch.float32, device=dev)

                    for j in range(Bp):
                        c = int(c_s[j]); a = int(a_s[j])
                        Lp = int(prefix_lens[j]); Lc = int(cot_lens[j]); La = int(ans_lens[j])
                        if Lp > 0:
                            ids_pack[j, :Lp] = ids_full[j, :c]
                            attn_pack[j, :Lp] = attn_full[j, :c]
                            pos_pack[j, :Lp] = pos_full[j, :c]
                        if Lc > 0:
                            ids_pack[j, Lp : Lp + Lc] = ids_full[j, c:a]
                            # Preserve attention semantics (mask out eos/pad carried in source)
                            attn_pack[j, Lp : Lp + Lc] = attn_full[j, c:a]
                            resp_pack[j, :Lc] = ids_full[j, c:a]
                            # Loss mask should also respect attention
                            mask_cot[j, :Lc] = attn_full[j, c:a].to(mask_cot.dtype)
                            A_pack[j, :Lc] = A_all[j]
                        if La > 0:
                            gt_row = gt_tokens[j]
                            if gt_row is None:
                                ans_ids = ids_full[j, a : a + La]
                            else:
                                ans_ids = torch.as_tensor(gt_row, device=dev, dtype=ids_full.dtype)
                            ids_pack[j, Lp + Lc : Lp + Lc + La] = ans_ids
                            attn_pack[j, Lp + Lc : Lp + Lc + La] = 1
                            # attention for GT answer tokens is 1 (no eos in GT by assumption)
                            resp_pack[j, Lc : Lc + La] = ans_ids
                            mask_ans[j, Lc : Lc + La] = 1
                            # First term: beta_supp * w_all[j] (existing)
                            # Second term: w_all[j] * last_token_log_prob will be added after forward pass
                            A_pack[j, Lc : Lc + La] = 0.0  # placeholder TODO: fix later 

                         
                      

                    # Derive positions from attention to ensure consistent left/right padding
                    if attn_pack.numel() > 0:
                        pos_from_mask = (attn_pack.cumsum(dim=1) - 1).clamp_min(0) * attn_pack
                        pos_pack[:, :] = pos_from_mask.to(dtype=pos_pack.dtype)

                    micro = {
                        "input_ids": ids_pack,
                        "attention_mask": attn_pack,
                        "position_ids": pos_pack,
                        "responses": resp_pack,
                    }
                    calculate_entropy = entropy_coeff != 0

                    entropy_tok, lp_combined, expected_values, last_token_log_probs = self._forward_micro_batch(
                        micro, temperature=temperature, calculate_entropy=calculate_entropy, regression=True, expected_prob_replace=True, 
                    )
                    
                    
                    l2_loss = ((expected_values - gt_values)**2).detach().mean().clone()
                    log_likelihood_loss = (-last_token_log_probs).detach().mean().clone()
                    
                    
                    use_log_loss = jepo_cfg.get("use_log_prob_loss", True)
                    use_extra_loss = jepo_cfg.get("use_extra_loss", True)
                    use_l2_loss = jepo_cfg.get("use_l2_loss", False)
                    extra_loss = 0.0
                    if use_extra_loss:
                        if use_log_loss and use_l2_loss:
                            extra_loss = float(beta_supp_extra) *   float(beta_supp) * ( (expected_values - gt_values)**2).mean() +  float(beta_supp_extra) *    (-last_token_log_probs).mean()
                        elif use_log_loss:
                            extra_loss = -  float(beta_supp_extra) *    last_token_log_probs.mean()
                        elif use_l2_loss:
                            extra_loss = float(beta_supp_extra) *   float(beta_supp) * ( (expected_values - gt_values)**2).mean()
                        else:
                            raise RuntimeError("At least one of use_log_loss or use_l2_loss must be True if use_extra_loss is True")
                    else:
                        extra_loss = torch.tensor(0.0, device=dev)
                  
                             
                         

                    gpg_fn = get_policy_loss_fn("gpg")
                    comb_mask = (mask_cot + mask_ans).clamp_max(1)
                    comb_adv = A_pack
                    
                   
                    
                    # breakpoint()

                    use_cot_loss = jepo_cfg.get("use_cot_loss", False)
                    if not use_cot_loss:
                        lp_combined = torch.zeros_like(lp_combined)
 

                    jepo_loss_part, cot_loss_backup, _, _ = gpg_fn(
                        old_log_prob=None,
                        log_prob=lp_combined,
                        advantages=comb_adv,
                        response_mask=comb_mask,
                        loss_agg_mode=loss_agg_mode,
                        extra_loss=extra_loss,
                        extra_loss_only=False
                    )


                    if calculate_entropy:
                        entropy_loss = agg_loss(
                            loss_mat=entropy_tok, loss_mask=comb_mask, loss_agg_mode=loss_agg_mode
                        )
                        jepo_loss_part = jepo_loss_part - entropy_coeff * entropy_loss

                    cot_log_probs = (lp_combined * (mask_cot > 0)).sum(dim=-1).detach()
                    answer_log_probs = (lp_combined * (mask_ans > 0)).sum(dim=-1).detach()

                    kl_loss_part = torch.tensor(0.0, device=dev)
                    if beta_kl > 0 and ("ref_log_prob" in micro_batch.batch.keys()):
                        pack_data = micro_batch
                        try:
                            micro_orig = {
                                "input_ids": pack_data.batch["input_ids"],
                                "attention_mask": pack_data.batch["attention_mask"],
                                "position_ids": pack_data.batch["position_ids"],
                                "responses": pack_data.batch["responses"],
                            }
                            _, lp_orig = self._forward_micro_batch(
                                micro_orig, temperature=temperature, calculate_entropy=False
                            )
                            ref_lp_orig = micro_batch.batch["ref_log_prob"]
                            resp_mask_orig = compute_response_mask(pack_data)
                            kld = kl_penalty(logprob=lp_orig, ref_logprob=ref_lp_orig, kl_penalty=kl_loss_type)
                            kl_loss_part = agg_loss(loss_mat=kld, loss_mask=resp_mask_orig, loss_agg_mode=loss_agg_mode) * beta_kl
                        except Exception:
                            kl_loss_part = torch.tensor(0.0, device=dev)

                    if use_dynamic_bsz:
                        loss_scale_factor = float(Bp) / float(max(mini_bs, 1))
                    else:
                        loss_scale_factor = 1.0 / float(max(self.gradient_accumulation, 1))
                    loss_chunk = (jepo_loss_part + kl_loss_part) * loss_scale_factor
                    loss_chunk.backward()

                    meters["total_loss"] += float(loss_chunk.detach())
                    meters["extra_loss"] += float(extra_loss.mean().detach())  
                    meters["jepo_loss"] += float(jepo_loss_part.detach()) * loss_scale_factor
                    
                    meters["raw_total_loss"] += float(loss_chunk.detach()) / loss_scale_factor
                    meters["raw_cot_loss"] += float(cot_loss_backup.detach()) 
                    meters["raw_supp_loss"] += float(l2_loss.detach()) + float(log_likelihood_loss.detach())
                    meters["raw_l2_loss"] += float(l2_loss.detach())
                    meters["raw_log_likelihood_loss"] += float(log_likelihood_loss.detach())

                    meters["raw_kl_loss"] += float(kl_loss_part.detach().item())
                    with torch.no_grad():
                        meters["jepo_advs_mean"] += float(A_all.mean().detach())
                        meters["jepo_advs_std"] += float(A_all.std(unbiased=False).detach())
                    if cot_log_probs.numel() > 0:
                        meters["cot_log_probs_mean"] += float(cot_log_probs.mean().detach())
                    if answer_log_probs.numel() > 0:
                        meters["log_mean_answer_probs_mean"] += float(answer_log_probs.mean().detach())
                    meters["kl_loss"] += float(kl_loss_part.detach()) * loss_scale_factor
                    meter_count += 1
                    print(f"micro-batch loss: {float(loss_chunk.detach()) / loss_scale_factor:.6f}")

                # End of mini-batch: step optimizer
                grad_norm = self._optimizer_step()
                meters["grad_norm"] += float(grad_norm.detach())
                self.actor_optimizer.zero_grad()

        print("number of responses has delimiter for this rank:", num_delim)

        # print raw losses
        print("="*20 + " JEPO Actor Losses " + "="*20)
        print(f"JEPO Actor raw_cot_loss: {meters['raw_cot_loss'] / max(meter_count, 1):.6f}")
        print(f"JEPO Actor raw_l2_loss: {meters['raw_l2_loss'] / max(meter_count, 1):.6f}")
        print(f"JEPO Actor raw_log_likelihood_loss: {meters['raw_log_likelihood_loss'] / max(meter_count, 1):.6f}")
        print(f"JEPO Actor raw_supp_loss: {meters['raw_supp_loss'] / max(meter_count, 1):.6f}")
        print(f"JEPO Actor raw_total_loss: {meters['raw_total_loss'] / max(meter_count, 1):.6f}")
        print(f"JEPO Actor extra_loss: {meters.get('extra_loss', 0.0) / max(meter_count, 1):.6f}")



        # average meters
        if meter_count > 0:
            for k in meters:
                meters[k] /= meter_count

        return {
            "jepo_actor/extra_loss": meters.get("extra_loss", 0.0),
            "jepo_actor/raw_cot_loss": meters["raw_cot_loss"],
            "jepo_actor/raw_total_loss": meters["raw_total_loss"],
            "jepo_actor/raw_kl_loss": meters["raw_kl_loss"],
            "jepo_actor/jepo_loss": meters["jepo_loss"],
            "jepo_actor/raw_supp_loss": meters["raw_supp_loss"],
            "jepo_actor/raw_l2_loss": meters["raw_l2_loss"],
            "jepo_actor/raw_log_likelihood_loss": meters["raw_log_likelihood_loss"],
            "jepo_actor/total_loss": meters["total_loss"],
            "jepo_actor/grad_norm": meters["grad_norm"],
            "jepo_actor/jepo_advs_mean": meters["jepo_advs_mean"],
            "jepo_actor/jepo_advs_std": meters["jepo_advs_std"],
            "jepo_actor/cot_log_probs_mean": meters["cot_log_probs_mean"],
            "jepo_actor/log_mean_answer_probs_mean": meters["log_mean_answer_probs_mean"],
            "jepo_actor/beta_supp": beta_supp,
            "jepo_actor/beta_supp_extra": beta_supp_extra,
            "jepo_actor/kl_loss": meters.get("kl_loss", 0.0),
            "jepo_actor/beta_kl": beta_kl,
            "jepo_buffer/num_has_delimiter": int(num_delim),
            "jepo_buffer/frac_has_delimiter": float(num_delim / max(num_responses, 1)),
            "jepo_buffer/frac_has_delimiter_global": frac_delim_glob,
            "jepo_buffer/format_adv_max": fmt_max,
            "jepo_actor/format_penalty": format_penalty,
        }

    

    @torch.no_grad()
    def _precompute_adv_w_with_verl(self, data: DataProto, temperature: float, format_penalty: float) -> DataProto:
        print("Precomputing JEPO advantages and weights with verl...")
        dev = data.batch["responses"].device
        pad_id = self._cached_tokenizer.pad_token_id if self._cached_tokenizer is not None else 0
        N = data.batch["responses"].shape[0]

        # NEW: Get vocab size for storing probability distributions
        vocab_size = self.actor_module.config.vocab_size if hasattr(self.actor_module, 'config') else 32000
        jepo_cfg = data.meta_info.get("jepo_config", {}) or {}
        store_last_token_probs = bool(jepo_cfg.get("store_last_token_probs", True))
        beta_supp = float(jepo_cfg.get("beta_supp", 0.001))

        # Outputs to fill (on device for final writeback)
        jepo_adv_raw = torch.zeros(N, device=dev)
        format_adv = torch.zeros(N, device=dev)
        jepo_weights = torch.zeros(N, device=dev)
        jepo_extra_weights = torch.zeros(N, device=dev)
        has_delim_all = torch.zeros(N, dtype=torch.bool, device=dev)
        gt_values_stored = torch.zeros(N, device=dev)
        
        # NEW: Store full probability distribution for last token if requested
        if store_last_token_probs:
            last_token_probs_all = torch.zeros((N, vocab_size), device=dev)
        else:
            last_token_probs_all = None

        # Rank gating for progress bars
        try:
            _is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
            _rank = torch.distributed.get_rank() if _is_dist else 0
        except Exception:
            _rank = 0

        # ---------------- Stage 1: no-grad dynamic microbatching via prepare_dynamic_batch ----------------
        jepo_cfg = data.meta_info.get("jepo_config", {}) or {}
        mini_bs = int(jepo_cfg.get("mini_batch_size_per_gpu", 8))
        micro_bs = int(jepo_cfg.get("micro_batch_size_per_gpu", 4))
        use_dynamic_bsz = bool(jepo_cfg.get("use_dynamic_bsz", True))
        max_token_len = (
            jepo_cfg.get("ppo_max_token_len_per_gpu", 16384)
            * getattr(self, "ulysses_sequence_parallel_size", 1)
        )
        # use_prob_as_reward
        use_prob_as_reward = bool(jepo_cfg.get("use_prob_as_reward", False))
        
        model_name = str(jepo_cfg.get("model_name", "unknown_model"))
        
        if 'qwen' in model_name.lower():
            token_to_digit = {16:1,17:2,18:3,19:4,20:5}
        elif 'mistral' in model_name.lower():
            token_to_digit = {28740:1, 28750:2, 28770:3, 28781:4, 28782:5}
        elif 'llama' in model_name.lower():
            token_to_digit =  {16:1,17:2,18:3,19:4,20:5}
        else:
            print("Unknown model for regression digit token ids, using default Mistral digit token ids.")
            token_to_digit = {28740:1, 28750:2, 28770:3, 28781:4, 28782:5}


        # Preallocate accumulators
        logp_sum_all = []  # accumulate per-mini then extend in order
        expected_values_all = []
        last_token_log_probs_all = []
        accs_all = []
        gts_all = []

        # One steady inner pbar across all microbatches (count responses, not just non-empty)
        # Option B: allow per-rank bars written to files to avoid console interleaving
        show_all_rank_pbar_to_file = bool(jepo_cfg.get("show_all_rank_pbar_to_file", False))
        pbar_file_dir = jepo_cfg.get("pbar_file_dir", "user_logs")
        _pbar_file_handle = None
        if show_all_rank_pbar_to_file:
            try:
                os.makedirs(pbar_file_dir, exist_ok=True)
                pbar_path = os.path.join(pbar_file_dir, f"jepo_tf_rank{_rank}.log")
                _pbar_file_handle = open(pbar_path, mode="w", buffering=1)
                _inner_pbar = tqdm(
                    total=N,
                    desc=f"R{_rank} TF answers",
                    leave=False,
                    disable=False,
                    file=_pbar_file_handle,
                    dynamic_ncols=False,
                    mininterval=0.2,
                )
            except Exception:
                raise RuntimeError(f"Cannot open JEPO pbar log file for rank {_rank} at {pbar_path}")
                _inner_pbar = tqdm(total=N, desc="Teacher-forced answers", leave=False, disable=(_rank != 0))
        else:
            _inner_pbar = tqdm(total=N, desc="Teacher-forced answers", leave=False, disable=(_rank != 0))

        # Cache original training mode and switch to eval
        prev_training = self.actor_module.training if hasattr(self, "actor_module") else False
        self.actor_module.eval()

        with torch.inference_mode():
            # Iterate like update_policy: first mini-batches, then micro-batches
            for mini_batch in data.split(mini_bs):
                Bmini = len(mini_batch)
                # prepare dynamic micro-batches following dp_actor
                if use_dynamic_bsz:
                    micro_batches, idx_lists = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = max(1, mini_bs // max(micro_bs, 1))
                    micro_batches = mini_batch.split(max(micro_bs, 1))
                    idx_lists = [
                        list(range(i * micro_bs, min((i + 1) * micro_bs, Bmini))) for i in range(len(micro_batches))
                    ]
                # per-mini accumulator in mini order
                logp_sum_mini = [0.0] * Bmini
                expected_values_mini = [0.0] * Bmini
                last_token_log_probs_mini = [0.0] * Bmini
                accs_mini = [0.0] * Bmini
                gts_mini = [None] * Bmini
                for micro_batch, idx_local in zip(micro_batches, idx_lists):
                    bs = len(idx_local)
                    if bs == 0:
                        raise RuntimeError("Zero-length micro-batch encountered in JEPO precompute")
                    # Build trimmed micro-batch tensors up to answer_end = ans_start + ans_len
                    ids_src = micro_batch.batch.get("batch_input_ids")
                    attn_src = micro_batch.batch.get("attention_mask")
                    pos_src = micro_batch.batch.get("position_ids")
                    ans_start = micro_batch.batch.get("answer_start_positions")
                    accs_mb = micro_batch.non_tensor_batch.get("acc", torch.zeros(bs, device=dev, dtype=torch.float32))
                    gt_tokens_mb = micro_batch.non_tensor_batch.get("ground_truth_answer_tokens", [])
                    
                    gt_tokens_len_mb = [
                        (int(len(t)) if (t is not None) else 0) for t in gt_tokens_mb
                    ]
                    if np.any(np.array(gt_tokens_len_mb) == 0):
                        gt_tokens_mb = torch.ones((bs, 1), dtype=torch.int64, device=dev)

                    # Compute per-row end lengths and maxima
                    ans_lens_mb = []
                    end_lens = []
                    for slot in range(bs):
                        gt_i = gt_tokens_mb[slot]
                        if isinstance(gt_i, (list, tuple)):
                            L_ans = int(len(gt_i))
                        elif isinstance(gt_i, np.ndarray):
                            L_ans = int(gt_i.size)
                        elif torch.is_tensor(gt_i):
                            L_ans = int(gt_i.numel())
                        else:
                            try:
                                L_ans = int(len(gt_i))
                            except Exception:
                                #L_ans = int(np.asarray(gt_i).size)
                                raise RuntimeError(f"Cannot interpret ground_truth_answer_tokens row {slot} of type {type(gt_i)}")
                        ans_lens_mb.append(L_ans)
                        end_lens.append(int(ans_start[slot].item() + L_ans) if L_ans > 0 else 0)
                    micro_max_len = int(max([e for e in end_lens if e > 0], default=0))
                    R_max = int(max(ans_lens_mb) if ans_lens_mb else 0)
                    R_max_flag = False
                    if micro_max_len == 0 or R_max == 0:
                        R_max_flag = True
                        _inner_pbar.update(bs)
                        #raise RuntimeError("Zero-length micro_max_len or R_max encountered in JEPO precompute")

                    ids_mb = torch.full((bs, micro_max_len), pad_id, dtype=ids_src.dtype, device=dev)
                    attn_mb = torch.zeros((bs, micro_max_len), dtype=attn_src.dtype, device=dev)
                    pos_mb = torch.zeros((bs, micro_max_len), dtype=pos_src.dtype, device=dev)
                    resp_mb = torch.full((bs, R_max), pad_id, dtype=ids_src.dtype, device=dev)

                    # --- Instrumentation: show per-microbatch lengths on tqdm (rank 0 only) ---
                    try:
                        # Answer lengths stats
                        ans_len_mean = float(np.mean(ans_lens_mb)) if ans_lens_mb else 0.0
                        ans_len_max = int(R_max)

                        # CoT lengths stats, if available
                        cot_lens_mb = None
                        if "cot_start_positions" in micro_batch.batch.keys():
                            cot_start = micro_batch.batch.get("cot_start_positions")
                            cot_lens_mb = [
                                max(0, int(ans_start[i].item()) - int(cot_start[i].item())) for i in range(bs)
                            ]
                        cot_len_mean = float(np.mean(cot_lens_mb)) if cot_lens_mb else 0.0
                        cot_len_max = int(max(cot_lens_mb)) if cot_lens_mb else 0

                        # Simple ASCII bars (scaled to micro_max_len to avoid huge bars)
                        def _mk_bar(val, vmax, width=20):
                            vmax = max(1, int(vmax))
                            n = max(0, min(width, int(round(width * float(val) / float(vmax)))))
                            return ("#" * n) + ("-" * (width - n))

                        ans_bar = _mk_bar(ans_len_mean, micro_max_len)
                        cot_bar = _mk_bar(cot_len_mean, micro_max_len)

                        # Decode one representative ground-truth answer (the row with max end length)
                        gt_preview = None
                        try:
                            # choose the heaviest row for preview
                            slot_show = 0
                            if end_lens:
                                slot_show = int(np.argmax(end_lens))
                            gt_i = gt_tokens_mb[slot_show]
                            # convert to list[int]
                            if isinstance(gt_i, (list, tuple)):
                                gt_ids = list(gt_i)
                            elif isinstance(gt_i, np.ndarray):
                                gt_ids = gt_i.astype(int).tolist()
                            elif torch.is_tensor(gt_i):
                                gt_ids = gt_i.to("cpu").tolist()
                            else:
                                gt_ids = list(gt_i)

                            if hasattr(self, "_cached_tokenizer") and self._cached_tokenizer is not None:
                                gt_text = self._cached_tokenizer.decode(gt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            else:
                                gt_text = str(gt_ids[:20])
                            # one-line, trimmed preview
                            gt_text = gt_text.replace("\n", " ").replace("\r", " ")
                            if len(gt_text) > 64:
                                gt_text = gt_text[:64] + "…"
                            gt_preview = gt_text
                        except Exception:
                            raise RuntimeError("Failed to decode ground_truth_answer_tokens for progress bar")

                        # Update postfix (rank 0 only; disabled elsewhere by tqdm constructor)
                        if gt_preview is not None and len(gt_preview) > 0:
                            _inner_pbar.set_postfix_str(
                                f"bs={bs} seq_max={micro_max_len} ans={ans_len_mean:.0f}/{ans_len_max} [{ans_bar}] "
                                f"cot={cot_len_mean:.0f}/{cot_len_max} [{cot_bar}] gt=\"{gt_preview}\""
                            )
                        else:
                            _inner_pbar.set_postfix_str(
                                f"bs={bs} seq_max={micro_max_len} ans={ans_len_mean:.0f}/{ans_len_max} [{ans_bar}] "
                                f"cot={cot_len_mean:.0f}/{cot_len_max} [{cot_bar}]"
                            )
                    except Exception:
                        # Never break training due to instrumentation
                        raise RuntimeError("Failed to decode ground_truth_answer_tokens for progress bar")

                    for slot in range(bs):
                        L_ans = int(ans_lens_mb[slot])
                        if L_ans <= 0:
                            continue
                        L_end = int(ans_start[slot].item() + L_ans)
                        ids_i = ids_src[slot, :L_end]
                        l_i = int(ids_i.size(0))
                        ids_mb[slot, :l_i] = ids_i
                        # copy attention from the source slice to respect eos/pad masking
                        attn_mb[slot, :l_i] = attn_src[slot, :l_i]

                        gt_i = gt_tokens_mb[slot]
                        if isinstance(gt_i, (list, tuple)):
                            gt_i_t = torch.tensor(gt_i, dtype=ids_src.dtype, device=dev)
                        elif isinstance(gt_i, torch.Tensor):
                            gt_i_t = gt_i.to(device=dev, dtype=ids_src.dtype)
                        else:
                            gt_i_t = torch.tensor(list(gt_i), dtype=ids_src.dtype, device=dev)
                        resp_mb[slot, :L_ans] = gt_i_t[:L_ans] 
                        
                        
                        print('ids_mb[slot, l_i - 1]', ids_mb[slot, l_i - 1], 'gt_i_t[0]', gt_i_t[0])
                         
                        # ids_mb[slot, l_i - 1] = gt_i_t[0]
                        
                        
                        
                        # breakpoint()


                    # Derive positions from attention so first attended token starts at 0
                    if attn_mb.numel() > 0:
                        pos_from_mask = (attn_mb.cumsum(dim=1) - 1).clamp_min(0) * attn_mb
                        pos_mb[:, :] = pos_from_mask.to(dtype=pos_mb.dtype)

                    micro_inputs = {
                        "input_ids": ids_mb,
                        "attention_mask": attn_mb,
                        "position_ids": pos_mb,
                        "responses": resp_mb,
                    }

                    if R_max_flag:
                        
                        _, lp_mb = self._forward_micro_batch(
                            micro_inputs, temperature=temperature, calculate_entropy=False
                        )
                        lp_mb = lp_mb * 0.0
                        # Set expected_values to zeros if R_max_flag and store_last_token_probs
                        if store_last_token_probs:
                            expected_values = torch.zeros(bs, device=dev, dtype=torch.float32)
                    else:
                       
                        # NEW (regression setting): If we need full probability distributions, get logits too
                        if store_last_token_probs:
                           if use_prob_as_reward:
                                _, lp_mb, expected_values, last_token_log_probs  = self._forward_micro_batch(
                                    micro_inputs, temperature=temperature, calculate_entropy=False, regression=True, expected_prob_replace=True)
                           else:
                               _, lp_mb, expected_values  = self._forward_micro_batch(
                                    micro_inputs, temperature=temperature, calculate_entropy=False, regression=True)
                               last_token_log_probs = torch.zeros(bs, device=dev, dtype=torch.float32)
                               
                        else:


                            # Original: just get log probs
                            _, lp_mb = self._forward_micro_batch(
                                micro_inputs, temperature=temperature, calculate_entropy=False
                            )

                    # Accumulate per-row sums into mini-batch order via idx_local
                    lp_mb_cpu = lp_mb.detach().to("cpu")
                    for slot, mini_pos in enumerate(idx_local):
                        L_ans = int(ans_lens_mb[slot])
                        if L_ans > 0:
                            logp_sum_mini[mini_pos] += float(lp_mb_cpu[slot, :L_ans].sum().item())
                    _inner_pbar.update(bs)

                    # Accumulate per-row sums into mini-batch order via idx_local for expected values
                    if store_last_token_probs:
                        expected_values_cpu = expected_values.detach().to("cpu")
                        last_token_log_probs_cpu = last_token_log_probs.detach().to("cpu")
                        
                        for slot, mini_pos in enumerate(idx_local):
                            L_ans = int(ans_lens_mb[slot])
                            if L_ans > 0:
                                expected_values_mini[mini_pos] = float(expected_values_cpu[slot].item())
                                last_token_log_probs_mini[mini_pos] = float(last_token_log_probs_cpu[slot].item())
                    
                    # Accumulate accs and gts
                    accs_mb_cpu = accs_mb
                    for slot, mini_pos in enumerate(idx_local):
                        accs_mini[mini_pos] = float(accs_mb_cpu[slot].item())
                        gts_mini[mini_pos] = gt_tokens_mb[slot]
                    

                # Append this mini-batch results to the global list in order
                expected_values_all.extend(expected_values_mini)
                last_token_log_probs_all.extend(last_token_log_probs_mini)
                accs_all.extend(accs_mini)
                gts_all.extend(gts_mini)
                logp_sum_all.extend(logp_sum_mini)
        _inner_pbar.close()
        try:
            if _pbar_file_handle is not None:
                _pbar_file_handle.flush()
                _pbar_file_handle.close()
        except Exception:
            raise RuntimeError("Failed to close JEPO precompute pbar file handle")
        if prev_training:
            self.actor_module.train()

        # ---------------- Stage 2: advantage/weight computation per UID group ----------------
        # NEW: Check if we should use regression-based advantages
        use_regression_reward = bool(jepo_cfg.get("use_regression_reward", True))
        
        # Build uid groups preserving order of first appearance
        uids = data.non_tensor_batch.get("uid")
        if isinstance(uids, torch.Tensor):
            uids_list = [str(x) for x in uids.cpu().tolist()]
        else:
            uids_list = [str(x) for x in (uids.tolist() if hasattr(uids, "tolist") else list(uids))]

        ans_logprob_all = torch.as_tensor(logp_sum_all, device=dev, dtype=torch.float32)
        expected_values_all = torch.as_tensor(expected_values_all, device=dev, dtype=torch.float32)  
        last_token_log_probs_all = torch.as_tensor(last_token_log_probs_all, device=dev, dtype=torch.float32)  
        accs_all = torch.as_tensor(accs_all, device=dev, dtype=torch.float32)
        gts_all = torch.as_tensor(gts_all, device=dev, dtype=torch.float32)
        has_delim_src = data.batch.get("has_delimiter")
        has_delim_all = has_delim_src.clone() if isinstance(has_delim_src, torch.Tensor) else torch.as_tensor(has_delim_src, device=dev, dtype=torch.bool)

        # Get ground truth answers for regression reward
        if use_regression_reward:
            gt_answers = data.non_tensor_batch.get("reward_model", {})
            gt_values = []
            for gt_dict in gt_answers:
                gt_str = gt_dict.get("ground_truth", "0")
                try:
                    # Extract numeric value from ground truth string
                    # Handle formats like "42", "\\boxed{42}", etc.
                    import re
                    numbers = re.findall(r'\d+', str(gt_str))
                    gt_val = float(numbers[0]) if numbers else 0.0
                except:
                    gt_val = 0.0
                gt_values.append(gt_val)
            gt_values_tensor = torch.tensor(gt_values, device=dev, dtype=torch.float32)

        groups = {}
        order = []
        for idx, u in enumerate(uids_list):
            if u not in groups:
                groups[u] = []
                order.append(u)
            groups[u].append(idx)

        _outer_pbar = tqdm(total=len(order), desc="JEPO precompute: prompts", disable=(_rank != 0))
        for u in order:
            idxs = groups[u]
            B = len(idxs)
            
            # Initialize A_prob for all code paths
            A_prob = torch.zeros(B, device=dev, dtype=torch.float32)
            A_acc = torch.zeros(B, device=dev, dtype=torch.float32)
            
            if use_regression_reward and store_last_token_probs:
                expected_values = expected_values_all[idxs]  # [B]
          
                # Get ground truth value for this group
                # gt_value = gt_values_tensor[idxs[0]]  # All samples in group have same GT
                
                gt_values_token = gts_all[idxs[0]]  # list of ground truth token sequences
                
                
                
                gt_value = token_to_digit.get(int(gt_values_token[0].item()), 0)
                
                # breakpoint()
                
                # Compute rewards: R_i = -(E[digit]_i - y)^2
                squared_errors = (expected_values - gt_value) ** 2
                rewards = -squared_errors  # Negative because we want to minimize error
                
               
                # Compute advantages using leave-one-out baseline
                if B > 1:
                    # Leave-one-out mean reward
                    total_reward = rewards.sum()
                    loo_mean_rewards = (total_reward - rewards) / (B - 1)
                    A_raw = rewards - loo_mean_rewards
                else:
                    # Single sample: no baseline
                    A_raw = rewards - rewards.mean()  # Zero-centered
                
                # Normalize advantages
                if bool(jepo_cfg.get("normalize_advantages", True)):
                    A_raw = A_raw / (A_raw.std(unbiased=False) + 1e-8)
                A_raw = A_raw.clamp(-1.0, 1.0)

                if _rank == 0:  # Only first 3 groups to avoid spam
                    print(f"\n[DEBUG Regression] UID: {u}")
                    print(f"  Group size (B): {B}")
                    print(f"  Ground truth: {gt_value:.2f}")
                    print(f"  Expected values: {expected_values.cpu().numpy()}")
                    print(f"  Squared errors: {squared_errors.cpu().numpy()}")
                    print(f"  Rewards (neg squared errors): {rewards.cpu().numpy()}")
                    print(f"  Digit probs for first response:")
                    print(f'LOO mean rewards: {loo_mean_rewards.cpu().numpy()}' if B > 1 else "  LOO mean rewards: N/A (B=1)")
                    # print A_raw
                    print(f'Advantages A_raw: {A_raw.cpu().numpy()}')
                    # for k in range(10):
                    #     print(f"    P({k}) = {digit_probs[0, k].item():.4f}")
                    print(f"  E[digit] = {expected_values[0].item():.4f}")
                
                # Compute weights based on rewards (higher reward = higher weight)
                # Use softmax on rewards (not log-probs)
                # w_full = torch.softmax(rewards, dim=0)
                w_full = - 2 * (expected_values - gt_value) 
                
                # Format advantage based on valid last token (1-5 check)
                has_delim = has_delim_all[idxs]
                fmt = torch.where(
                    has_delim,
                    torch.zeros(B, device=dev, dtype=torch.float32),
                    torch.tensor(-float(format_penalty), device=dev, dtype=torch.float32),
                )
                fmt = fmt - fmt.mean()

                ans_logprob = ans_logprob_all[idxs]
                extra_w_full = torch.softmax(ans_logprob, dim=0)
                
                
                
            
                #################### calculate the accuracy reward ####################
                last_token_log_probs = last_token_log_probs_all[idxs] if last_token_log_probs_all is not None else None
                if last_token_log_probs is not None:
                    rewards = torch.exp(last_token_log_probs)  # Convert log-probs to probs, [B]
                    if B > 1:
                        # Leave-one-out mean reward
                        total_reward = rewards.sum()
                        loo_mean_rewards = (total_reward - rewards) / (B - 1)
                        A_prob = rewards - loo_mean_rewards
                    else:
                        # Single sample: no baseline
                        A_prob = rewards - rewards.mean()  # Zero-centered
                    
                    ###########################################
                    
                    # # Groupwise math (identical)
                    # lse_all = torch.logsumexp(last_token_log_probs, dim=0)
                    # if B > 1:
                    #     d = last_token_log_probs - lse_all
                    #     lse_others = lse_all + torch.log((-torch.expm1(d)).clamp_min(1e-12))
                    #     v_i = lse_others - math.log(B - 1)
                    # else:
                    #     v_i = ans_logprob.new_full((B,), float("-inf"))
                    # log_mean = lse_all - math.log(max(B, 1))
                    
                    # A_prob = (log_mean - v_i)
                                     
                    # Normalize advantages
                    if bool(jepo_cfg.get("normalize_advantages", True)):
                        A_prob = A_prob / (A_prob.std(unbiased=False) + 1e-8)
                    A_prob = A_prob.clamp(-1.0, 1.0)
                    
                    ###############################################
                    
                    if _rank == 0:  # Only first 3 groups to avoid spam
                        print(f"\n[DEBUG Regression - Accuracy Reward] UID: {u}")
                        print(f"  Last token probs: {last_token_log_probs.cpu().numpy()}")
                        # print prob, exp(log_prob)
                        probs = torch.exp(last_token_log_probs)
                        print(f"  Last token probabilities: {probs.cpu().numpy()}")
                        print(f'  LOO mean rewards: {loo_mean_rewards.cpu().numpy()}' if B > 1 else "  LOO mean rewards: N/A (B=1)")
                        print(f'  Advantages A_prob: {A_prob.cpu().numpy()}')
                        
                        
                # # use accuracy as extra advantage
                # rewards_acc = accs_all[idxs]  # [B]
                
                # if B > 1:
                #     # Leave-one-out mean reward
                #     total_reward_acc = rewards_acc.sum()
                #     loo_mean_rewards_acc = (total_reward_acc - rewards_acc) / (B - 1)
                #     A_acc = rewards_acc - loo_mean_rewards_acc
                # else:
                #     # Single sample: no baseline
                #     A_acc = rewards_acc - rewards_acc.mean()  # Zero-centered
                # # Normalize advantages
                # if bool(jepo_cfg.get("normalize_advantages", True)):
                #     A_acc = A_acc / (A_acc.std(unbiased=False) + 1e-8)
                # A_acc = A_acc.clamp(-1.0, 1.0)  

                # if _rank == 0:  # Only first 3 groups to avoid spam
                #     print(f"\n[DEBUG Regression - Accuracy Advantage] UID: {u}")
                #     print(f"  Accuracies: {rewards_acc.cpu().numpy()}")
                #     print(f'  LOO mean rewards (acc): {loo_mean_rewards_acc.cpu().numpy()}' if B > 1 else "  LOO mean rewards (acc): N/A (B=1)")
                #     print(f'  Advantages A_acc: {A_acc.cpu().numpy()}')
            
                    
                        
                

                
            else:
                # ============ ORIGINAL: Log-probability-based advantage computation ============
                ans_logprob = ans_logprob_all[idxs]
                
                # Groupwise math (identical)
                lse_all = torch.logsumexp(ans_logprob, dim=0)
                if B > 1:
                    d = ans_logprob - lse_all
                    lse_others = lse_all + torch.log((-torch.expm1(d)).clamp_min(1e-12))
                    v_i = lse_others - math.log(B - 1)
                else:
                    v_i = ans_logprob.new_full((B,), float("-inf"))
                log_mean = lse_all - math.log(max(B, 1))
                
                A_raw = (log_mean - v_i)
                A_raw = A_raw / (A_raw.std(unbiased=False) + 1e-8)
                A_raw = A_raw.clamp(-1.0, 1.0)

                has_delim = has_delim_all[idxs]
                fmt = torch.where(
                    has_delim,
                    torch.zeros(B, device=dev, dtype=torch.float32),
                    torch.tensor(-float(format_penalty), device=dev, dtype=torch.float32),
                )
                fmt = fmt - fmt.mean()
                w_full = torch.softmax(ans_logprob, dim=0)

            jepo_adv_raw[idxs] = A_raw + beta_supp * A_prob 
            format_adv[idxs] = fmt
            jepo_weights[idxs] = w_full
            jepo_extra_weights[idxs] = extra_w_full if use_regression_reward else torch.zeros_like(w_full)
            
            if use_regression_reward:
                gt_values_stored[idxs] = gt_value  

            _outer_pbar.update(1)
        _outer_pbar.close()

        # Write back to batch
        data.batch["jepo_adv_raw"] = jepo_adv_raw
        data.batch["format_adv"] = format_adv
        data.batch["jepo_weights"] = jepo_weights
        data.batch["has_delimiter"] = has_delim_all
        data.batch["jepo_extra_weights"] = jepo_extra_weights
        data.batch["gt_values"] = gt_values_stored
        
        # NEW: Write back last token probabilities if collected
        if store_last_token_probs and last_token_probs_all is not None:
            data.batch["last_token_probs"] = last_token_probs_all
            if _rank == 0:
                print(f"\n[DEBUG Summary] Stored last_token_probs with shape: {last_token_probs_all.shape}")
        
        # DEBUG: Print precompute summary statistics
        if _rank == 0:
            print(f"\n[DEBUG Precompute Summary]")
            print(f"  Total responses (N): {N}")
            print(f"  Total unique prompts: {len(order)}")
            print(f"  Use regression reward: {use_regression_reward}")
            print(f"  Store last token probs: {store_last_token_probs}")
            print(f"  JEPO advantages - mean: {jepo_adv_raw.mean().item():.4f}, std: {jepo_adv_raw.std().item():.4f}")
            print(f"  Format advantages - mean: {format_adv.mean().item():.4f}, std: {format_adv.std().item():.4f}")
            print(f"  Has delimiter (valid format) - count: {has_delim_all.sum().item()}/{N} ({100*has_delim_all.float().mean().item():.1f}%)")
            if use_regression_reward:
                print(f"  Weights - min: {jepo_weights.min().item():.4f}, max: {jepo_weights.max().item():.4f}, mean: {jepo_weights.mean().item():.4f}")
                print(f"  Extra Weights - min: {jepo_extra_weights.min().item():.4f}, max: {jepo_extra_weights.max().item():.4f}, mean: {jepo_extra_weights.mean().item():.4f}")
                print(f"  GT values stored - min: {gt_values_stored.min().item():.4f}, max: {gt_values_stored.max().item():.4f}, mean: {gt_values_stored.mean().item():.4f}")
            
        
        return data

    

