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

        # -------- build teacher-forced packs per question --------
        data = attach_jepo_adv_to_dataproto(
            data=data,
            model=self.actor_module,
            jepo_cfg=jepo_cfg,
            cached_tokenizer=self._cached_tokenizer
        )
        # Precompute A_raw, format_adv, and weights using VERL internals on teacher-forced batches
        if "jepo_data_dicts" in data.non_tensor_batch:
            data = self._precompute_adv_w_with_verl(data, temperature=temperature, format_penalty=format_penalty)
        
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
                        
                        
                        # Build GRPO-style token-level loss using VERL internals (_forward_micro_batch)
                        # Mask A_raw by has_delimiter and add mean-centered format advantage
                        A_vec = (
                            jepo_adv_raw_all[step_indices] * has_delimiter_mask[step_indices].float()
                            + format_adv_all[step_indices]
                        )
                        was_training_actor = self.actor_module.training
                        self.actor_module.eval()
                        dev = step_data.batch["batch_input_ids"].device
                        loss_agg_mode = "token-mean"

                        pad_id = self._cached_tokenizer.pad_token_id if self._cached_tokenizer is not None else 0

                        # Precompute lengths
                        a_s_all = step_data.batch["answer_start_positions"].tolist()
                        c_s_all = step_data.batch["cot_start_positions"].tolist()
                        ids_full_all = step_data.batch["batch_input_ids"]
                        attn_full_all = step_data.batch["attention_mask"]
                        pos_full_all = step_data.batch["position_ids"]
                        gt_tokens_all = step_data.non_tensor_batch.get("ground_truth_answer_tokens", [None] * B_step)

                        cot_len_all = [max(0, int(a - c)) for a, c in zip(a_s_all, c_s_all)]
                        ans_len_all = [int(len(t)) if t is not None else max(0, int(ids_full_all[i].numel()) - int(a_s_all[i])) for i, t in enumerate(gt_tokens_all)]

                        max_cot_len = max(cot_len_all) if cot_len_all else 0
                        max_ans_len = max(ans_len_all) if ans_len_all else 0

                        # Initialize combined mats
                        cot_lp_mat = torch.zeros((B_step, max_cot_len), device=dev)
                        cot_mask = torch.zeros((B_step, max_cot_len), device=dev)
                        cot_adv_mat = torch.zeros((B_step, max_cot_len), device=dev)

                        ans_lp_mat = torch.zeros((B_step, max_ans_len), device=dev)
                        ans_mask = torch.zeros((B_step, max_ans_len), device=dev)
                        ans_adv_mat = torch.zeros((B_step, max_ans_len), device=dev)

                        # Batch CoT over sub-chunks
                        for s in range(0, B_step, resp_micro_bs):
                            e = min(s + resp_micro_bs, B_step)
                            rows = list(range(s, e))
                            # Build sub-batch for rows with non-zero CoT length
                            cot_rows = [r for r in rows if cot_len_all[r] > 0]
                            if not cot_rows:
                                continue
                            # Build ragged lists
                            in_ids_list, in_attn_list, in_pos_list, resp_list = [], [], [], []
                            for r in cot_rows:
                                a = int(a_s_all[r]); c = int(c_s_all[r])
                                in_ids_list.append(ids_full_all[r][:a])
                                in_attn_list.append(attn_full_all[r][:a])
                                in_pos_list.append(pos_full_all[r][:a])
                                resp_list.append(ids_full_all[r][c:a])
                            L_max = max(int(x.size(0)) for x in in_ids_list)
                            R_max = max(int(x.size(0)) for x in resp_list)
                            bsz = len(cot_rows)
                            # Pad and stack
                            def _pad1d_list(lst, fill_val):
                                out = torch.full((bsz, L_max), fill_val, dtype=lst[0].dtype, device=dev)
                                for i, t in enumerate(lst):
                                    l = int(t.size(0)); out[i, :l] = t
                                return out
                            def _pad1d_resp(lst, fill_val):
                                out = torch.full((bsz, R_max), fill_val, dtype=lst[0].dtype, device=dev)
                                for i, t in enumerate(lst):
                                    l = int(t.size(0)); out[i, :l] = t
                                return out
                            in_ids = _pad1d_list(in_ids_list, pad_id)
                            in_attn = _pad1d_list(in_attn_list, 0)
                            in_pos = _pad1d_list(in_pos_list, 0)
                            resp_tok = _pad1d_resp(resp_list, pad_id)
                            micro_cot = {
                                "input_ids": in_ids,
                                "attention_mask": in_attn,
                                "position_ids": in_pos,
                                "responses": resp_tok,
                            }
                            _, lp_cot_sub = self._forward_micro_batch(micro_cot, temperature=temperature, calculate_entropy=False)
                            # Scatter back
                            for j, r in enumerate(cot_rows):
                                Lr = cot_len_all[r]
                                cot_lp_mat[r, :Lr] = lp_cot_sub[j, :Lr]
                                cot_mask[r, :Lr] = 1
                                cot_adv_mat[r, :Lr] = A_vec[r]

                        # Batch Answer over sub-chunks
                        for s in range(0, B_step, resp_micro_bs):
                            e = min(s + resp_micro_bs, B_step)
                            rows = list(range(s, e))
                            ans_rows = [r for r in rows if ans_len_all[r] > 0]
                            if not ans_rows:
                                continue
                            ids_batch = ids_full_all[ans_rows]
                            attn_batch = attn_full_all[ans_rows]
                            pos_batch = pos_full_all[ans_rows]
                            resp_list = []
                            R_max = 0
                            for r in ans_rows:
                                a = int(a_s_all[r]); Lr = ans_len_all[r]
                                t = ids_full_all[r][a:a+Lr]
                                resp_list.append(t)
                                R_max = max(R_max, int(t.size(0)))
                            if R_max == 0:
                                continue
                            resp_tok = torch.full((len(ans_rows), R_max), pad_id, dtype=ids_batch.dtype, device=dev)
                            for j, t in enumerate(resp_list):
                                l = int(t.size(0)); resp_tok[j, :l] = t
                            micro_ans = {
                                "input_ids": ids_batch,
                                "attention_mask": attn_batch,
                                "position_ids": pos_batch,
                                "responses": resp_tok,
                            }
                            _, lp_ans_sub = self._forward_micro_batch(micro_ans, temperature=temperature, calculate_entropy=False)
                            for j, r in enumerate(ans_rows):
                                Lr = ans_len_all[r]
                                ans_lp_mat[r, :Lr] = lp_ans_sub[j, :Lr]
                                ans_mask[r, :Lr] = 1
                                ans_adv_mat[r, :Lr] = 1.0  # uniform support

                        if was_training_actor:
                            self.actor_module.train()

                        # Compute policy losses via GPG (token-wise -logp*A)
                        gpg_fn = get_policy_loss_fn("gpg")
                        if max_cot_len > 0 and cot_mask.any():
                            jepo_loss_part, _, _, _ = gpg_fn(
                                old_log_prob=None,
                                log_prob=cot_lp_mat,
                                advantages=cot_adv_mat,
                                response_mask=cot_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            cot_log_probs = cot_lp_mat.sum(dim=-1).detach()
                        else:
                            jepo_loss_part = torch.tensor(0.0, device=dev)
                            cot_log_probs = torch.zeros((B_step,), device=dev)

                        if max_ans_len > 0 and ans_mask.any():
                            supp_loss_part, _, _, _ = gpg_fn(
                                old_log_prob=None,
                                log_prob=ans_lp_mat,
                                advantages=ans_adv_mat * float(beta_supp),
                                response_mask=ans_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            answer_log_probs = ans_lp_mat.sum(dim=-1).detach()
                        else:
                            supp_loss_part = torch.tensor(0.0, device=dev)
                            answer_log_probs = torch.zeros((B_step,), device=dev)

                        # KL divergence on raw responses (use precomputed ref_log_prob from batch)
                        kl_loss_part = torch.tensor(0.0, device=dev)
                        if beta_kl > 0 and ("ref_log_prob" in data.batch.keys()):
                            # compute actor log_prob on original responses for this step
                            try:
                                micro_orig = {
                                    "input_ids": step_data.batch["input_ids"],
                                    "attention_mask": step_data.batch["attention_mask"],
                                    "position_ids": step_data.batch["position_ids"],
                                    "responses": step_data.batch["responses"],
                                }
                                _, lp_orig = self._forward_micro_batch(micro_orig, temperature=temperature, calculate_entropy=False)
                                ref_lp_orig = data.batch["ref_log_prob"][step_indices]
                                resp_mask_orig = compute_response_mask(step_data)
                                kld = kl_penalty(logprob=lp_orig, ref_logprob=ref_lp_orig, kl_penalty=kl_loss_type)
                                kl_loss_part = agg_loss(loss_mat=kld, loss_mask=resp_mask_orig, loss_agg_mode=loss_agg_mode) * beta_kl
                            except Exception:
                                kl_loss_part = torch.tensor(0.0, device=dev)
                        
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

    @torch.no_grad()
    def _precompute_adv_w_with_verl(self, data: DataProto, temperature: float, format_penalty: float) -> DataProto:
        dev = data.batch["responses"].device
        pad_id = self._cached_tokenizer.pad_token_id if self._cached_tokenizer is not None else 0
        data_dicts = data.non_tensor_batch.get("jepo_data_dicts", [])
        N = data.batch["responses"].shape[0]
        jepo_adv_raw = torch.zeros(N, device=dev)
        format_adv = torch.zeros(N, device=dev)
        jepo_weights = torch.zeros(N, device=dev)
        has_delim_all = torch.zeros(N, dtype=torch.bool, device=dev)

        current_idx = 0
        for dd in data_dicts:
            B = len(dd["has_delimiter"]) if isinstance(dd["has_delimiter"], list) else int(dd["has_delimiter"].numel())
            ans_logprob = torch.zeros(B, device=dev)
            for i in range(B):
                a_s = int(dd["answer_start_positions"][i])
                ids_full = dd["batch_input_ids"][i]
                attn_full = dd["attention_mask"][i]
                pos_full = dd["position_ids"][i]
                gt_row = dd["ground_truth_answer_tokens"][i]
                if isinstance(gt_row, list):
                    ans_len = len(gt_row)
                else:
                    ans_len = int((torch.as_tensor(gt_row, device=dev) != pad_id).sum().item())
                if ans_len <= 0:
                    continue
                micro_ans = {
                    "input_ids": ids_full.unsqueeze(0),
                    "attention_mask": attn_full.unsqueeze(0),
                    "position_ids": pos_full.unsqueeze(0),
                    "responses": ids_full[a_s:a_s + ans_len].unsqueeze(0),
                }
                _, lp_ans = self._forward_micro_batch(micro_ans, temperature=temperature, calculate_entropy=False)
                ans_logprob[i] = lp_ans.sum().detach()

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

            has_delim = torch.as_tensor(dd["has_delimiter"], device=dev, dtype=torch.bool)
            fmt = torch.where(
                has_delim,
                torch.zeros(B, device=dev, dtype=torch.float32),
                torch.tensor(-float(format_penalty), device=dev, dtype=torch.float32),
            )
            fmt = fmt - fmt.mean()
            w_full = torch.softmax(ans_logprob, dim=0)

            jepo_adv_raw[current_idx:current_idx + B] = A_raw
            format_adv[current_idx:current_idx + B] = fmt
            jepo_weights[current_idx:current_idx + B] = w_full
            has_delim_all[current_idx:current_idx + B] = has_delim
            current_idx += B

        data.batch["jepo_adv_raw"] = jepo_adv_raw
        data.batch["format_adv"] = format_adv
        data.batch["jepo_weights"] = jepo_weights
        data.batch["has_delimiter"] = has_delim_all
        return data
