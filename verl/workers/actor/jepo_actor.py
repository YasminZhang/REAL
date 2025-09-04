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
from verl.utils.seqlen_balancing import (
    prepare_dynamic_batch,
    restore_dynamic_batch,
    get_seqlen_balanced_partitions,
    ceildiv,
)
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


# -------- training loop using token-capped micro-batches and single forward --------
def _pack_rows_by_token_budget(row_indices, prefix_lens, cot_lens, ans_lens, max_tokens, max_rows=None):
    packs = []
    cur = []
    cur_tokens = 0
    for r in row_indices:
        t = int(prefix_lens[r]) + int(cot_lens[r]) + int(ans_lens[r])
        # if any single row exceeds budget, put it alone
        if t > max_tokens and not cur:
            packs.append([r])
            continue
        # if adding this row exceeds budget or row cap, flush
        if (cur and (cur_tokens + t > max_tokens)) or (max_rows is not None and len(cur) >= max_rows):
            packs.append(cur)
            cur = []
            cur_tokens = 0
        cur.append(r)
        cur_tokens += t
    if cur:
        packs.append(cur)
    return packs

def _pack_rows_by_dynamic_balance(row_indices, prefix_lens, cot_lens, ans_lens, max_tokens, max_rows=None):
    """
    Build packs using the same balancing idea behind prepare_dynamic_batch
    but with JEPO's effective token cost per row: prefix + cot + answer.

    Note: This does not strictly enforce a hard per-pack token cap; it
    mirrors prepare_dynamic_batch's strategy of choosing the number of
    micro-batches via ceildiv(total_tokens, max_tokens), then balances.
    """
    if not row_indices:
        return []
    # Compute per-row effective lengths
    eff = [int(prefix_lens[r]) + int(cot_lens[r]) + int(ans_lens[r]) for r in row_indices]
    total = sum(eff)
    # Choose number of partitions similar to prepare_dynamic_batch
    num_micro = max(1, min(len(row_indices), ceildiv(total, max_tokens)))
    # Use seqlen balancer to partition indices
    partitions = get_seqlen_balanced_partitions(eff, num_micro, equal_size=False)
    # Map back to original row indices
    packs = [[row_indices[i] for i in part] for part in partitions]
    # Optional row cap per pack
    if max_rows is not None and max_rows > 0:
        capped = []
        for p in packs:
            if len(p) <= max_rows:
                capped.append(p)
            else:
                for s in range(0, len(p), max_rows):
                    capped.append(p[s : s + max_rows])
        packs = capped
    return packs


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
        micro_bs = int(jepo_cfg.get("micro_batch_size_per_gpu", 4))      # questions per backward call
        format_penalty = float(jepo_cfg.get("format_penalty", 0.0))
        beta_supp = float(jepo_cfg.get("beta_supp", 0.001))
        beta_kl = float(jepo_cfg.get("beta_kl", 0.0))
        kl_loss_type = getattr(self.config, "kl_loss_type", "low_var_kl")
        temperature = float(data.meta_info["temperature"])

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
            data = self._precompute_adv_w_with_verl(data, temperature=temperature, format_penalty=format_penalty)
            # Drop grouping artifact to allow generic per-response slicing downstream
            try:
                del data.non_tensor_batch["jepo_data_dicts"]
            except Exception:
                pass
        
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
        max_token_len = (getattr(jepo_cfg, "ppo_max_token_len_per_gpu", 16384) * getattr(self, "ulysses_sequence_parallel_size", 1))
        resp_row_cap = None

        rows_all = list(range(num_responses))
        # Pre-gather full tensors once
        dev_all = data.batch["batch_input_ids"].device
        pad_id = self._cached_tokenizer.pad_token_id if self._cached_tokenizer is not None else 0
        ids_full_all = data.batch["batch_input_ids"]
        attn_full_all = data.batch["attention_mask"]
        pos_full_all = data.batch["position_ids"]
        a_s_all = data.batch["answer_start_positions"].tolist()
        c_s_all = data.batch["cot_start_positions"].tolist()
        gt_tokens_nd = data.non_tensor_batch.get("ground_truth_answer_tokens", None)
        if gt_tokens_nd is None:
            gt_tokens_all = [None] * num_responses
        else:
            try:
                import numpy as _np
                gt_tokens_all = gt_tokens_nd.tolist() if isinstance(gt_tokens_nd, _np.ndarray) else gt_tokens_nd
            except Exception:
                gt_tokens_all = gt_tokens_nd
        # A, weights per response
        A_all = (jepo_adv_raw_all + format_adv_all)
        w_all = data.batch.get("jepo_weights", torch.ones((num_responses,), device=dev_all))

        cot_lens = [max(0, int(a - c)) for a, c in zip(a_s_all, c_s_all)]
        ans_lens = [int(len(t)) if t is not None else max(0, int(ids_full_all[i].numel()) - int(a_s_all[i])) for i, t in enumerate(gt_tokens_all)]
        prefix_lens = [int(c) for c in c_s_all]

        for _ in range(epochs):
            self.actor_optimizer.zero_grad()
            self.actor_module.train()

            # Iterate by mini-batches of size `mini_bs` (dp_actor style)
            for mb_start in range(0, len(rows_all), max(mini_bs, 1)):
                mb_rows = rows_all[mb_start : mb_start + max(mini_bs, 1)]

                # Build micro packs inside this mini-batch
                if use_dynamic_bsz:
                    use_balancer = bool(jepo_cfg.get("use_dynamic_balancer", False))
                    if use_balancer:
                        packs = _pack_rows_by_dynamic_balance(
                            mb_rows, prefix_lens, cot_lens, ans_lens, max_token_len, resp_row_cap
                        )
                    else:
                        packs = _pack_rows_by_token_budget(
                            mb_rows, prefix_lens, cot_lens, ans_lens, max_token_len, resp_row_cap
                        )
                else:
                    # fixed-size micro packs inside this mini-batch
                    packs = [mb_rows[i : i + micro_bs] for i in range(0, len(mb_rows), micro_bs)]

                # For non-dynamic path, use fixed grad accumulation like dp_actor
                if not use_dynamic_bsz:
                    grad_accum = max(1, mini_bs // max(micro_bs, 1))

                for pi, pack_rows in enumerate(packs):
                    if len(pack_rows) == 0:
                        continue
                    # FSDP sync: only sync on last micro in the mini-batch
                    is_last_in_mb = (pi == len(packs) - 1)
                    sync_ctx = (
                        self.actor_module.no_sync()
                        if isinstance(self.actor_module, (FSDP, FSDPModule)) and not is_last_in_mb
                        else nullcontext()
                    )
                    with sync_ctx:
                        Bp = len(pack_rows)
                        # Build combined teacher-forced inputs: prefix up to cot_start, then [CoT|Answer]; right-pad both dimensions
                        # Compute per-row sizes in this pack
                        pre_ls = [prefix_lens[r] for r in pack_rows]
                        cot_ls = [cot_lens[r] for r in pack_rows]
                        ans_ls = [ans_lens[r] for r in pack_rows]
                        R_ls = [cot_ls[i] + ans_ls[i] for i in range(Bp)]
                        P_max = max(pre_ls) if pre_ls else 0
                        R_max = max(R_ls) if R_ls else 0

                        if P_max == 0 and R_max == 0:
                            # do a cheap dummy backward to preserve FSDP sync count
                            dummy_backward_fsdp_safe(self.actor_module, scaler=None)
                            continue

                        # Allocate tensors
                        ids_pack = torch.full((Bp, P_max + R_max), pad_id, dtype=ids_full_all.dtype, device=dev_all)
                        attn_pack = torch.zeros((Bp, P_max + R_max), dtype=attn_full_all.dtype, device=dev_all)
                        pos_pack = torch.zeros((Bp, P_max + R_max), dtype=pos_full_all.dtype, device=dev_all)
                        resp_pack = torch.full((Bp, R_max), pad_id, dtype=ids_full_all.dtype, device=dev_all)
                        mask_cot = torch.zeros((Bp, R_max), dtype=torch.float32, device=dev_all)
                        mask_ans = torch.zeros((Bp, R_max), dtype=torch.float32, device=dev_all)
                        A_pack = torch.zeros((Bp, R_max), dtype=torch.float32, device=dev_all)

                        # Fill rows
                        for j, r in enumerate(pack_rows):
                            c = int(c_s_all[r]); a = int(a_s_all[r])
                            Lp = int(pre_ls[j]); Lc = int(cot_ls[j]); La = int(ans_ls[j])
                            # prefix ids/attn/pos up to cot_start
                            if Lp > 0:
                                ids_pack[j, :Lp] = ids_full_all[r][:c]
                                attn_pack[j, :Lp] = attn_full_all[r][:c]
                                pos_pack[j, :Lp] = pos_full_all[r][:c]
                            # responses: CoT tokens then GT answer tokens
                            # CoT tokens from existing ids
                            if Lc > 0:
                                ids_pack[j, Lp : Lp + Lc] = ids_full_all[r][c:a]
                                attn_pack[j, Lp : Lp + Lc] = 1
                                pos_pack[j, Lp : Lp + Lc] = pos_full_all[r][c:a]
                                resp_pack[j, :Lc] = ids_full_all[r][c:a]
                                mask_cot[j, :Lc] = 1
                                # advantages for CoT span
                                A_pack[j, :Lc] = A_all[r]
                            # Answer tokens: prefer provided GT tokens; fallback to ids_full
                            if La > 0:
                                # choose token ids
                                gt_row = gt_tokens_all[r]
                                if gt_row is None:
                                    ans_ids = ids_full_all[r][a : a + La]
                                else:
                                    ans_ids = torch.as_tensor(gt_row, device=dev_all, dtype=ids_full_all.dtype)
                                ids_pack[j, Lp + Lc : Lp + Lc + La] = ans_ids
                                attn_pack[j, Lp + Lc : Lp + Lc + La] = 1
                                # try position_ids from full tensor; if out of bound, extend sequentially
                                try:
                                    pos_pack[j, Lp + Lc : Lp + Lc + La] = pos_full_all[r][a : a + La]
                                except Exception:
                                    if Lp + Lc > 0:
                                        start_pos = int(pos_pack[j, Lp + Lc - 1].item()) + 1
                                    else:
                                        start_pos = 0
                                    pos_pack[j, Lp + Lc : Lp + Lc + La] = torch.arange(start_pos, start_pos + La, device=dev_all, dtype=pos_pack.dtype)
                                resp_pack[j, Lc : Lc + La] = ans_ids
                                mask_ans[j, Lc : Lc + La] = 1
                                # advantages for Answer span
                                A_pack[j, Lc : Lc + La] = float(beta_supp) * w_all[r]

                        # Forward once to get combined CoT+Answer log-probs tail of length R_max
                        micro = {
                            "input_ids": ids_pack,
                            "attention_mask": attn_pack,
                            "position_ids": pos_pack,
                            "responses": resp_pack,
                        }
                        # Compute log-probs; optionally compute entropy when enabled
                        calculate_entropy = entropy_coeff != 0
                        entropy_tok, lp_combined = self._forward_micro_batch(
                            micro, temperature=temperature, calculate_entropy=calculate_entropy
                        )

                        # Compute losses (GPG) using masks for CoT and Answer; backward on combined
                        gpg_fn = get_policy_loss_fn("gpg")
                        # Combined mask and advantages
                        comb_mask = (mask_cot + mask_ans).clamp_max(1)
                        comb_adv = A_pack
                        jepo_loss_part, _, _, _ = gpg_fn(
                            old_log_prob=None,
                            log_prob=lp_combined,
                            advantages=comb_adv,
                            response_mask=comb_mask,
                            loss_agg_mode=loss_agg_mode,
                            #loss_agg_mode="seq-mean-token-mean",
                        )
                        # Optional entropy term, aggregated like in dp_actor
                        if calculate_entropy:
                            entropy_loss = agg_loss(
                                loss_mat=entropy_tok, loss_mask=comb_mask, loss_agg_mode=loss_agg_mode
                            )
                            jepo_loss_part = jepo_loss_part - entropy_coeff * entropy_loss
                        # For logging, extract means
                        cot_log_probs = (lp_combined * (mask_cot > 0)).sum(dim=-1).detach()
                        answer_log_probs = (lp_combined * (mask_ans > 0)).sum(dim=-1).detach()

                        kl_loss_part = torch.tensor(0.0, device=dev_all)
                        # Optional KL on original responses (separate token-capped pass)
                        if beta_kl > 0 and ("ref_log_prob" in data.batch.keys()):
                            pack_data = data[pack_rows]
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
                                ref_lp_orig = data.batch["ref_log_prob"][pack_rows]
                                resp_mask_orig = compute_response_mask(pack_data)
                                kld = kl_penalty(logprob=lp_orig, ref_logprob=ref_lp_orig, kl_penalty=kl_loss_type)
                                kl_loss_part = agg_loss(loss_mat=kld, loss_mask=resp_mask_orig, loss_agg_mode="token-mean") * beta_kl
                            except Exception:
                                kl_loss_part = torch.tensor(0.0, device=dev_all)
                        # Scale loss per micro like dp_actor
                        if use_dynamic_bsz:
                            loss_scale_factor = float(Bp) / float(max(mini_bs, 1))
                        else:
                            loss_scale_factor = 1.0 / float(max(grad_accum, 1))
                        loss_chunk = (jepo_loss_part + kl_loss_part) * loss_scale_factor
                        loss_chunk.backward()
                        print(f"finish one chunk loss gradient")

                        # metrics accumulation
                        meters["total_loss"] += float(loss_chunk.detach())
                        meters["jepo_loss"] += float(jepo_loss_part.detach()) * loss_scale_factor
                        meters["supp_loss"] += 0.0  # already merged into combined via A_pack
                        with torch.no_grad():
                            # A_vec mean/std for this pack
                            A_vec_pack = A_all[pack_rows]
                            meters["jepo_advs_mean"] += float(A_vec_pack.mean().detach())
                            meters["jepo_advs_std"] += float(A_vec_pack.std().detach())
                        if cot_log_probs.numel() > 0:
                            meters["cot_log_probs_mean"] += float(cot_log_probs.mean().detach())
                        if answer_log_probs.numel() > 0:
                            meters["log_mean_answer_probs_mean"] += float(answer_log_probs.mean().detach())
                        meters["kl_loss"] += float(kl_loss_part.detach()) * loss_scale_factor
                        meter_count += 1

                # End of mini-batch: step optimizer
                grad_norm = self._optimizer_step()
                meters["grad_norm"] += float(grad_norm.detach())
                self.actor_optimizer.zero_grad()

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
            # Build per-row answer lengths
            ans_lens = []
            for i in range(B):
                gt_row = dd["ground_truth_answer_tokens"][i]
                ans_lens.append(len(gt_row) if isinstance(gt_row, list) else int(torch.as_tensor(gt_row, device=dev).numel()))
            # Indices with non-zero answer length
            ans_rows = [i for i, L in enumerate(ans_lens) if L > 0]
            if ans_rows:
                chunk_sz = max(1, min(len(ans_rows), 64))
                for s in range(0, len(ans_rows), chunk_sz):
                    e = min(s + chunk_sz, len(ans_rows))
                    rows = ans_rows[s:e]
                    ids_batch = dd["batch_input_ids"][rows]
                    attn_batch = dd["attention_mask"][rows]
                    pos_batch = dd["position_ids"][rows]
                    # Build ragged answer tokens batch
                    R_max = 0
                    resp_list = []
                    for r in rows:
                        a_s = int(dd["answer_start_positions"][r])
                        Lr = int(ans_lens[r])
                        t = dd["batch_input_ids"][r][a_s:a_s+Lr]
                        resp_list.append(t)
                        R_max = max(R_max, int(t.size(0)))
                    if R_max == 0:
                        continue
                    resp_tok = torch.full((len(rows), R_max), pad_id, dtype=ids_batch.dtype, device=dev)
                    for j, t in enumerate(resp_list):
                        l = int(t.size(0)); resp_tok[j, :l] = t
                    micro_ans = {
                        "input_ids": ids_batch,
                        "attention_mask": attn_batch,
                        "position_ids": pos_batch,
                        "responses": resp_tok,
                    }
                    _, lp_ans_sub = self._forward_micro_batch(micro_ans, temperature=temperature, calculate_entropy=False)
                    for j, r in enumerate(rows):
                        Lr = int(ans_lens[r])
                        ans_logprob[r] = lp_ans_sub[j, :Lr].sum().detach()
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
