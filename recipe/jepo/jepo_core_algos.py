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

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from collections import defaultdict
import verl.utils.torch_functional as verl_F

def dummy_backward_fsdp_safe(model, scaler=None):
    device = next(model.parameters()).device
    input_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
    model.train()
    out = model(input_ids=input_ids)
    loss = out.logits[..., :1].sum() * 0.0
    (scaler.scale(loss) if scaler else loss).backward()

import torch.distributed as dist

def _to_tensor_list(x, device):
    return torch.as_tensor(x, device=device) if not isinstance(x, torch.Tensor) else x.to(device)

def _nonzero_idx(bool_list, device):
    t = torch.as_tensor(bool_list, device=device, dtype=torch.bool)
    return torch.nonzero(t, as_tuple=False).flatten()

def _allreduce_sum_scalar(val: float | int, device) -> float:
    t = torch.tensor([float(val)], device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())

def _maybe_dummy_backward(model):
    # If you already have dummy_backward_fsdp_safe, call that instead.
    try:
        dummy_backward_fsdp_safe(model, scaler=None)
    except NameError:
        # ultra-cheap zero grad anchor
        with torch.no_grad():
            z = next(model.parameters()).sum() * 0.0
        (z * 0.0).backward()


@dataclass
class JEPOConfig:
    delimiter: str = "\n\n"
    format_penalty: float = 0.1
    beta_supp: float = 1.0
    beta_kl: float = 0.1
    buffer_size: int = 1000
    jepo_steps: int = 5,
    epochs: int = 1,
    mini_batch_size_per_gpu: int = 8,  # questions per optimizer step per rank
    micro_batch_size_per_gpu: int = 1,  # questions per backward pass per rank
    responses_micro_batch_size: int = 8  # responses per question when calculating loss
    num_response_per_question: int = 8,
    accum_steps: int = 4,  # fixed accumulate steps for consistent backwards
    

def _find_subsequence(haystack_ids: torch.Tensor, needle_ids: List[int]) -> int:
    if len(needle_ids) == 0:
        return -1
    T = haystack_ids.numel()
    L = len(needle_ids)
    if L > T:
        return -1
    needle = torch.tensor(needle_ids, device=haystack_ids.device, dtype=haystack_ids.dtype)
    for s in range(0, T - L + 1):
        if torch.equal(haystack_ids[s : s + L], needle):
            return s
    return -1

def _rfind_subsequence(haystack_ids: torch.Tensor, needle_ids: List[int]) -> int:
    """Return start index of the last occurrence of needle_ids in haystack_ids, or -1."""
    if len(needle_ids) == 0:
        return -1
    T = haystack_ids.numel()
    L = len(needle_ids)
    if L > T:
        return -1
    needle = torch.tensor(needle_ids, device=haystack_ids.device, dtype=haystack_ids.dtype)
    for s in range(T - L, -1, -1):
        if torch.equal(haystack_ids[s : s + L], needle):
            return s
    return -1


def _find_delimiter_position(
    resp_ids: torch.Tensor,
    delimiter_ids: List[int],
    enable_suffix_anchor: bool = True,
    min_suffix_len: int = 2,
) -> tuple[int, str]:
    """Return (start_index, kind) for delimiter match.
    kind in {"none","full","tail"}. If enable_suffix_anchor, also tries the tail
    (last min_suffix_len tokens) and picks the LAST occurrence among candidates.
    """
    s_full = _rfind_subsequence(resp_ids, delimiter_ids)
    best_idx = s_full
    best_kind = "full" if s_full >= 0 else "none"
    if enable_suffix_anchor and len(delimiter_ids) >= min_suffix_len:
        tail = delimiter_ids[-min_suffix_len:]
        s_tail = _rfind_subsequence(resp_ids, tail)
        if s_tail >= 0 and (best_idx < 0 or s_tail > best_idx):
            best_idx = s_tail
            best_kind = "tail"
    return best_idx, best_kind


def build_jepo_teacher_forced_batch(
    response_tokens: torch.Tensor,
    prompt_tokens: torch.Tensor,
    ground_truth_answer_tokens: np.ndarray,
    delimiter_str: str,
    format_penalty: float,
    model,
    device: torch.device,
    pad_token: int,
    tokenizer,
    *,
    delimiter_suffix_anchor: bool = True,
    delimiter_suffix_min_len: int = 2,
):
    # Token-ID based delimiter splitting (avoid decode/encode drift)
    n = response_tokens.size(0)
    max_response_length = response_tokens.size(1)

    delimiter_ids = tokenizer.encode(delimiter_str, add_special_tokens=False)
    delimiter_len = len(delimiter_ids)
    gt_list: List[List[int]] = [list(x) if isinstance(x, (list, tuple)) else list(x.tolist()) for x in ground_truth_answer_tokens]

    has_delimiter: List[bool] = []
    cot_tokens_list: List[List[int]] = []
    delimiter_match_kind: List[str] = []

    for i in range(n):
        resp_i = response_tokens[i].detach().clone()
        s, match_kind = _find_delimiter_position(
            resp_i, delimiter_ids,
            enable_suffix_anchor=delimiter_suffix_anchor,
            min_suffix_len=max(1, delimiter_suffix_min_len),
        )
        found = s >= 0
        has_delimiter.append(bool(found))
        delimiter_match_kind.append(match_kind)
        cot_ids = resp_i[:s].tolist() if found else resp_i.tolist()
        gt_len = len(gt_list[i])
        max_cot = max(0, max_response_length - delimiter_len - gt_len)
        if len(cot_ids) > max_cot:
            cot_ids = cot_ids[:max_cot]
        cot_tokens_list.append(cot_ids)

    batch_input_tokens: List[torch.Tensor] = []
    cot_start_positions: List[int] = []
    answer_start_positions: List[int] = []

    for i in range(n):
        prompt_i = prompt_tokens[i] if isinstance(prompt_tokens, torch.Tensor) else torch.tensor(prompt_tokens[i], device=device)
        prompt_i = prompt_i.to(device=device, dtype=torch.long)
        cot_i = torch.tensor(cot_tokens_list[i], device=device, dtype=torch.long)
        delim_i = torch.tensor(delimiter_ids, device=device, dtype=torch.long)
        gt_i = torch.tensor(gt_list[i], device=device, dtype=torch.long)

        left = torch.cat([prompt_i, cot_i], dim=0)
        left_plus_delim = torch.cat([left, delim_i], dim=0)
        full = torch.cat([left_plus_delim, gt_i], dim=0)

        batch_input_tokens.append(full)
        cot_start_positions.append(len(prompt_i))
        answer_start_positions.append(len(left_plus_delim))

    max_len = max(int(t.numel()) for t in batch_input_tokens) if batch_input_tokens else 0
    padded_tokens: List[torch.Tensor] = []
    attention_masks: List[List[int]] = []

    for t in batch_input_tokens:
        pad_len = max_len - int(t.numel())
        if pad_len > 0:
            padding = torch.full((pad_len,), pad_token, dtype=torch.long, device=device)
            padded = torch.cat([t, padding], dim=0)
            mask = [1] * int(t.numel()) + [0] * pad_len
        else:
            padded = t
            mask = [1] * int(t.numel())
        padded_tokens.append(padded)
        attention_masks.append(mask)

    batch_input_ids = torch.stack(padded_tokens).to(dtype=torch.long, device=device) if padded_tokens else torch.empty((0, 0), dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device) if attention_masks else torch.empty((0, 0), dtype=torch.long, device=device)
    position_ids = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0).repeat(n, 1) if max_len > 0 else torch.empty((n, 0), dtype=torch.long, device=device)
    for i in range(n):
        if max_len > batch_input_tokens[i].numel():
            position_ids[i, batch_input_tokens[i].numel():] = 0

    return {
        'batch_input_ids': batch_input_ids,
        'attention_mask': attention_mask, 
        'position_ids': position_ids,
        'cot_start_positions': cot_start_positions,
        'answer_start_positions': answer_start_positions,
        'cot_tokens_list': cot_tokens_list,
        'ground_truth_answer_tokens': gt_list,
        'has_delimiter': has_delimiter,
        'delimiter_match_kind': delimiter_match_kind,
        'max_len': max_len
    }


def build_jepo_batches_by_prompt(
    response_tokens,
    prompt_tokens,
    ground_truth_answer_tokens,
    delimiter_str,
    format_penalty,
    model,
    device,
    pad_token,
    index,
    tokenizer,
    *,
    delimiter_suffix_anchor: bool = True,
    delimiter_suffix_min_len: int = 2,
):

    # response_tokens has shape [bsz, max_response_length]
    # prompt_tokens has shape [bsz, max_prompt_length]
    # ground_truth_answer_tokens is a list with shape bsz, np.ndarray
    # delimiter_str is a str
    # index is a list of uid

    uuid = np.unique(index)

    for uid in uuid:
        uid_mask = (index == uid)
        response_tokens_uid = response_tokens[uid_mask]
        prompt_tokens_uid = prompt_tokens[uid_mask]
        ground_truth_answer_tokens_uid = ground_truth_answer_tokens[uid_mask]
        
        if len(response_tokens_uid) == 0:
            continue
        data_dict = build_jepo_teacher_forced_batch(
            response_tokens=response_tokens_uid,
            prompt_tokens=prompt_tokens_uid,
            ground_truth_answer_tokens=ground_truth_answer_tokens_uid,
            delimiter_str=delimiter_str,
            format_penalty=format_penalty,
            model=model,
            device=device,
            pad_token=pad_token,
            tokenizer=tokenizer,
            delimiter_suffix_anchor=delimiter_suffix_anchor,
            delimiter_suffix_min_len=delimiter_suffix_min_len,
        )
        
        if 'data_dicts' not in locals():
            data_dicts = [data_dict]
        else:
            data_dicts.append(data_dict)
    
    return data_dicts

"""
Backwards-compatibility aliases to minimize churn in existing code.
Prefer the new names in future code paths.
"""
compute_single_jepo_advantages = build_jepo_teacher_forced_batch
compute_jepo_advantages = build_jepo_batches_by_prompt
compute_jepo_from_logits_efficient = compute_segment_logprobs_shifted
compute_jepo_from_logits_sparse = compute_segment_logprobs_sparse



import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def _normalize(vec: torch.Tensor) -> torch.Tensor:
    return (vec - vec.mean()) / (vec.std(unbiased=False) + 1e-8)

def _rowwise_logsumexp_streaming(logits: torch.Tensor,
                                 b_rows: torch.Tensor,
                                 t_rows: torch.Tensor,
                                 vocab_chunk: int = 8192) -> torch.Tensor:
    """
    Compute logsumexp over vocab for selected (b,t) rows in a memory-friendly way.
    Returns: lse [U] for U = len(b_rows).
    """
    U = b_rows.numel()
    device = logits.device
    lse = torch.full((U,), float("-inf"), device=device, dtype=torch.float32)
    V = logits.size(-1)

    # stream over vocab to avoid materializing [U, V]
    for start in range(0, V, vocab_chunk):
        end = min(start + vocab_chunk, V)
        # [U, w]
        chunk = logits[b_rows, t_rows, start:end].float()
        m = chunk.max(dim=-1).values                          # [U]
        # logsumexp per chunk
        lse_chunk = m + torch.log(torch.clamp(
            torch.sum(torch.exp(chunk - m.unsqueeze(-1)), dim=-1), min=1e-45
        ))
        # combine across chunks in log-space
        lse = torch.logaddexp(lse, lse_chunk)
    return lse  # float32 for stability

def _sparse_logprob_sum(logits: torch.Tensor,
                        b_idx: torch.Tensor,
                        t_idx: torch.Tensor,
                        v_idx: torch.Tensor,
                        B: int,
                        vocab_chunk: int = 8192) -> torch.Tensor:
    """
    Compute sum of log-probs per batch item for a sparse set of (b,t,v) indices.
    Returns: per-sample sums [B] with grad.
    """
    if b_idx.numel() == 0:
        return torch.zeros(B, device=logits.device, dtype=logits.dtype)

    # Group by unique (b,t) rows
    rows = torch.stack([b_idx, t_idx], dim=1)                       # [K, 2]
    uniq_rows, inv = torch.unique(rows, dim=0, return_inverse=True) # uniq_rows: [U,2], inv: [K]
    b_rows = uniq_rows[:, 0].long()
    t_rows = uniq_rows[:, 1].long()

    # Row-wise logsumexp over vocab, streamed
    lse_rows = _rowwise_logsumexp_streaming(logits, b_rows, t_rows, vocab_chunk=vocab_chunk)  # [U], float32

    # Label logits for each sparse index
    label_logits = logits[b_idx, t_idx, v_idx].float()              # [K], float32

    # Sparse log-prob per index: log p = logit(label) - logsumexp(row)
    logprob_flat = label_logits - lse_rows[inv]                     # [K], float32

    # Sum per batch element
    out = torch.zeros(B, device=logits.device, dtype=torch.float32)
    out.index_add_(0, b_idx, logprob_flat)
    return out.to(logits.dtype)                                     # match model dtype (bf16/fp16/fp32)

def compute_segment_logprobs_sparse(
    logits, data_dict, format_penalty, has_delimiter, vocab_chunk=8192
):
    """
    Memory-friendly replacement for compute_jepo_from_logprobs_*.
    Takes raw logits [B, T, V] (already temperature-scaled), never builds BxTxV log_softmax.
    """
    device, dtype = logits.device, logits.dtype
    B, T, V = logits.shape

    cot_start = data_dict['cot_start_positions']
    ans_start = data_dict['answer_start_positions']
    cot_tok_list = data_dict['cot_tokens_list']
    ans_tok_list = data_dict['ground_truth_answer_tokens']

    # --- build sparse indices for CoT tokens ---
    b_c, t_c, v_c = [], [], []
    for i, (s, toks) in enumerate(zip(cot_start, cot_tok_list)):
        if len(toks) == 0: continue
        t = torch.arange(s, s + len(toks), device=device) - 1
        m = (t >= 0) & (t < T)
        if m.any():
            b_c.append(torch.full((int(m.sum().item()),), i, device=device, dtype=torch.long))
            t_c.append(t[m].long())
            v_c.append(torch.as_tensor(toks, device=device, dtype=torch.long)[m])
    if b_c:
        b_c = torch.cat(b_c); t_c = torch.cat(t_c); v_c = torch.cat(v_c)
        cot_log_probs = _sparse_logprob_sum(logits, b_c, t_c, v_c, B, vocab_chunk=vocab_chunk)  # [B]
    else:
        cot_log_probs = torch.zeros(B, device=device, dtype=dtype)

    # --- build sparse indices for ANSWER tokens ---
    b_a, t_a, v_a = [], [], []
    for i, (s, toks) in enumerate(zip(ans_start, ans_tok_list)):
        if len(toks) == 0: continue
        t = torch.arange(s, s + len(toks), device=device) - 1
        m = (t >= 0) & (t < T)
        if m.any():
            b_a.append(torch.full((int(m.sum().item()),), i, device=device, dtype=torch.long))
            t_a.append(t[m].long())
            v_a.append(torch.as_tensor(toks, device=device, dtype=torch.long)[m])
    if b_a:
        b_a = torch.cat(b_a); t_a = torch.cat(t_a); v_a = torch.cat(v_a)
        answer_log_probs = _sparse_logprob_sum(logits, b_a, t_a, v_a, B, vocab_chunk=vocab_chunk)  # [B]
    else:
        answer_log_probs = torch.zeros(B, device=device, dtype=dtype)

    # ---- log-mean over answers (WITH grad) ----
    # scalar: log_mean = logsumexp(answer_log_probs) - log(B)
    lse_all = torch.logsumexp(answer_log_probs.float(), dim=0)      # float32 for stability
    log_mean_prob = (lse_all - math.log(max(B, 1))).to(dtype)

    # ---- JEPO advantage (outside graph where appropriate) ----
    with torch.no_grad():
        ans_det = answer_log_probs.detach().float()
        lse_all_d = torch.logsumexp(ans_det, dim=0)
        d = ans_det - lse_all_d
        if B > 1:
            lse_others = lse_all_d + torch.log((-torch.expm1(d)).clamp_min(1e-12))
            v_i = lse_others - math.log(B - 1)
        else:
            v_i = ans_det.new_full((B,), float("-inf"))
        A = (log_mean_prob.detach().float() - v_i)
        A = torch.clamp(_normalize(A), -1.0, 1.0)

        has_delim = torch.as_tensor(has_delimiter, device=device, dtype=torch.bool)
        fmt = torch.where(
            has_delim,
            torch.zeros(B, device=device, dtype=torch.float32),
            torch.tensor(-float(format_penalty), device=device, dtype=torch.float32),
        )
        # Mean-only normalization
        fmt = fmt - fmt.mean()

        jepo_advantage = A  # + fmt if you want to include format penalty
        jepo_advantage = jepo_advantage.to(dtype)

    return jepo_advantage, cot_log_probs, answer_log_probs, log_mean_prob


def compute_segment_logprobs_shifted(
    logits, data_dict, format_penalty, has_delimiter, vocab_chunk=8192
):
    """
    Efficient replacement using standard shift and gather operations.
    Much faster than sparse indexing approach.
    """
    device, dtype = logits.device, logits.dtype
    B, T, V = logits.shape

    cot_start = data_dict['cot_start_positions']
    ans_start = data_dict['answer_start_positions']
    cot_tok_list = data_dict['cot_tokens_list']
    ans_tok_list = data_dict['ground_truth_answer_tokens']

    # Standard language modeling setup: shift logits and create targets
    shift_logits = logits[..., :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = torch.full((B, T-1), -100, dtype=torch.long, device=device)  # [B, T-1]
    
    # Create masks for CoT and answer regions
    cot_mask = torch.zeros((B, T-1), dtype=torch.bool, device=device)
    ans_mask = torch.zeros((B, T-1), dtype=torch.bool, device=device)
    
    # Fill in the target tokens and masks
    for i in range(B):
        # CoT tokens
        cot_tokens = cot_tok_list[i]
        cot_s = cot_start[i]
        if len(cot_tokens) > 0:
            cot_end = min(cot_s + len(cot_tokens), T-1)
            cot_len = cot_end - cot_s
            if cot_len > 0:
                shift_labels[i, cot_s:cot_end] = torch.tensor(cot_tokens[:cot_len], device=device, dtype=torch.long)
                cot_mask[i, cot_s:cot_end] = True
        
        # Answer tokens  
        ans_tokens = ans_tok_list[i]
        ans_s = ans_start[i]
        if len(ans_tokens) > 0:
            ans_end = min(ans_s + len(ans_tokens), T-1)
            ans_len = ans_end - ans_s
            if ans_len > 0:
                shift_labels[i, ans_s:ans_end] = torch.tensor(ans_tokens[:ans_len], device=device, dtype=torch.long)
                ans_mask[i, ans_s:ans_end] = True

        # Include delimiter tokens (between CoT end and answer start) into CoT mask
        # so g1 and formatting updates can act on them when applicable.
        # We do not need the delimiter IDs explicitly: use the teacher-forced tokens
        # already present in batch_input_ids to set labels for these positions.
        # CoT end index (clamped within [0, T-1])
        cot_end_idx = min(cot_s + (len(cot_tokens) if len(cot_tokens) > 0 else 0), T-1)
        # Delimiter occupies [del_start, del_end) right before answer_start
        del_start = max(0, cot_end_idx)
        del_end = min(ans_s, T-1)
        if del_end > del_start:
            # Labels for these positions are just the next tokens in the input stream
            # represented by batch_input_ids (after shifting by one handled globally).
            # Since our shift_labels convention elsewhere uses direct tokens for the
            # segment, we follow the same here.
            shift_labels[i, del_start:del_end] = data_dict["batch_input_ids"][i, del_start:del_end]
            cot_mask[i, del_start:del_end] = True
    
    # Compute log probabilities efficiently without OOM
    # Chunk the computation to handle long sequences with large vocab
    token_log_probs = torch.zeros((B, T-1), device=device, dtype=dtype)
    valid_mask = (shift_labels != -100)
    
    if valid_mask.any():
        # Get indices where we need to compute log probs
        valid_b, valid_t = torch.where(valid_mask)
        valid_labels = shift_labels[valid_mask]  # [num_valid]
        num_valid = len(valid_b)
        
        # Chunk size to avoid OOM - adjust based on available memory
        # Conservative estimate: ~1K positions at a time for large vocab
        chunk_size = min(vocab_chunk // 100, 1024)  # vocab_chunk is typically 8192, so this gives ~80 positions
        
        for start_idx in range(0, num_valid, chunk_size):
            end_idx = min(start_idx + chunk_size, num_valid)
            
            # Get chunk indices
            chunk_b = valid_b[start_idx:end_idx]
            chunk_t = valid_t[start_idx:end_idx]
            chunk_labels = valid_labels[start_idx:end_idx]
            
            # Compute log softmax only for this chunk
            chunk_logits = shift_logits[chunk_b, chunk_t]  # [chunk_size, V]
            chunk_log_probs = F.log_softmax(chunk_logits, dim=-1)  # [chunk_size, V]
            
            # Gather the target token probabilities
            chunk_target_probs = chunk_log_probs.gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1)  # [chunk_size]
            
            # Put back into the full tensor
            chunk_mask = torch.zeros_like(valid_mask)
            chunk_mask[valid_b[start_idx:end_idx], valid_t[start_idx:end_idx]] = True
            token_log_probs[chunk_mask] = chunk_target_probs
            
            # Clean up intermediate tensors
            del chunk_logits, chunk_log_probs, chunk_target_probs, chunk_mask
    
    # Sum over CoT and answer regions
    cot_log_probs = (token_log_probs * cot_mask.float()).sum(dim=-1)  # [B]
    answer_log_probs = (token_log_probs * ans_mask.float()).sum(dim=-1)  # [B]
    
    # ---- log-mean over answers (WITH grad) ----
    lse_all = torch.logsumexp(answer_log_probs.float(), dim=0)
    log_mean_prob = (lse_all - math.log(max(B, 1))).to(dtype)

    return cot_log_probs, answer_log_probs, log_mean_prob


def token_level_jepo_loss(
    logits,
    data_dict,
    A_vec: torch.Tensor,
    w_vec: torch.Tensor,
    beta_supp: float,
    normalize_by: str = "tokens",
    vocab_chunk: int = 8192,
):
    """
    Token-level JEPO loss in a GRPO-style formulation.

    - For CoT + delimiter tokens, per-token advantage = A_vec[i]
    - For answer tokens, per-token advantage = beta_supp * w_vec[i]

    Args:
        logits: [B, T, V]
        data_dict: contains positions and token lists built by build_dataset-like util
        A_vec: [B] normalized JEPO advantages (including format if desired)
        w_vec: [B] softmax weights for exact grad of log-mean over answers
        beta_supp: scalar multiplier for support (answer) term
        normalize_by: "tokens" | "responses" — denominator for averaging
        vocab_chunk: chunking knob for memory safety

    Returns dict with:
        loss: scalar
        jepo_loss_part: scalar (CoT part only)
        supp_loss_part: scalar (answer part only)
        cot_log_probs: [B] summed CoT token log-probs (for metrics)
        ans_log_probs: [B] summed answer token log-probs (for metrics)
    """
    device, dtype = logits.device, logits.dtype
    B, T, V = logits.shape

    cot_start = data_dict['cot_start_positions']
    ans_start = data_dict['answer_start_positions']
    cot_tok_list = data_dict['cot_tokens_list']
    ans_tok_list = data_dict['ground_truth_answer_tokens']

    # Standard language modeling setup: shift logits and create targets
    shift_logits = logits[..., :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = torch.full((B, T-1), -100, dtype=torch.long, device=device)  # [B, T-1]

    # Create masks for CoT and answer regions
    cot_mask = torch.zeros((B, T-1), dtype=torch.bool, device=device)
    ans_mask = torch.zeros((B, T-1), dtype=torch.bool, device=device)

    # Fill in the target tokens and masks
    for i in range(B):
        # CoT tokens
        cot_tokens = cot_tok_list[i]
        cot_s = cot_start[i]
        if len(cot_tokens) > 0:
            cot_end = min(cot_s + len(cot_tokens), T-1)
            cot_len = cot_end - cot_s
            if cot_len > 0:
                shift_labels[i, cot_s:cot_end] = torch.tensor(cot_tokens[:cot_len], device=device, dtype=torch.long)
                cot_mask[i, cot_s:cot_end] = True

        # Answer tokens
        ans_tokens = ans_tok_list[i]
        ans_s = ans_start[i]
        if len(ans_tokens) > 0:
            ans_end = min(ans_s + len(ans_tokens), T-1)
            ans_len = ans_end - ans_s
            if ans_len > 0:
                shift_labels[i, ans_s:ans_end] = torch.tensor(ans_tokens[:ans_len], device=device, dtype=torch.long)
                ans_mask[i, ans_s:ans_end] = True

        # Include delimiter tokens (between CoT end and answer start) into CoT mask
        cot_end_idx = min(cot_s + (len(cot_tokens) if len(cot_tokens) > 0 else 0), T-1)
        del_start = max(0, cot_end_idx)
        del_end = min(ans_s, T-1)
        if del_end > del_start:
            shift_labels[i, del_start:del_end] = data_dict["batch_input_ids"][i, del_start:del_end]
            cot_mask[i, del_start:del_end] = True

    # Compute per-position token log-probs
    token_log_probs = torch.zeros((B, T-1), device=device, dtype=dtype)
    valid_mask = (shift_labels != -100)
    if valid_mask.any():
        valid_b, valid_t = torch.where(valid_mask)
        valid_labels = shift_labels[valid_mask]
        num_valid = len(valid_b)

        chunk_size = min(vocab_chunk // 100, 1024)
        for start_idx in range(0, num_valid, chunk_size):
            end_idx = min(start_idx + chunk_size, num_valid)
            cb = valid_b[start_idx:end_idx]
            ct = valid_t[start_idx:end_idx]
            cl = valid_labels[start_idx:end_idx]
            chunk_logits = shift_logits[cb, ct]
            chunk_log_probs = torch.log_softmax(chunk_logits, dim=-1)
            chunk_target = chunk_log_probs.gather(-1, cl.unsqueeze(-1)).squeeze(-1)
            token_log_probs[cb, ct] = chunk_target
            del chunk_logits, chunk_log_probs, chunk_target

    # Build per-token advantages
    adv_mat = torch.zeros((B, T-1), device=device, dtype=token_log_probs.dtype)
    if B > 0:
        adv_mat[cot_mask] = A_vec.to(device=device, dtype=token_log_probs.dtype).repeat_interleave(
            cot_mask.sum(dim=1).clamp_min(0)
        )[: int(cot_mask.sum().item())]
        # For answer tokens, use beta_supp * w_i
        w_scaled = (w_vec.to(device=device, dtype=token_log_probs.dtype) * float(beta_supp))
        adv_mat[ans_mask] = w_scaled.repeat_interleave(ans_mask.sum(dim=1).clamp_min(0))[: int(ans_mask.sum().item())]

    resp_mask = (cot_mask | ans_mask)

    # Separate parts for metrics
    cot_lp_vec = (token_log_probs * cot_mask.float()).sum(dim=-1)  # [B]
    ans_lp_vec = (token_log_probs * ans_mask.float()).sum(dim=-1)  # [B]

    # Losses
    jepo_term = -(adv_mat * token_log_probs * cot_mask.float()).sum()
    supp_term = -(adv_mat * token_log_probs * ans_mask.float()).sum()

    if normalize_by == "tokens":
        denom = resp_mask.float().sum().clamp_min(1.0)
    else:  # responses
        denom = resp_mask.any(dim=1).float().sum().clamp_min(1.0)

    loss = (jepo_term + supp_term) / denom

    return {
        "loss": loss,
        "jepo_loss_part": jepo_term / denom,
        "supp_loss_part": supp_term / denom,
        "cot_log_probs": cot_lp_vec.detach(),
        "answer_log_probs": ans_lp_vec.detach(),
        "denom": denom.detach(),
    }

import math
import numpy as np
import torch
from contextlib import nullcontext
from verl import DataProto

# --- helper: slice a single question's data_dict by response indices ---
def _slice_data_dict(dd, idxs):
    def take_list(lst): return [lst[i] for i in idxs]
    return {
        "batch_input_ids": dd["batch_input_ids"][idxs],
        "attention_mask": dd["attention_mask"][idxs],
        "position_ids": dd["position_ids"][idxs],
        "cot_start_positions": take_list(dd["cot_start_positions"]),
        "answer_start_positions": take_list(dd["answer_start_positions"]),
        "cot_tokens_list": take_list(dd["cot_tokens_list"]),
        "ground_truth_answer_tokens": take_list(dd["ground_truth_answer_tokens"]),
        "has_delimiter": take_list(dd["has_delimiter"]),
    }

def _chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]



@torch.no_grad()
def precompute_adv_for_dd(
    model,
    dd: dict,
    temperature: float,
    format_penalty: float,
    responses_micro_bs: int,
    vocab_chunk: int,
    device_name: str,
):
    """
    Fills dd with:
      - 'A_full' [B] (detached, computed over ALL responses)
      - 'w_full' [B] (detached, softmax weights over ALL responses)
      - 'keep_idx' [K] (indices with has_delimiter=True)
      - 'log_mean_det' (scalar, detached)
    """
    dev = dd["batch_input_ids"].device
    B = len(dd["cot_start_positions"])
    all_idx = torch.arange(B, device=dev)
    ans_det_chunks = []

    for s in range(0, B, responses_micro_bs):
        idxs = all_idx[s:s+responses_micro_bs]
        dd_s = _slice_data_dict(dd, idxs.tolist())
        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            out = model(
                input_ids=dd_s["batch_input_ids"].detach(),
                attention_mask=dd_s["attention_mask"],
                position_ids=dd_s["position_ids"],
                use_cache=False,
            )
            logits = out.logits
            logits.div_(temperature)
        # only need detached answer log-probs here
        _, _, ans_lp, _ = compute_segment_logprobs_sparse(
            logits, dd_s, format_penalty, dd_s["has_delimiter"], vocab_chunk=vocab_chunk
        )
        ans_det_chunks.append(ans_lp.detach().float().cpu())
        del out, logits

    ans_det = torch.cat(ans_det_chunks, dim=0).to(dev)       # [B]
    lse_all = torch.logsumexp(ans_det, dim=0)                # scalar
    d = ans_det - lse_all
    if B > 1:
        lse_others = lse_all + torch.log((-torch.expm1(d)).clamp_min(1e-12))
        v_i = lse_others - math.log(B - 1)
    else:
        v_i = ans_det.new_full((B,), float("-inf"))
    log_mean_det = lse_all - math.log(B)                     # scalar

    # JEPO RAW advantage A_i over ALL responses (detach)
    A_raw = (log_mean_det - v_i)
    A_raw = A_raw / (A_raw.std(unbiased=False) + 1e-8)
    A_raw = A_raw.clamp(-1.0, 1.0)

    # format penalty term (normalized) – keep same as your pass-2 logic
    has_delim = torch.as_tensor(dd["has_delimiter"], device=dev, dtype=torch.bool)
    fmt = torch.where(
        has_delim,
        torch.zeros(B, device=dev, dtype=torch.float32),
        torch.tensor(-float(format_penalty), device=dev, dtype=torch.float32),
    )
    # Mean-only normalization so the penalty scale remains effective
    fmt = fmt - fmt.mean()
    A_full = (A_raw + fmt).detach()

    # weights for exact gradient of logsumexp over ALL responses
    w_full = torch.softmax(ans_det, dim=0).detach()

    keep_idx = _nonzero_idx(dd["has_delimiter"], device=dev)

    # cache in dd (CPU or GPU is fine; keep on same dev to avoid copies later)
    dd["A_full"] = A_full
    dd["A_raw"] = A_raw.detach()
    dd["fmt_norm"] = fmt.detach()
    dd["w_full"] = w_full
    dd["keep_idx"] = keep_idx
    dd["log_mean_det"] = log_mean_det.detach().item()



@torch.no_grad()
def attach_jepo_adv_to_dataproto(data: DataProto, model, jepo_cfg: dict, cached_tokenizer):
    """
    Compute JEPO advantages and necessary info for data, maintaining DataProto class.
    This function does not require any gradients and only adds keys and values to data proto.
    
    Args:
        data: DataProto object containing batch data
        model: The actor model 
        jepo_cfg: JEPO configuration dictionary
        cached_tokenizer: Tokenizer for encoding/decoding
        
    Returns:
        DataProto object with added advantage information in data.batch
    """
    # Extract config
    format_penalty = float(jepo_cfg.get("format_penalty", 0.0))
    temperature = float(data.meta_info["temperature"])
    resp_micro_bs = int(jepo_cfg.get("responses_micro_batch_size", 8))
    delimiter = jepo_cfg.get("delimiter", "\n\n")
    # Configurable suffix-anchor matching for delimiters
    use_suffix_anchor = bool(jepo_cfg.get("delimiter_suffix_anchor", True))
    suffix_min_len = int(jepo_cfg.get("delimiter_suffix_min_len", 2))
    
    # Prepare model inputs
    model_inputs = {**data.batch, **data.non_tensor_batch}
    ground_truths = model_inputs.get("reward_model", {})
    ground_truths_tokens = np.array(
        [cached_tokenizer.encode(gt.get("ground_truth", []), add_special_tokens=False) for gt in ground_truths],
        dtype=object,
    )
    pad_token = cached_tokenizer.pad_token_id
    
    # Compute advantages for all questions
    data_dicts = build_jepo_batches_by_prompt(
        response_tokens=data.batch["responses"],
        prompt_tokens=data.batch["prompts"],
        ground_truth_answer_tokens=ground_truths_tokens,
        delimiter_str=delimiter,
        format_penalty=format_penalty,
        model=model,
        device=model.device,
        pad_token=pad_token,
        index=data.non_tensor_batch["uid"],
        tokenizer=cached_tokenizer,
        delimiter_suffix_anchor=use_suffix_anchor,
        delimiter_suffix_min_len=suffix_min_len,
    )
    
    # Precompute advantages for all data dicts
    for dd in data_dicts:
        precompute_adv_for_dd(
            model=model,
            dd=dd,
            temperature=temperature,
            format_penalty=format_penalty,
            responses_micro_bs=resp_micro_bs,
            vocab_chunk=8192,
            device_name="cuda" if torch.cuda.is_available() else "cpu",
        )
    
    # Extract A and w from data_dicts and reconstruct batch tensors
    batch_size = data.batch["responses"].shape[0]
    device = data.batch["responses"].device
    
    # Initialize tensors for advantages and weights
    jepo_adv = torch.zeros(batch_size, device=device)
    jepo_adv_raw = torch.zeros(batch_size, device=device)
    format_adv = torch.zeros(batch_size, device=device)
    jepo_weights = torch.zeros(batch_size, device=device) 
    has_delimiter = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Initialize lists for position information needed for log prob calculation
    all_cot_start_positions = []
    all_answer_start_positions = []
    all_cot_tokens_list = []
    all_ground_truth_answer_tokens = []
    
    # Map data_dict results back to original batch indices
    current_idx = 0
    for dd in data_dicts:
        n_responses = len(dd["has_delimiter"])
        
        # Get advantages and weights from precomputed data
        A_full = dd["A_full"]  # [n_responses]
        A_raw = dd.get("A_raw")  # [n_responses]
        F_norm = dd.get("fmt_norm") # [n_responses]
        w_full = dd["w_full"]  # [n_responses]
        has_delim = dd["has_delimiter"]  # list of booleans
        
        # Place back into batch tensors
        jepo_adv[current_idx:current_idx + n_responses] = A_full
        if A_raw is not None:
            jepo_adv_raw[current_idx:current_idx + n_responses] = A_raw
        if F_norm is not None:
            format_adv[current_idx:current_idx + n_responses] = F_norm
        jepo_weights[current_idx:current_idx + n_responses] = w_full
        has_delimiter[current_idx:current_idx + n_responses] = torch.tensor(has_delim, device=device)
        
        # Collect position information for log prob calculation
        all_cot_start_positions.extend(dd["cot_start_positions"])
        all_answer_start_positions.extend(dd["answer_start_positions"])
        all_cot_tokens_list.extend(dd["cot_tokens_list"])
        all_ground_truth_answer_tokens.extend(dd["ground_truth_answer_tokens"])
        
        current_idx += n_responses
    
    # Reconstruct batch_input_ids, attention_mask, position_ids from data_dicts
    # First, find the maximum sequence length across all data_dicts
    max_seq_len = 0
    all_batch_input_ids = []
    all_attention_mask = []
    all_position_ids = []
    
    for dd in data_dicts:
        max_seq_len = max(max_seq_len, dd["batch_input_ids"].shape[1])
        all_batch_input_ids.append(dd["batch_input_ids"])
        all_attention_mask.append(dd["attention_mask"])
        all_position_ids.append(dd["position_ids"])
    
    # Pad all tensors to the same length
    pad_token_id = cached_tokenizer.pad_token_id
    padded_batch_input_ids = []
    padded_attention_mask = []
    padded_position_ids = []
    
    for i, (input_ids, attn_mask, pos_ids) in enumerate(zip(all_batch_input_ids, all_attention_mask, all_position_ids)):
        current_len = input_ids.shape[1]
        if current_len < max_seq_len:
            pad_len = max_seq_len - current_len
            
            # Pad input_ids with pad_token_id
            padded_input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
            
            # Pad attention_mask with 0s
            padded_attn_mask = F.pad(attn_mask, (0, pad_len), value=0)
            
            # Pad position_ids with 0s
            padded_pos_ids = F.pad(pos_ids, (0, pad_len), value=0)
            
            padded_batch_input_ids.append(padded_input_ids)
            padded_attention_mask.append(padded_attn_mask)
            padded_position_ids.append(padded_pos_ids)
        else:
            padded_batch_input_ids.append(input_ids)
            padded_attention_mask.append(attn_mask)
            padded_position_ids.append(pos_ids)
    
    # Concatenate all tensors
    batch_input_ids = torch.cat(padded_batch_input_ids, dim=0)
    attention_mask = torch.cat(padded_attention_mask, dim=0)
    position_ids = torch.cat(padded_position_ids, dim=0)
    
    # Convert position information to tensors for data.batch
    # Pad cot_tokens_list and ground_truth_answer_tokens to same length for batching
    max_cot_len = max(len(tokens) for tokens in all_cot_tokens_list) if all_cot_tokens_list else 0
    max_ans_len = max(len(tokens) for tokens in all_ground_truth_answer_tokens) if all_ground_truth_answer_tokens else 0
    
    # Create padded tensors
    cot_start_positions_tensor = torch.tensor(all_cot_start_positions, device=device, dtype=torch.long)
    answer_start_positions_tensor = torch.tensor(all_answer_start_positions, device=device, dtype=torch.long)
    
    # Pad token lists
    padded_cot_tokens = torch.full((batch_size, max_cot_len), pad_token_id, device=device, dtype=torch.long)
    padded_ans_tokens = torch.full((batch_size, max_ans_len), pad_token_id, device=device, dtype=torch.long)
    
    for i, cot_tokens in enumerate(all_cot_tokens_list):
        if len(cot_tokens) > 0:
            padded_cot_tokens[i, :len(cot_tokens)] = torch.tensor(cot_tokens, device=device, dtype=torch.long)
    
    for i, ans_tokens in enumerate(all_ground_truth_answer_tokens):
        if len(ans_tokens) > 0:
            padded_ans_tokens[i, :len(ans_tokens)] = torch.tensor(ans_tokens, device=device, dtype=torch.long)
    
    # Add to data.batch
    data.batch["jepo_adv"] = jepo_adv
    data.batch["jepo_adv_raw"] = jepo_adv_raw
    data.batch["format_adv"] = format_adv
    data.batch["jepo_weights"] = jepo_weights
    data.batch["has_delimiter"] = has_delimiter
    data.batch["batch_input_ids"] = batch_input_ids
    data.batch["batch_attention_mask"] = attention_mask
    data.batch["batch_position_ids"] = position_ids
    data.batch["cot_start_positions"] = cot_start_positions_tensor
    data.batch["answer_start_positions"] = answer_start_positions_tensor
    data.batch["cot_tokens"] = padded_cot_tokens
    data.batch["ground_truth_tokens"] = padded_ans_tokens
    
    return data
