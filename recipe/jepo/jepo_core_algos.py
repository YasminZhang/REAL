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
from verl import DataProto

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
    beta_supp: float = 1e-3
    beta_kl: float = 1e-3
    buffer_size: int = 1000
    jepo_steps: int = 5
    epochs: int = 1
    mini_batch_size_per_gpu: int = 8  # questions per optimizer step per rank
    micro_batch_size_per_gpu: int = 1  # questions per backward pass per rank
    num_response_per_question: int = 8
    accum_steps: int = 4  # fixed accumulate steps for consistent backwards
    responses_micro_batch_size: int = 8


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
    # print('enable suffix_anchor:', enable_suffix_anchor)
    
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
    
    # print('Delimiter str:', delimiter_str)
    if delimiter_str[0] != ' ':
        delimiter_str = ' ' + delimiter_str

    delimiter_ids = tokenizer.encode(delimiter_str, add_special_tokens=False)
    delimiter_len = len(delimiter_ids)
    gt_list: List[List[int]] = [list(x) if isinstance(x, (list, tuple)) else list(x.tolist()) for x in ground_truth_answer_tokens]

    has_delimiter: List[bool] = []
    cot_tokens_list: List[List[int]] = []
    delimiter_match_kind: List[str] = []

    batch_input_tokens: List[torch.Tensor] = []
    cot_start_positions: List[int] = []
    answer_start_positions: List[int] = []

    eos_id = getattr(tokenizer, "eos_token_id", None)

    for i in range(n):
        resp_i = response_tokens[i].detach().clone()
        
        # find the first position of EOS in resp_i
        s, match_kind = _find_delimiter_position(
            resp_i, [eos_id],
            enable_suffix_anchor=delimiter_suffix_anchor,
            min_suffix_len=max(1, delimiter_suffix_min_len),
        )

        # print(f"Response {i}: found delimiter at {s}, kind={match_kind}")  

        prompt_i = prompt_tokens[i] if isinstance(prompt_tokens, torch.Tensor) else torch.tensor(prompt_tokens[i], device=device)
        prompt_i = prompt_i.to(device=device, dtype=torch.long)

        if s > 0:
            # Found EOS at s.
            # Last token before EOS is at s-1.
            # CoT is everything before s-1.
            cot_ids = resp_i[:s].tolist()
            
            # Use GT answer
            # gt_i = torch.tensor(gt_list[i], device=device, dtype=torch.long)
            cot_i = torch.tensor(cot_ids, device=device, dtype=torch.long)
            
            full = torch.cat([prompt_i, cot_i], dim=0)
            
            batch_input_tokens.append(full)
            cot_start_positions.append(len(prompt_i))
            answer_start_positions.append(len(prompt_i) + len(cot_i) - 1)
            
            has_delimiter.append(True)
            delimiter_match_kind.append(match_kind)
            cot_tokens_list.append(cot_ids)
        else:
            # EOS not found. Fallback: treat last token as answer.
            if len(resp_i) > 0:
                cot_ids = resp_i.tolist()
            else:
                cot_ids = []
                
            # gt_i = torch.tensor(gt_list[i], device=device, dtype=torch.long)
            cot_i = torch.tensor(cot_ids, device=device, dtype=torch.long)
            
            full = torch.cat([prompt_i, cot_i], dim=0)
            
            batch_input_tokens.append(full)
            cot_start_positions.append(len(prompt_i))
            answer_start_positions.append(len(prompt_i) + len(cot_i) -1 )
            
            has_delimiter.append(False)
            delimiter_match_kind.append("none")
            cot_tokens_list.append(cot_ids)
            breakpoint()
         

    # Validate pad token id
    if pad_token is None:
        raise ValueError("pad_token_id is required but missing (None)")

    max_len = max(int(t.numel()) for t in batch_input_tokens) if batch_input_tokens else 0
    padded_tokens: List[torch.Tensor] = []
    attention_masks: List[List[int]] = []

    eos_id = getattr(tokenizer, "eos_token_id", None)

    for t in batch_input_tokens:
        pad_len = max_len - int(t.numel())
        if pad_len > 0:
            padding = torch.full((pad_len,), pad_token, dtype=torch.long, device=device)
            padded = torch.cat([t, padding], dim=0)
        else:
            padded = t
        # Build attention mask by masking pad and eos tokens anywhere
        mask_t = torch.ones_like(padded, dtype=torch.long, device=device)
        mask_t = mask_t * (padded != pad_token).long()
        if eos_id is not None:
            mask_t = mask_t * (padded != eos_id).long()
        padded_tokens.append(padded)
        attention_masks.append(mask_t.tolist())

    batch_input_ids = (
        torch.stack(padded_tokens).to(dtype=torch.long, device=device)
        if padded_tokens
        else torch.empty((0, 0), dtype=torch.long, device=device)
    )
    attention_mask = (
        torch.tensor(attention_masks, dtype=torch.long, device=device)
        if attention_masks
        else torch.empty((0, 0), dtype=torch.long, device=device)
    )
    # Derive position_ids from attention_mask so first attended token has pos=0
    if max_len > 0:
        position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0) * attention_mask
    else:
        position_ids = torch.empty((n, 0), dtype=torch.long, device=device)

    # let gt_list be the second token in each entry
    gt_list = [(gt[1],) if len(gt) > 1 else gt for gt in gt_list]

  

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
    # dynamic chunking is used downstream; no fixed responses_micro_batch_size
    
    # NEW: Option to use last token position instead of delimiter
    use_last_token_as_answer = bool(jepo_cfg.get("use_last_token_as_answer", False))
    answer_token_length = int(jepo_cfg.get("answer_token_length", 1))  # How many tokens to treat as answer
    
    delimiter = jepo_cfg.get("delimiter", " So the overall score is ")
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
    
    # Build teacher-forced batches per question
    # if use_last_token_as_answer:
    #     # NEW: Use last token position instead of delimiter
    #     data_dicts = build_jepo_batches_by_prompt_last_token(
    #         response_tokens=data.batch["responses"],
    #         prompt_tokens=data.batch["prompts"],
    #         ground_truth_answer_tokens=ground_truths_tokens,
    #         answer_token_length=answer_token_length,
    #         device=(next(model.parameters()).device),
    #         pad_token=pad_token,
    #         index=data.non_tensor_batch["uid"],
    #         tokenizer=cached_tokenizer,
    #     )
    # else:
        # Original: Use delimiter-based approach
    data_dicts = build_jepo_batches_by_prompt(
        response_tokens=data.batch["responses"],
        prompt_tokens=data.batch["prompts"],
        ground_truth_answer_tokens=ground_truths_tokens,
        delimiter_str=delimiter,
        format_penalty=format_penalty,
        model=model,
        device=(next(model.parameters()).device),
        pad_token=pad_token,
        index=data.non_tensor_batch["uid"],
        tokenizer=cached_tokenizer,
        delimiter_suffix_anchor=use_suffix_anchor,
        delimiter_suffix_min_len=suffix_min_len,
    )
    # Only attach per-question teacher-forced packs; JEPO actor computes A/w later using VERL internals
    data.non_tensor_batch["jepo_data_dicts"] = data_dicts
    # Flatten teacher-forced fields into top-level batch for per-response slicing in the JEPO actor
    flat_batch_input_ids = []
    flat_attention_mask = []
    flat_position_ids = []
    flat_cot_start = []
    flat_ans_start = []
    flat_has_delim = []
    flat_gt_tokens = []

    for dd in data_dicts:
        if dd.get("batch_input_ids") is not None and dd["batch_input_ids"].numel() > 0:
            flat_batch_input_ids.append(dd["batch_input_ids"])  # [B_i, T_i]
            # attention_mask is long 0/1; keep as long to match typical masks
            flat_attention_mask.append(dd["attention_mask"].to(dtype=dd["batch_input_ids"].dtype))
            flat_position_ids.append(dd["position_ids"])
        if "cot_start_positions" in dd:
            flat_cot_start.extend([int(x) for x in dd["cot_start_positions"]])
        if "answer_start_positions" in dd:
            flat_ans_start.extend([int(x) for x in dd["answer_start_positions"]])
        if "has_delimiter" in dd:
            flat_has_delim.extend([bool(x) for x in dd["has_delimiter"]])
        if "ground_truth_answer_tokens" in dd:
            flat_gt_tokens.extend([list(x) for x in dd["ground_truth_answer_tokens"]])

    dev = data.batch["responses"].device
    if flat_batch_input_ids:
        # Pad all chunks to a global max length before concatenation
        lens = [int(t.size(1)) for t in flat_batch_input_ids]
        global_max_len = max(lens)
        padded_ids = []
        padded_attn = []
        padded_pos = []
        for ids_t, attn_t, pos_t in zip(flat_batch_input_ids, flat_attention_mask, flat_position_ids):
            cur_len = int(ids_t.size(1))
            if cur_len < global_max_len:
                pad_w = global_max_len - cur_len
                pad_ids = torch.full((ids_t.size(0), pad_w), pad_token, dtype=ids_t.dtype, device=ids_t.device)
                pad_mask = torch.zeros((attn_t.size(0), pad_w), dtype=attn_t.dtype, device=attn_t.device)
                pad_pos = torch.zeros((pos_t.size(0), pad_w), dtype=pos_t.dtype, device=pos_t.device)
                ids_t = torch.cat([ids_t, pad_ids], dim=1)
                attn_t = torch.cat([attn_t, pad_mask], dim=1)
                pos_t = torch.cat([pos_t, pad_pos], dim=1)
            padded_ids.append(ids_t)
            padded_attn.append(attn_t)
            padded_pos.append(pos_t)
        data.batch["batch_input_ids"] = torch.cat(padded_ids, dim=0)
        data.batch["attention_mask"] = torch.cat(padded_attn, dim=0)
        data.batch["position_ids"] = torch.cat(padded_pos, dim=0)
    if flat_cot_start:
        data.batch["cot_start_positions"] = torch.as_tensor(flat_cot_start, dtype=torch.long, device=dev)
    if flat_ans_start:
        data.batch["answer_start_positions"] = torch.as_tensor(flat_ans_start, dtype=torch.long, device=dev)
    if flat_has_delim:
        data.batch["has_delimiter"] = torch.as_tensor(flat_has_delim, dtype=torch.bool, device=dev)
    # keep ground_truth_answer_tokens in non_tensor for variable lengths
    # store as numpy object array to enable safe slicing/indexing
    try:
        data.non_tensor_batch["ground_truth_answer_tokens"] = np.array(flat_gt_tokens, dtype=object)
    except Exception:
        data.non_tensor_batch["ground_truth_answer_tokens"] = flat_gt_tokens
    return data
