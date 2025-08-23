"""
CoT Log Probability Computation (2D responses)

This module computes CoT log-probability ratios using the existing actor
worker group's compute_log_prob interface. It supports 2D "responses"
([B, T]) and creates two synthetic evaluation batches per row:

- With CoT:  prompt + cot + answer  -> compute π_θ(a|c,x)
- Without:   prompt + answer        -> compute π_θ(a|x)

It returns one result dict per row containing the log-probs and ratio.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from verl import DataProto
from recipe.cot_reward.cot_reward_function import (
    extract_ground_truth_answer,
    split_response_by_delimiter,
)


@dataclass
class CoTLogProbConfig:
    delimiter: str = "\\boxed{"
    max_sequence_length: int = 8192
    truncate_tokens: int = 50


def _process_single_response(
    prompt_str: str,
    response_str: str,
    ground_truth: str,
    tokenizer,
    config: CoTLogProbConfig,
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[int], Dict[str, Any]]:
    """Build token sequences for with/without-CoT and answer tokens.

    Returns eight lists (ids/masks/pos/answers for with/without) and a metadata dict.
    """
    gt_answer = extract_ground_truth_answer(ground_truth)
    if not gt_answer:
        gt_answer = "Unknown"

    cot_part, _ = split_response_by_delimiter(response_str, config.delimiter)
    has_delim = config.delimiter in response_str

    prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
    # Ensure there is at least one prefix token to align logits for first answer token
    if len(prompt_tokens) == 0:
        bos_id = getattr(tokenizer, "bos_token_id", None)
        fallback_id = bos_id if bos_id is not None else (
            getattr(tokenizer, "eos_token_id", None) if getattr(tokenizer, "eos_token_id", None) is not None else (
                getattr(tokenizer, "pad_token_id", None) if getattr(tokenizer, "pad_token_id", None) is not None else 0
            )
        )
        prompt_tokens = [fallback_id]
    cot_tokens = tokenizer.encode(cot_part, add_special_tokens=False)
    # Target answer includes the delimiter prefix
    delimiter_tokens = tokenizer.encode(str(config.delimiter), add_special_tokens=False)
    gt_only_tokens = tokenizer.encode(str(gt_answer), add_special_tokens=False)
    answer_target_tokens = delimiter_tokens + gt_only_tokens

    # Truncate CoT a bit if there is no delimiter to keep sequence under limit
    if not has_delim and config.truncate_tokens > 0 and len(cot_tokens) > config.truncate_tokens:
        cot_tokens = cot_tokens[:-config.truncate_tokens]

    # With CoT
    tokens_with_cot = prompt_tokens + cot_tokens + answer_target_tokens
    attention_mask_with_cot = [1] * len(tokens_with_cot)
    position_ids_with_cot = list(range(len(tokens_with_cot)))
    responses_with_cot = answer_target_tokens

    # Without CoT
    tokens_without_cot = prompt_tokens + answer_target_tokens
    attention_mask_without_cot = [1] * len(tokens_without_cot)
    position_ids_without_cot = list(range(len(tokens_without_cot)))
    responses_without_cot = answer_target_tokens

    metadata = {
        "cot_part": cot_tokens,
        "gt_answer": gt_answer,
        "has_delimiter": has_delim,
        # Allow selecting only gt tokens (exclude delimiter) when computing sums
        "answer_offset": len(delimiter_tokens),
        "answer_len": len(gt_only_tokens),
    }

    return (
        tokens_with_cot,
        attention_mask_with_cot,
        position_ids_with_cot,
        responses_with_cot,
        tokens_without_cot,
        attention_mask_without_cot,
        position_ids_without_cot,
        responses_without_cot,
        metadata,
    )


def _pad_to_length(rows: List[List[int]], pad_id: int, max_len: int) -> torch.Tensor:
    out: List[List[int]] = []
    for x in rows:
        if len(x) < max_len:
            out.append(x + [pad_id] * (max_len - len(x)))
        else:
            out.append(x[: max_len])
    return torch.tensor(out, dtype=torch.long)


def _pad_mask(rows: List[List[int]], max_len: int) -> torch.Tensor:
    out: List[List[int]] = []
    for x in rows:
        if len(x) < max_len:
            out.append(x + [0] * (max_len - len(x)))
        else:
            out.append(x[: max_len])
    return torch.tensor(out, dtype=torch.long)


def _make_pos(ids: torch.Tensor) -> torch.Tensor:
    bsz, seqlen = ids.shape
    return torch.arange(seqlen, dtype=torch.long).unsqueeze(0).expand(bsz, seqlen)


def prepare_cot_log_prob_batches(
    batch: DataProto, tokenizer, config: CoTLogProbConfig
) -> Tuple[DataProto, DataProto, List[Dict[str, Any]]]:
    """Prepare two DataProto batches to compute log-probs with/without CoT.

    Expects batch.batch to contain "prompts" and "responses" (2D). Ground truth is
    read from non_tensor_batch["reward_model"][i]["ground_truth"].
    """
    assert "prompts" in batch.batch and "responses" in batch.batch
    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    assert responses.dim() == 2, "responses must be 2D [B, T]"
    B = responses.shape[0]

    all_with = {"input_ids": [], "attention_mask": [], "position_ids": [], "responses": []}
    all_wo = {"input_ids": [], "attention_mask": [], "position_ids": [], "responses": []}
    metadata_list: List[Dict[str, Any]] = []

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for i in range(B):
        prompt_ids = prompts[i]
        # Robust to left/right truncation and any padding: rely on skip_special_tokens
        prompt_str = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        resp_ids = responses[i]
        response_str = tokenizer.decode(resp_ids, skip_special_tokens=True)

        reward_info = batch[i].non_tensor_batch.get("reward_model", {})
        gt = reward_info.get("ground_truth", "")

        (
            ids_w,
            mask_w,
            pos_w,
            ans_w,
            ids_wo,
            mask_wo,
            pos_wo,
            ans_wo,
            md,
        ) = _process_single_response(prompt_str, response_str, gt, tokenizer, config)

        all_with["input_ids"].append(ids_w)
        all_with["attention_mask"].append(mask_w)
        all_with["position_ids"].append(pos_w)
        all_with["responses"].append(ans_w)

        all_wo["input_ids"].append(ids_wo)
        all_wo["attention_mask"].append(mask_wo)
        all_wo["position_ids"].append(pos_wo)
        all_wo["responses"].append(ans_wo)

        metadata_list.append(md)

    max_len_with = min(
        config.max_sequence_length, max(len(x) for x in all_with["input_ids"]) if B > 0 else 0
    )
    max_len_wo = min(
        config.max_sequence_length, max(len(x) for x in all_wo["input_ids"]) if B > 0 else 0
    )
    max_ans_with = max((len(x) for x in all_with["responses"]), default=0)
    max_ans_wo = max((len(x) for x in all_wo["responses"]), default=0)

    ids_w = _pad_to_length(all_with["input_ids"], pad_token_id, max_len_with)
    mask_w = _pad_mask(all_with["attention_mask"], max_len_with)
    pos_w = _make_pos(ids_w)
    ans_w = _pad_to_length(all_with["responses"], pad_token_id, max_ans_with)

    ids_wo = _pad_to_length(all_wo["input_ids"], pad_token_id, max_len_wo)
    mask_wo = _pad_mask(all_wo["attention_mask"], max_len_wo)
    pos_wo = _make_pos(ids_wo)
    ans_wo = _pad_to_length(all_wo["responses"], pad_token_id, max_ans_wo)

    batch_with_cot = DataProto.from_dict(
        tensors={
            "input_ids": ids_w,
            "attention_mask": mask_w,
            "position_ids": pos_w,
            "responses": ans_w,
        }
    )
    batch_without_cot = DataProto.from_dict(
        tensors={
            "input_ids": ids_wo,
            "attention_mask": mask_wo,
            "position_ids": pos_wo,
            "responses": ans_wo,
        }
    )

    return batch_with_cot, batch_without_cot, metadata_list


def compute_cot_log_prob_ratios(
    batch: DataProto, actor_worker_group, tokenizer, config: CoTLogProbConfig
) -> List[Dict[str, Any]]:
    """Compute per-row CoT ratios using actor workers.

    Returns a list of dicts aligned with rows in the input batch.
    """
    batch_w, batch_wo, md_list = prepare_cot_log_prob_batches(batch, tokenizer, config)
    out_w = actor_worker_group.compute_log_prob(batch_w)
    out_wo = actor_worker_group.compute_log_prob(batch_wo)
    breakpoint()

    # worker returns key 'old_log_probs' for log-prob tensors of targets
    lp_w = out_w.batch["old_log_probs"]  # [B, L_ans]
    lp_wo = out_wo.batch["old_log_probs"]  # [B, L_ans]

    results: List[Dict[str, Any]] = []
    for i, md in enumerate(md_list):
        try:
            off = int(md.get("answer_offset", 0))
            ln = int(md.get("answer_len", lp_w.size(1)))
            # Guard invalid/empty windows to avoid zero-length sum
            if ln <= 0 or off < 0 or off >= lp_w.size(1):
                off, ln = 0, lp_w.size(1)
            with_cot = float(lp_w[i, off:off + ln].sum().item())
            without_cot = float(lp_wo[i, off:off + ln].sum().item())
            log_ratio = with_cot - without_cot
            ratio = float(torch.exp(torch.tensor(log_ratio)).clamp(min=0.0, max=10.0).item())
            results.append(
                {
                    "log_prob_with_cot": with_cot,
                    "log_prob_without_cot": without_cot,
                    "log_ratio": log_ratio,
                    "ratio": ratio,
                    "has_valid_gt": bool(md.get("gt_answer")),
                    "cot_length": len(md.get("cot_part", "")),
                    "has_delimiter": bool(md.get("has_delimiter", False)),
                    "cot_part": md.get("cot_part", ""),
                    "gt_answer": md.get("gt_answer", ""),
                }
            )
        except Exception as e:
            results.append(
                {
                    "log_prob_with_cot": float("-inf"),
                    "log_prob_without_cot": float("-inf"),
                    "log_ratio": float("-inf"),
                    "ratio": 0.0,
                    "has_valid_gt": False,
                    "cot_length": 0,
                    "has_delimiter": False,
                    "error": str(e),
                }
            )

    return results
