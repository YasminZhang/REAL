"""
CoT Log Probability Computation

This module implements efficient CoT log probability calculation using existing
actor worker infrastructure. It handles:
- Response splitting by delimiter
- Padding and attention mask management
- Batch processing for π_θ(a|c,x) and π_θ(a|x) calculation
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from copy import deepcopy

from verl import DataProto
from recipe.cot_reward.cot_reward_function import split_response_by_delimiter, extract_ground_truth_answer


@dataclass
class CoTLogProbConfig:
    """Configuration for CoT log probability computation."""
    delimiter: str = "\\boxed{"
    max_sequence_length: int = 8192
    truncate_tokens: int = 50  # Number of tokens to remove from end of CoT if no delimiter found


def prepare_cot_log_prob_batches(
    batch: DataProto,
    tokenizer,
    config: CoTLogProbConfig
) -> Tuple[DataProto, DataProto, List[Dict[str, Any]]]:
    """
    Prepare two batches for computing π_θ(a|c,x) and π_θ(a|x).
    
    Args:
        batch: Original DataProto with responses
        tokenizer: Tokenizer for encoding/decoding
        config: Configuration for CoT computation
        
    Returns:
        Tuple of:
        - batch_with_cot: DataProto for computing π_θ(a|c,x)
        - batch_without_cot: DataProto for computing π_θ(a|x)  
        - metadata: List of metadata for each sample
    """
    responses = batch.batch["responses"]  # [batch_size, n_responses, response_length]
    batch_size, n_responses = responses.shape[:2]
    
    # Get prompts and ground truths
    prompts = batch.batch.get("input_ids", batch.batch.get("prompts"))
    ground_truths = []
    
    # Extract ground truths from batch
    for i in range(batch_size):
        if hasattr(batch, '__getitem__'):
            item = batch[i]
            gt = item.non_tensor_batch.get("ground_truth")
            if gt is None:
                reward_model_info = item.non_tensor_batch.get("reward_model", {})
                gt = reward_model_info.get("ground_truth", "")
        else:
            gt = ""
        ground_truths.append(gt)
    
    # Prepare batches for all responses
    all_input_ids_with_cot = []
    all_attention_mask_with_cot = []
    all_input_ids_without_cot = []
    all_attention_mask_without_cot = []
    all_metadata = []
    
    for batch_idx in range(batch_size):
        prompt_tokens = prompts[batch_idx] if prompts is not None else None
        ground_truth = ground_truths[batch_idx]
        
        # Decode prompt if needed
        if prompt_tokens is not None:
            # Find actual prompt length (before padding)
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                actual_prompt_length = (prompt_tokens != tokenizer.pad_token_id).sum().item()
            else:
                actual_prompt_length = len(prompt_tokens)
            prompt_str = tokenizer.decode(prompt_tokens[:actual_prompt_length], skip_special_tokens=True)
        else:
            prompt_str = "Solve the following problem:"
            
        for resp_idx in range(n_responses):
            response_tokens = responses[batch_idx, resp_idx]
            
            # Decode response
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                actual_response_length = (response_tokens != tokenizer.pad_token_id).sum().item()
            else:
                actual_response_length = len(response_tokens)
            response_str = tokenizer.decode(response_tokens[:actual_response_length], skip_special_tokens=True)
            
            # Process this response
            input_ids_with_cot, attention_mask_with_cot, input_ids_without_cot, attention_mask_without_cot, metadata = \
                _process_single_response(
                    prompt_str, response_str, ground_truth, tokenizer, config
                )
            
            all_input_ids_with_cot.append(input_ids_with_cot)
            all_attention_mask_with_cot.append(attention_mask_with_cot)
            all_input_ids_without_cot.append(input_ids_without_cot)
            all_attention_mask_without_cot.append(attention_mask_without_cot)
            all_metadata.append(metadata)
    
    # Pad to same length and create DataProto batches
    max_length = min(config.max_sequence_length, max(
        max(len(seq) for seq in all_input_ids_with_cot),
        max(len(seq) for seq in all_input_ids_without_cot)
    ))
    
    # Pad sequences
    padded_input_ids_with_cot = []
    padded_attention_mask_with_cot = []
    padded_input_ids_without_cot = []
    padded_attention_mask_without_cot = []
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    for i in range(len(all_input_ids_with_cot)):
        # Pad with_cot sequences
        seq_with_cot = all_input_ids_with_cot[i]
        mask_with_cot = all_attention_mask_with_cot[i]
        if len(seq_with_cot) < max_length:
            padding_length = max_length - len(seq_with_cot)
            seq_with_cot = seq_with_cot + [pad_token_id] * padding_length
            mask_with_cot = mask_with_cot + [0] * padding_length
        else:
            seq_with_cot = seq_with_cot[:max_length]
            mask_with_cot = mask_with_cot[:max_length]
        
        padded_input_ids_with_cot.append(seq_with_cot)
        padded_attention_mask_with_cot.append(mask_with_cot)
        
        # Pad without_cot sequences
        seq_without_cot = all_input_ids_without_cot[i]
        mask_without_cot = all_attention_mask_without_cot[i]
        if len(seq_without_cot) < max_length:
            padding_length = max_length - len(seq_without_cot)
            seq_without_cot = seq_without_cot + [pad_token_id] * padding_length
            mask_without_cot = mask_without_cot + [0] * padding_length
        else:
            seq_without_cot = seq_without_cot[:max_length]
            mask_without_cot = mask_without_cot[:max_length]
            
        padded_input_ids_without_cot.append(seq_without_cot)
        padded_attention_mask_without_cot.append(mask_without_cot)
    
    # Create DataProto batches
    batch_with_cot = DataProto(
        batch={
            "input_ids": torch.tensor(padded_input_ids_with_cot, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask_with_cot, dtype=torch.long),
        },
        non_tensor_batch={}
    )
    
    batch_without_cot = DataProto(
        batch={
            "input_ids": torch.tensor(padded_input_ids_without_cot, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask_without_cot, dtype=torch.long),
        },
        non_tensor_batch={}
    )
    
    return batch_with_cot, batch_without_cot, all_metadata


def _process_single_response(
    prompt_str: str,
    response_str: str,
    ground_truth: str,
    tokenizer,
    config: CoTLogProbConfig
) -> Tuple[List[int], List[int], List[int], List[int], Dict[str, Any]]:
    """
    Process a single response to create input sequences for log probability calculation.
    
    Returns:
        Tuple of:
        - input_ids_with_cot: tokens for π_θ(a|c,x) calculation
        - attention_mask_with_cot: attention mask for with_cot
        - input_ids_without_cot: tokens for π_θ(a|x) calculation  
        - attention_mask_without_cot: attention mask for without_cot
        - metadata: metadata about the processing
    """
    # Extract ground truth answer
    gt_answer = extract_ground_truth_answer(ground_truth)
    if not gt_answer:
        gt_answer = "Unknown"
    
    # Split response by delimiter
    cot_part, answer_part = split_response_by_delimiter(response_str, config.delimiter)
    
    # Handle case where no delimiter is found
    if config.delimiter not in response_str:
        # Treat whole response as CoT, truncate end and append delimiter + ground truth
        cot_tokens = tokenizer.encode(cot_part, add_special_tokens=False)
        
        # Remove last few tokens to make room for delimiter + answer
        gt_tokens = tokenizer.encode(f"{config.delimiter}{gt_answer}", add_special_tokens=False)
        available_length = config.max_sequence_length - len(tokenizer.encode(prompt_str, add_special_tokens=False)) - len(gt_tokens) - 10  # 10 buffer tokens
        
        if len(cot_tokens) > available_length:
            # Remove tokens from end
            tokens_to_remove = max(config.truncate_tokens, len(cot_tokens) - available_length)
            cot_tokens = cot_tokens[:-tokens_to_remove]
        
        # Reconstruct CoT part without delimiter
        cot_part_no_delimiter = tokenizer.decode(cot_tokens, skip_special_tokens=True)
        # CoT part with delimiter for π_θ(a|c,x)
        cot_part_with_delimiter = cot_part_no_delimiter + config.delimiter
    else:
        # Delimiter found, split properly
        # For π_θ(a|c,x): CoT includes delimiter
        cot_part_with_delimiter = cot_part + config.delimiter
        # For π_θ(a|x): CoT without delimiter
        cot_part_no_delimiter = cot_part
    
    # Create sequences
    # For π_θ(a|c,x): prompt + cot_with_delimiter + answer
    seq_with_cot = f"{prompt_str}{cot_part_with_delimiter}{gt_answer}"
    
    # For π_θ(a|x): prompt + answer (no CoT)
    seq_without_cot = f"{prompt_str}{gt_answer}"
    
    # Tokenize sequences
    tokens_with_cot = tokenizer.encode(seq_with_cot, add_special_tokens=True)
    tokens_without_cot = tokenizer.encode(seq_without_cot, add_special_tokens=True)
    
    # Calculate where the answer starts in each sequence
    prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=True)
    cot_with_delimiter_tokens = tokenizer.encode(cot_part_with_delimiter, add_special_tokens=False)
    answer_tokens = tokenizer.encode(gt_answer, add_special_tokens=False)
    
    # For with_cot: answer starts after prompt + cot_with_delimiter
    answer_start_with_cot = len(prompt_tokens) + len(cot_with_delimiter_tokens) - 1  # -1 because we don't double count special tokens
    
    # For without_cot: answer starts after prompt
    answer_start_without_cot = len(prompt_tokens) - 1  # -1 because we don't double count special tokens
    
    # Create attention masks (1 for valid tokens, 0 for padding)
    attention_mask_with_cot = [1] * len(tokens_with_cot)
    attention_mask_without_cot = [1] * len(tokens_without_cot)
    
    # Create metadata
    metadata = {
        "cot_part": cot_part_no_delimiter,
        "gt_answer": gt_answer,
        "answer_start_with_cot": answer_start_with_cot,
        "answer_start_without_cot": answer_start_without_cot,
        "answer_length": len(answer_tokens),
        "has_delimiter": config.delimiter in response_str,
        "total_length_with_cot": len(tokens_with_cot),
        "total_length_without_cot": len(tokens_without_cot)
    }
    
    return tokens_with_cot, attention_mask_with_cot, tokens_without_cot, attention_mask_without_cot, metadata


def extract_answer_log_probs(
    log_probs_with_cot: torch.Tensor,
    log_probs_without_cot: torch.Tensor,
    metadata_list: List[Dict[str, Any]]
) -> List[Dict[str, float]]:
    """
    Extract log probabilities for answer tokens and compute ratios.
    
    Args:
        log_probs_with_cot: Log probabilities from π_θ(a|c,x) computation
        log_probs_without_cot: Log probabilities from π_θ(a|x) computation  
        metadata_list: Metadata for each sample
        
    Returns:
        List of dictionaries containing log probability ratios and metadata
    """
    results = []
    
    for i, metadata in enumerate(metadata_list):
        try:
            # Extract answer log probabilities for with_cot
            answer_start_with_cot = metadata["answer_start_with_cot"]
            answer_length = metadata["answer_length"]
            answer_end_with_cot = answer_start_with_cot + answer_length
            
            if answer_end_with_cot <= log_probs_with_cot.shape[1]:
                answer_log_probs_with_cot = log_probs_with_cot[i, answer_start_with_cot:answer_end_with_cot]
                log_prob_with_cot = answer_log_probs_with_cot.sum().item()
            else:
                log_prob_with_cot = float('-inf')
            
            # Extract answer log probabilities for without_cot
            answer_start_without_cot = metadata["answer_start_without_cot"]
            answer_end_without_cot = answer_start_without_cot + answer_length
            
            if answer_end_without_cot <= log_probs_without_cot.shape[1]:
                answer_log_probs_without_cot = log_probs_without_cot[i, answer_start_without_cot:answer_end_without_cot]
                log_prob_without_cot = answer_log_probs_without_cot.sum().item()
            else:
                log_prob_without_cot = float('-inf')
            
            # Calculate log ratio and convert to probability ratio
            if log_prob_without_cot != float('-inf') and log_prob_with_cot != float('-inf'):
                log_ratio = log_prob_with_cot - log_prob_without_cot
                ratio = torch.exp(torch.tensor(log_ratio)).item()
                # Clamp to reasonable range
                ratio = max(0.0, min(ratio, 10.0))
            else:
                log_ratio = float('-inf')
                ratio = 0.0
            
            result = {
                "log_prob_with_cot": log_prob_with_cot,
                "log_prob_without_cot": log_prob_without_cot,
                "log_ratio": log_ratio,
                "ratio": ratio,
                "has_valid_gt": bool(metadata["gt_answer"]),
                "cot_length": len(metadata["cot_part"]),
                "has_delimiter": metadata["has_delimiter"],
                "cot_part": metadata["cot_part"],
                "gt_answer": metadata["gt_answer"]
            }
            
        except Exception as e:
            result = {
                "log_prob_with_cot": float('-inf'),
                "log_prob_without_cot": float('-inf'),
                "log_ratio": float('-inf'),
                "ratio": 0.0,
                "has_valid_gt": False,
                "cot_length": 0,
                "has_delimiter": False,
                "error": str(e)
            }
        
        results.append(result)
    
    return results


def compute_cot_log_prob_ratios(
    batch: DataProto,
    actor_worker_group,
    tokenizer,
    config: CoTLogProbConfig
) -> List[List[Dict[str, Any]]]:
    """
    Main function to compute CoT log probability ratios for a batch.
    
    Args:
        batch: DataProto containing responses
        actor_worker_group: Actor worker group for log probability computation
        tokenizer: Tokenizer for text processing
        config: Configuration for CoT computation
        
    Returns:
        List of lists containing log probability data for each sample and response
    """
    # Prepare batches for log probability computation
    batch_with_cot, batch_without_cot, metadata_list = prepare_cot_log_prob_batches(
        batch, tokenizer, config
    )
    
    # Compute log probabilities using actor workers
    log_probs_result_with_cot = actor_worker_group.compute_log_prob(batch_with_cot)
    log_probs_result_without_cot = actor_worker_group.compute_log_prob(batch_without_cot)
    
    # Extract log probabilities tensors
    log_probs_with_cot = log_probs_result_with_cot.batch["log_probs"]
    log_probs_without_cot = log_probs_result_without_cot.batch["log_probs"]
    
    # Extract answer log probabilities and compute ratios
    results = extract_answer_log_probs(
        log_probs_with_cot, log_probs_without_cot, metadata_list
    )
    
    # Reshape results to match original batch structure [batch_size][n_responses]
    responses = batch.batch["responses"]
    batch_size, n_responses = responses.shape[:2]
    
    structured_results = []
    result_idx = 0
    for batch_idx in range(batch_size):
        sample_results = []
        for resp_idx in range(n_responses):
            sample_results.append(results[result_idx])
            result_idx += 1
        structured_results.append(sample_results)
    
    return structured_results
"""
CoT Log Probability Computation (2D response tensor compatible)

This module computes CoT log-probability ratios using the existing actor
worker group's compute_log_prob interface. It supports 2D "responses"
([B, T]) and creates two synthetic evaluation batches per row:

- With CoT:  prompt + cot + answer  -> compute π_θ(a|c,x)
- Without:   prompt + answer        -> compute π_θ(a|x)

It returns a list of dicts (length B), each containing the log-probs and
ratio for the row's ground-truth answer.
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
    cot_tokens = tokenizer.encode(cot_part, add_special_tokens=False)
    answer_tokens = tokenizer.encode(str(gt_answer), add_special_tokens=False)

    # Truncate CoT a bit if there is no delimiter to keep sequence under limit
    if not has_delim and config.truncate_tokens > 0 and len(cot_tokens) > config.truncate_tokens:
        cot_tokens = cot_tokens[:-config.truncate_tokens]

    # With CoT
    tokens_with_cot = prompt_tokens + cot_tokens + answer_tokens
    attention_mask_with_cot = [1] * len(tokens_with_cot)
    position_ids_with_cot = list(range(len(tokens_with_cot)))
    responses_with_cot = answer_tokens

    # Without CoT
    tokens_without_cot = prompt_tokens + answer_tokens
    attention_mask_without_cot = [1] * len(tokens_without_cot)
    position_ids_without_cot = list(range(len(tokens_without_cot)))
    responses_without_cot = answer_tokens

    metadata = {
        "cot_part": cot_part,
        "gt_answer": gt_answer,
        "has_delimiter": has_delim,
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
        # derive valid prompt length via attention mask-like check to avoid decoding pads
        if tokenizer.pad_token_id is not None:
            plen = int((prompt_ids != tokenizer.pad_token_id).sum().item())
        else:
            plen = prompt_ids.shape[-1]
        prompt_str = tokenizer.decode(prompt_ids[:plen], skip_special_tokens=True)

        resp_ids = responses[i]
        if tokenizer.pad_token_id is not None:
            rlen = int((resp_ids != tokenizer.pad_token_id).sum().item())
        else:
            rlen = resp_ids.shape[-1]
        response_str = tokenizer.decode(resp_ids[:rlen], skip_special_tokens=True)

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

    # compute per-batch lengths and pad
    max_len_with = min(
        config.max_sequence_length, max(len(x) for x in all_with["input_ids"]) if B > 0 else 0
    )
    max_len_wo = min(
        config.max_sequence_length, max(len(x) for x in all_wo["input_ids"]) if B > 0 else 0
    )
    # response (answer) lengths need to be unified within each batch
    max_ans_with = max((len(x) for x in all_with["responses"]), default=0)
    max_ans_wo = max((len(x) for x in all_wo["responses"]), default=0)

    # with CoT
    ids_w = _pad_to_length(all_with["input_ids"], pad_token_id, max_len_with)
    mask_w = _pad_mask(all_with["attention_mask"], max_len_with)
    pos_w = _make_pos(ids_w)
    # pad answers with pad_token (won't be scored since targets drive log-probs)
    ans_w = _pad_to_length(all_with["responses"], pad_token_id, max_ans_with)

    # without CoT
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

    # worker returns key 'old_log_probs' for log-prob tensors of targets
    lp_w = out_w.batch["old_log_probs"]  # [B, L_ans]
    lp_wo = out_wo.batch["old_log_probs"]  # [B, L_ans]

    results: List[Dict[str, Any]] = []
    for i, md in enumerate(md_list):
        try:
            with_cot = float(lp_w[i].sum().item())
            without_cot = float(lp_wo[i].sum().item())
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
