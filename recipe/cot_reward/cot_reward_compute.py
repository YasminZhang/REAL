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
import torch.nn.functional as F

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
        "cot_part": cot_part,
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


# ============================
# White-box teacher-forced log-prob helpers (optional)
# ============================

@torch.no_grad()
def logp_of_answer(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    temperature: float = 1.0,
    exclude_prefix_len: int = 0,
):
    """
    Returns (total_log_likelihood, n_target_tokens) for `answer` given `prompt`.
    Teacher-forced next-token prediction, summing only over answer tokens.
    """
    device = next(model.parameters()).device  # infer device

    full = prompt + answer
    enc_full = tokenizer(full, return_tensors="pt")
    input_ids = enc_full.input_ids.to(device)
    attn_mask = enc_full.attention_mask.to(device)

    enc_prompt = tokenizer(prompt, return_tensors="pt")
    n_prompt = enc_prompt.input_ids.shape[1]

    labels = input_ids.clone().to(device).long()
    labels[:, :n_prompt] = -100  # ignore prompt positions

    # Forward (fp32 for stability)
    with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float32, enabled=False):
        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits  # [B, T, V]

    # Shift for next-token prediction
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    # Ensure same device for all tensors
    dev = logits.device
    if labels.device != dev:
        labels = labels.to(dev)
    mask = (labels != -100)
    if mask.device != dev:
        mask = mask.to(dev)

    # Temperature scaling and per-token logprobs
    if temperature is not None and temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    logprobs = F.log_softmax(logits, dim=-1)  # [B, T-1, V]
    # Avoid invalid gather on masked positions (-100)
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    tgt_logprobs = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # Sum only over answer tokens, excluding the first `exclude_prefix_len` tokens
    if mask.any():
        cum = mask.int().cumsum(dim=1)
        keep = mask & (cum > exclude_prefix_len)
        total_logp = tgt_logprobs[keep].sum().item() if keep.any() else 0.0
        length = int(keep.sum().item())
    else:
        total_logp, length = 0.0, 0
    return total_logp, length


@torch.no_grad()
def compute_cot_pmi_whitebox(
    model,
    tokenizer,
    x: str,
    c: str,
    a_star: str,
    delimiter: str = "\\boxed{",
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Compute PMI-style CoT advantage with a direct model forward:
      - prompt_with_c:   x + c + "Final Answer:" (or delimiter)
      - prompt_without_c:x + "Final Answer:"
      - answer is delimiter + a_star (matches our pipeline)

    Returns dict with logp_c, logp_0, Lc, L0, pmi_log, pmi_norm, ratio.
    """
    # Build prompts
    # You can customize the separator/headers to match your dataset format
    prompt_with_c = f"{x}{c}"
    prompt_without_c = f"{x}"
    # Tokenize delimiter and GT to compute lengths; exclude delimiter from sum
    delim_ids = tokenizer.encode(delimiter, add_special_tokens=False)
    gt_ids = tokenizer.encode(str(a_star), add_special_tokens=False)
    answer = f"{delimiter}{a_star}"

    logp_c, _ = logp_of_answer(
        model, tokenizer, prompt_with_c, answer, temperature=temperature, exclude_prefix_len=len(delim_ids)
    )
    logp_0, _ = logp_of_answer(
        model, tokenizer, prompt_without_c, answer, temperature=temperature, exclude_prefix_len=len(delim_ids)
    )
    pmi_log = logp_c - logp_0
    # length-normalized PMI to reduce verbosity bias
    # length-normalized PMI to reduce verbosity bias (normalize by GT length only)
    Lc = max(len(gt_ids), 0)
    L0 = max(len(gt_ids), 0)
    pmi_norm = (logp_c / max(Lc, 1)) - (logp_0 / max(L0, 1))
    ratio = torch.exp(torch.tensor(pmi_log)).item()

    return {
        "logp_c": logp_c,
        "logp_0": logp_0,
        "Lc": Lc,
        "L0": L0,
        "pmi_log": pmi_log,
        "pmi_norm": pmi_norm,
        "ratio": ratio,
    }


# ============================
# ID-based white-box PMI (avoid prompt/CoT re-tokenization)
# ============================

def _possible_delimiter_tokenizations(tokenizer, delimiter: str) -> list[list[int]]:
    cands = [delimiter, " " + delimiter]
    ids = []
    for s in cands:
        try:
            t = tokenizer.encode(s, add_special_tokens=False)
            if t and t not in ids:
                ids.append(t)
        except Exception:
            pass
    return ids


def _find_subsequence(hay: list[int], needles: list[list[int]]) -> int | None:
    """Return start index of first occurrence of any needle in hay, or None.
    Tries each needle; robust to different leading-space BPE variants.
    """
    n = len(hay)
    for nd in needles:
        m = len(nd)
        if m == 0 or m > n:
            continue
        # search
        for i in range(0, n - m + 1):
            if hay[i : i + m] == nd:
                return i
    return None


@torch.no_grad()
def logp_of_answer_ids(model, input_ids_full: list[int], answer_total_len: int, delimiter_len: int, temperature: float = 1.0):
    """
    Compute sum of log-probs for only the GT tokens within the tail answer span (exclude delimiter).

    input_ids_full: prompt + (optional cot) + answer_target_ids (delimiter+gt)
    answer_total_len: len(delimiter+gt) ids at the end
    delimiter_len: number of ids belonging to delimiter at the head of answer span
    """
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids_full], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)
    T = ids.shape[1]
    # Labels: ignore everything before the answer span; include full answer span labels initially
    labels = ids.clone()
    start_answer = T - answer_total_len
    if start_answer > 0:
        labels[:, :start_answer] = -100

    # Forward
    with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float32, enabled=False):
        out = model(input_ids=ids, attention_mask=attn)
        logits = out.logits

    # Shift next-token prediction
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    dev = logits.device
    if labels.device != dev:
        labels = labels.to(dev)

    # Temperature
    if temperature is not None and temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    logprobs = F.log_softmax(logits, dim=-1)

    # Gather all positions (we will sum only gt positions)
    mask_all = (labels != -100)
    safe_labels = labels.clone()
    safe_labels[~mask_all] = 0
    tgt_logprobs = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    # Build mask for gt-only inside answer span (exclude delimiter ids)
    # After shift, the answer span corresponds to indices [start_answer-1 : T-1] in logits/labels
    # We keep only the last (answer_total_len - delimiter_len) positions
    gt_len = max(answer_total_len - delimiter_len, 0)
    if gt_len == 0:
        return 0.0, 0
    # start index in shifted space
    start_in_shift = max(start_answer - 1, 0) + delimiter_len
    end_in_shift = start_in_shift + gt_len
    idx = torch.arange(tgt_logprobs.shape[1], device=dev)
    gt_mask = (idx >= start_in_shift) & (idx < end_in_shift)

    total_logp = tgt_logprobs[0][gt_mask].sum().item()
    length = int(gt_mask.sum().item())
    return total_logp, length


@torch.no_grad()
def compute_cot_pmi_whitebox_ids(
    model,
    tokenizer,
    prompt_ids: list[int],
    response_ids: list[int],
    gt_text: str,
    delimiter: str = "\\boxed{",
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Build with/without-CoT input ids using existing prompt/response ids and GT text,
    then compute PMI on ids without re-tokenizing the prompt/CoT.
    """
    # Find delimiter position inside response ids (handle leading-space variants)
    del_tok_cands = _possible_delimiter_tokenizations(tokenizer, delimiter)
    split_idx = _find_subsequence(response_ids, del_tok_cands)
    cot_ids = response_ids[:split_idx] if split_idx is not None else response_ids

    # Tokenize answer target (delimiter + gt_text)
    delim_ids = tokenizer.encode(delimiter, add_special_tokens=False)
    gt_ids = tokenizer.encode(str(gt_text), add_special_tokens=False)
    answer_target = delim_ids + gt_ids

    # Compose full sequences
    full_with = prompt_ids + cot_ids + answer_target
    full_wo = prompt_ids + answer_target

    # Compute teacher-forced logp for GT-only within answer span
    logp_c, Lc = logp_of_answer_ids(
        model, full_with, answer_total_len=len(answer_target), delimiter_len=len(delim_ids), temperature=temperature
    )
    logp_0, L0 = logp_of_answer_ids(
        model, full_wo, answer_total_len=len(answer_target), delimiter_len=len(delim_ids), temperature=temperature
    )

    pmi_log = logp_c - logp_0
    pmi_norm = (logp_c / max(Lc, 1)) - (logp_0 / max(L0, 1))
    ratio = torch.exp(torch.tensor(pmi_log)).item()
    return {
        "logp_c": logp_c,
        "logp_0": logp_0,
        "Lc": Lc,
        "L0": L0,
        "pmi_log": pmi_log,
        "pmi_norm": pmi_norm,
        "ratio": ratio,
    }
