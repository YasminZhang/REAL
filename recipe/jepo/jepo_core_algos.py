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


@dataclass
class JEPOConfig:
    delimiter: str = "\n\n"
    format_penalty: float = 0.1
    beta_supp: float = 1.0
    beta_kl: float = 0.1
    buffer_size: int = 1000
    jepo_steps: int = 5


def compute_jepo_advantages(
    response_tokens: List[List[int]],
    prompt_tokens: List[int],
    ground_truth_answer_tokens: List[int],
    delimiter_tokens: List[int],
    format_penalty: float,
    model,
    device: torch.device,
    pad_token: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # compute jepo adv for a single question
    
    n = len(response_tokens)
    
    # Parse CoT and delimiter positions
    has_delimiter = []
    cot_tokens_list = []
    delimiter_positions = []
    
    for tokens in response_tokens:
        tokens = tokens.detach().clone()
        delimiter_start_pos = None
        
        for j in range(len(tokens) - len(delimiter_tokens) + 1):
            if tokens[j:j+len(delimiter_tokens)] == delimiter_tokens:
                delimiter_start_pos = j
                break
        
        if delimiter_start_pos is not None:
            cot_tokens = tokens[:delimiter_start_pos]
            has_delimiter.append(True)
            delimiter_positions.append(delimiter_start_pos)
        else:
            cot_tokens = tokens
            has_delimiter.append(False)
            delimiter_positions.append(len(tokens))
        
        cot_tokens_list.append(cot_tokens)
    
    # Prepare batch input: prompt + cot + delimiter + ground_truth for all responses
    batch_input_tokens = []
    cot_start_positions = []
    answer_start_positions = []
    for cot_tokens in cot_tokens_list:
        # Convert all to tensors if they aren't already
        prompt_tokens_tensor = torch.tensor(prompt_tokens, device=device) if not isinstance(prompt_tokens, torch.Tensor) else prompt_tokens
        cot_tokens_tensor = torch.tensor(cot_tokens, device=device) if not isinstance(cot_tokens, torch.Tensor) else cot_tokens
        delimiter_tokens_tensor = torch.tensor(delimiter_tokens, device=device) if not isinstance(delimiter_tokens, torch.Tensor) else delimiter_tokens
        ground_truth_tokens_tensor = torch.tensor(ground_truth_answer_tokens, device=device) if not isinstance(ground_truth_answer_tokens, torch.Tensor) else ground_truth_answer_tokens

        prompt_with_cot_tokens = torch.cat([prompt_tokens_tensor, cot_tokens_tensor, delimiter_tokens_tensor])
        full_input_tokens = torch.cat([prompt_with_cot_tokens, ground_truth_tokens_tensor])
        
        batch_input_tokens.append(full_input_tokens)
        cot_start_positions.append(len(prompt_tokens_tensor))
        answer_start_positions.append(len(prompt_with_cot_tokens))
    
    # Pad sequences to same length for batching
    max_len = max(len(tokens) for tokens in batch_input_tokens)
    padded_tokens = []
    attention_masks = []
    
    for tokens in batch_input_tokens:
        pad_length = max_len - len(tokens)
        padding = torch.full((pad_length,), pad_token, dtype=tokens.dtype, device=tokens.device)
        padded = torch.cat([tokens, padding])
        mask = [1] * len(tokens) + [0] * pad_length
        padded_tokens.append(padded)
        attention_masks.append(mask)
    
    # Single batched forward pass
    batch_input_ids = torch.stack(padded_tokens).to(dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
    
    outputs = model(batch_input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs_batch = torch.log_softmax(logits, dim=-1)
    
    # Extract log probabilities for CoT and answers using positions
    cot_log_probs = []
    answer_log_probs = []
    
    for i in range(n):
        cot_tokens = cot_tokens_list[i]
        cot_start = cot_start_positions[i]
        answer_start = answer_start_positions[i]
        # CoT log probabilities - keep gradients
        cot_log_prob_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        for j, token_id in enumerate(cot_tokens):
            pos = cot_start + j
            if pos > 0 and pos < log_probs_batch.shape[1]:
                token_log_prob = log_probs_batch[i, pos - 1, token_id]
                cot_log_prob_sum += token_log_prob
        # Answer log probabilities - detach for advantage calculation
        answer_log_prob_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        for j, token_id in enumerate(ground_truth_answer_tokens):
            pos = answer_start + j
            if pos > 0 and pos < log_probs_batch.shape[1]:
                token_log_prob = log_probs_batch[i, pos - 1, token_id].detach()
                answer_log_prob_sum += token_log_prob
        
        cot_log_probs.append(cot_log_prob_sum)
        answer_log_probs.append(answer_log_prob_sum)
    
    cot_log_probs_tensor = torch.stack(cot_log_probs)
    answer_log_probs_tensor = torch.stack(answer_log_probs)
    
    # Compute advantages (detached)
    log_mean_prob = torch.logsumexp(answer_log_probs_tensor, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32, device=device))
    
    if n > 1:
        indices = torch.arange(n, device=device)
        mask = indices.unsqueeze(1) != indices.unsqueeze(0)
        
        masked_log_probs = answer_log_probs_tensor.unsqueeze(0).expand(n, -1)
        masked_log_probs = torch.where(mask, masked_log_probs, torch.tensor(float('-inf'), device=device))
        
        v_i = torch.logsumexp(masked_log_probs, dim=1) - torch.log(torch.tensor(n-1, dtype=torch.float32, device=device))
    else:
        v_i = torch.tensor(float('-inf'), device=device).expand(n)
    
    A_tensor = log_mean_prob - v_i
    
    A_std = torch.std(A_tensor)
    if A_std > 1e-8:
        tilde_A_i = torch.clamp(A_tensor / A_std, -1.0, 1.0)
    else:
        tilde_A_i = torch.zeros_like(A_tensor)
    
    # Format penalty
    has_delimiter_tensor = torch.tensor(has_delimiter, dtype=torch.bool, device=device)
    A_i_format_tensor = torch.where(has_delimiter_tensor, 0.0, -format_penalty)
    
    format_mean = torch.mean(A_i_format_tensor)
    format_std = torch.std(A_i_format_tensor)
    
    if format_std > 1e-8:
        tilde_A_i_ref = (A_i_format_tensor - format_mean) / format_std
    else:
        tilde_A_i_ref = torch.zeros_like(A_i_format_tensor)
    
    # Detach advantages but keep gradients for CoT log probs
    tilde_A_i = tilde_A_i.detach()
    tilde_A_i_ref = tilde_A_i_ref.detach()
    
    return (
        tilde_A_i.to(device), 
        tilde_A_i_ref.to(device), 
        cot_log_probs_tensor.to(device), 
        answer_log_probs_tensor.to(device)
    )



def compute_jepo_advantages_batched(
    questions_response_tokens: List[List[List[int]]],
    questions_prompts: List[str],
    ground_truth_answers: List[str],
    tokenizer,
    delimiter: str,
    format_penalty: float,
    model,
    device: torch.device,
) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    
    # note that questions_response_tokens contain pad tokens

    delimiter_tokens = tokenizer.encode(delimiter, add_special_tokens=False)
    tilde_A_i_list = []
    tilde_A_i_ref_list = []
    cot_log_probs_list = []
    answer_log_probs_list = []
    
    for q_idx in range(len(questions_response_tokens)):
        response_tokens = questions_response_tokens[q_idx]
        prompt_tokens = questions_prompts[q_idx]
        ground_truth = ground_truth_answers[q_idx]
        
        ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)
        pad_token = tokenizer.pad_token_id # get the pad token
        
        tilde_A_i, tilde_A_i_ref, cot_log_probs, answer_log_probs = compute_jepo_advantages(
            response_tokens=response_tokens,
            prompt_tokens=prompt_tokens,
            ground_truth_answer_tokens=ground_truth_tokens,
            delimiter_tokens=delimiter_tokens,
            format_penalty=format_penalty,
            model=model,
            device=device,
            pad_token=pad_token
        )
        
        tilde_A_i_list.append(tilde_A_i)
        tilde_A_i_ref_list.append(tilde_A_i_ref)
        cot_log_probs_list.append(cot_log_probs)
        answer_log_probs_list.append(answer_log_probs)
    
    return tilde_A_i_list, tilde_A_i_ref_list, cot_log_probs_list, answer_log_probs_list


def jepo_loss_batched(
    questions_cot_log_probs: List[torch.Tensor],
    questions_answer_log_probs: List[torch.Tensor], 
    questions_tilde_A_i: List[torch.Tensor],
    questions_tilde_A_i_ref: List[torch.Tensor],
    beta_supp: float,
    beta_kl: float = 0.0
) -> Dict[str, torch.Tensor]:
    
    if not questions_cot_log_probs:
        return {
            "total_loss": torch.tensor(0.0, requires_grad=True),
            "pg_loss": torch.tensor(0.0),
            "supp_loss": torch.tensor(0.0),
            "kl_loss": torch.tensor(0.0),
        }
    
    all_cot_log_probs = torch.cat(questions_cot_log_probs, dim=0)
    all_tilde_A_i = torch.cat(questions_tilde_A_i, dim=0)
    all_tilde_A_i_ref = torch.cat(questions_tilde_A_i_ref, dim=0)
    
    combined_advantages = all_tilde_A_i + all_tilde_A_i_ref
    combined_advantages = combined_advantages.detach()
    
    pg_loss = -torch.mean(combined_advantages * all_cot_log_probs)
    
    supp_losses = []
    for answer_log_probs in questions_answer_log_probs:
        mean_answer_log_prob = torch.logsumexp(answer_log_probs, dim=0) - torch.log(torch.tensor(len(answer_log_probs), dtype=torch.float32))
        supp_losses.append(mean_answer_log_prob)
    
    supp_loss = -beta_supp * torch.mean(torch.stack(supp_losses))
    
    kl_loss = torch.tensor(0.0, requires_grad=True)
    
    total_loss = pg_loss + supp_loss + kl_loss
    
    return {
        "total_loss": total_loss,
        "pg_loss": pg_loss,
        "supp_loss": supp_loss,
        "kl_loss": kl_loss,
    }


def jepo_loss(
    chain_of_thought_log_probs: torch.Tensor,
    answer_log_probs: torch.Tensor,
    tilde_A_i: torch.Tensor,
    tilde_A_i_ref: torch.Tensor,
    beta_supp: float,
    beta_kl: float = 0.0
) -> Dict[str, torch.Tensor]:
    
    n = tilde_A_i.shape[0]
    
    combined_advantages = tilde_A_i + tilde_A_i_ref
    combined_advantages = combined_advantages.detach()
    
    pg_loss = -torch.mean(combined_advantages * chain_of_thought_log_probs)
    
    mean_answer_log_prob = torch.logsumexp(answer_log_probs, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32))
    supp_loss = -beta_supp * mean_answer_log_prob
    
    kl_loss = torch.tensor(0.0, requires_grad=True)
    
    total_loss = pg_loss + supp_loss + kl_loss
    
    return {
        'total_loss': total_loss,
        'pg_loss': pg_loss,
        'supp_loss': supp_loss,
        'kl_loss': kl_loss,
    }