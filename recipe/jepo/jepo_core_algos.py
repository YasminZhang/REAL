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


@dataclass
class JEPOConfig:
    delimiter: str = "\n\n"
    format_penalty: float = 0.1
    beta_supp: float = 1.0
    beta_kl: float = 0.1
    buffer_size: int = 1000
    jepo_steps: int = 5


def compute_single_jepo_advantages(
    response_tokens: List[List[int]],
    prompt_tokens: List[List[int]],
    ground_truth_answer_tokens: List[List[int]],
    delimiter_tokens: List[int],
    format_penalty: float,
    model,
    device: torch.device,
    pad_token: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # return shape [n],[n],[n],[n]
    # compute jepo adv for a single question
    
    n = len(response_tokens)
    
    # Parse CoT and delimiter positions
    has_delimiter = []
    cot_tokens_list = []
    delimiter_positions = []

    for i, tokens in enumerate(response_tokens):
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
            len_cot = len(tokens) - len(ground_truth_answer_tokens[i]) - len(delimiter_tokens)
            cot_tokens = tokens[:len_cot]
            has_delimiter.append(False)
            delimiter_positions.append(len_cot)
        
        cot_tokens_list.append(cot_tokens)
    
    # Prepare batch input: prompt + cot + delimiter + ground_truth for all responses
    batch_input_tokens = []
    cot_start_positions = []
    answer_start_positions = []
    for i,cot_tokens in enumerate(cot_tokens_list):
        # Convert all to tensors if they aren't already
        prompt_tokens_tensor = torch.tensor(prompt_tokens[i], device=device) if not isinstance(prompt_tokens[i], torch.Tensor) else prompt_tokens[i]
        cot_tokens_tensor = torch.tensor(cot_tokens, device=device) if not isinstance(cot_tokens, torch.Tensor) else cot_tokens
        delimiter_tokens_tensor = torch.tensor(delimiter_tokens, device=device) if not isinstance(delimiter_tokens, torch.Tensor) else delimiter_tokens
        ground_truth_tokens_tensor = torch.tensor(ground_truth_answer_tokens[i], device=device) if not isinstance(ground_truth_answer_tokens[i], torch.Tensor) else ground_truth_answer_tokens[i]

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
    # Prepare data for DataProto creation
    batch_input_ids = torch.stack(padded_tokens).to(dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
    
    # Create position_ids (assuming standard sequential positions)
    position_ids = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0).repeat(n, 1)
    # Mask out positions for padded tokens
    for i, tokens in enumerate(batch_input_tokens):
        position_ids[i, len(tokens):] = 0
    
    # Return data needed for DataProto instead of doing model forward here
    return {
        'batch_input_ids': batch_input_ids,
        'attention_mask': attention_mask, 
        'position_ids': position_ids,
        'cot_start_positions': cot_start_positions,
        'answer_start_positions': answer_start_positions,
        'cot_tokens_list': cot_tokens_list,
        'ground_truth_answer_tokens': ground_truth_answer_tokens,
        'has_delimiter': has_delimiter,
        'max_len': max_len
    }

def compute_jepo_advantages_from_logprobs(
    log_probs_batch: torch.Tensor,
    data_dict: dict,
    format_penalty: float,
    has_delimiter: list,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute JEPO advantages from log probabilities tensor"""
    
    n = log_probs_batch.shape[0]
    cot_start_positions = data_dict['cot_start_positions']
    answer_start_positions = data_dict['answer_start_positions']
    cot_tokens_list = data_dict['cot_tokens_list']
    ground_truth_answer_tokens = data_dict['ground_truth_answer_tokens']
    
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
        for j, token_id in enumerate(ground_truth_answer_tokens[i]):
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
    jepo_advantage = tilde_A_i + tilde_A_i_ref
    log_mean_answer_prob = log_mean_prob.repeat(n)

    return (jepo_advantage.to(device), cot_log_probs_tensor, answer_log_probs_tensor, log_mean_answer_prob)


def compute_jepo_advantages(
    response_tokens,
    prompt_tokens,
    ground_truth_answer_tokens,
    delimiter_tokens,
    format_penalty,
    model,
    device,
    pad_token,
    index
):

    # response_tokens has shape [bsz, max_response_length]
    # prompt_tokens has shape [bsz, max_prompt_length]
    # ground_truth_answer_tokens is a list with shape bsz, np.ndarray
    # delimiter_tokens is a lift of tokens
    # index is a list of uid

    uuid = np.unique(index)

    for uid in uuid:
        uid_mask = (index == uid)
        response_tokens_uid = response_tokens[uid_mask]
        prompt_tokens_uid = prompt_tokens[uid_mask]
        ground_truth_answer_tokens_uid = ground_truth_answer_tokens[uid_mask]
        
        if len(response_tokens_uid) == 0:
            continue
        
        data_dict = compute_single_jepo_advantages(
            response_tokens=response_tokens_uid,
            prompt_tokens=prompt_tokens_uid,
            ground_truth_answer_tokens=ground_truth_answer_tokens_uid,
            delimiter_tokens=delimiter_tokens,
            format_penalty=format_penalty,
            model=model,
            device=device,
            pad_token=pad_token
        )
        
        if 'data_dicts' not in locals():
            data_dicts = [data_dict]
        else:
            data_dicts.append(data_dict)
    
    return data_dicts