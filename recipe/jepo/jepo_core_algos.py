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
"""
JEPO (Just Exploration with Policy Optimization) core algorithm implementation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from verl.utils.torch_functional import masked_mean


@dataclass
class JEPOConfig:
    """Configuration for JEPO algorithm"""
    delimiter: str = "\n\n"  # delimiter to split chain-of-thought from response
    format_penalty: float = 0.1  # penalty p for responses without delimiter
    beta_supp: float = 1.0  # coefficient for grad2 (suppression gradient)
    beta_kl: float = 0.1  # coefficient for KL divergence gradient
    buffer_size: int = 1000  # maximum buffer size for storing incorrect responses
    jepo_steps: int = 5  # number of JEPO steps when buffer is full


class JEPOBuffer:
    """Buffer to store responses where all answers are incorrect (reward = 0)"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, prompt: str, answer: str, responses: List[str], extra_data: Dict[str, Any] = None):
        """Add a batch of incorrect responses to buffer"""
        if len(self.buffer) >= self.max_size:
            # Remove oldest entry when buffer is full
            self.buffer.pop(0)
        
        entry = {
            'prompt': prompt,
            'answer': answer,
            'responses': responses
        }
        
        # Add any extra data (like batch data for training)
        if extra_data:
            entry.update(extra_data)
            
        self.buffer.append(entry)
    
    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size
    
    def clear(self):
        self.buffer.clear()
    
    def get_batch(self) -> List[Dict[str, Any]]:
        return self.buffer.copy()


def compute_jepo_advantages(
    log_probs: torch.Tensor,
    response_tokens: List[List[int]],
    tokenizer,
    delimiter: str,
    format_penalty: float,
    ground_truth_answer: str,
    model,
    question: str,
    device: torch.device,
    responses: List[str] = None,  # Optional - can work with just tokens
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute JEPO advantages based on the algorithm described in jepo.md
    For each single question with n responses
    
    Args:
        responses: List of n response strings
        log_probs: Log probabilities for each response [n, seq_len] 
        response_tokens: List of tokenized responses for each response
        tokenizer: Tokenizer to decode tokens and find delimiter positions
        delimiter: String used to split chain-of-thought from response
        format_penalty: Penalty p for responses without delimiter
        ground_truth_answer: The correct answer from dataset for computing π_θ(a*|x,c_j)
        model: The policy model for computing ground truth answer probabilities
        question: The original question x
        device: Device to place tensors on
        
    Returns:
        tilde_A_i: Clipped advantages for chain-of-thought
        tilde_A_i_ref: Normalized format advantages
        cot_log_probs_tensor: Chain-of-thought log probabilities [n]
        answer_log_probs_tensor: Ground truth answer log probabilities [n]
    """
    n = len(responses)
    
    # Step 1: Split responses and find token boundaries for CoT, delimiter, and answer
    chain_of_thoughts = []
    has_delimiter = []
    cot_log_probs = []  # Log probs for chain-of-thought tokens only
    answer_log_probs = []  # Log probs for answer tokens only
    
    # Vectorized preprocessing - find delimiter positions for all responses
    delimiter_tokens = tokenizer.encode(delimiter, add_special_tokens=False)
    
    # Handle case where responses are None (working with tokens only)
    if responses is None:
        responses = [None] * len(response_tokens)
    
    for i, (response, tokens) in enumerate(zip(responses, response_tokens)):
        # Search for delimiter token sequence directly in the response tokens (no string decoding needed)
        delimiter_start_pos = None
        
        # Search for delimiter token sequence in the response tokens
        for j in range(len(tokens) - len(delimiter_tokens) + 1):
            if tokens[j:j+len(delimiter_tokens)] == delimiter_tokens:
                delimiter_start_pos = j
                break
        
        if delimiter_start_pos is not None:
            # Found delimiter in tokens - extract chain-of-thought text only for storage
            cot_tokens = tokens[:delimiter_start_pos]
            cot_text = tokenizer.decode(cot_tokens, skip_special_tokens=True)
            
            # Extract log probs for chain-of-thought part (before delimiter)
            cot_log_prob = torch.sum(log_probs[i, :delimiter_start_pos])
            # Extract log probs for answer part (after delimiter)
            delimiter_end_pos = delimiter_start_pos + len(delimiter_tokens)
            answer_log_prob = torch.sum(log_probs[i, delimiter_end_pos:])
            has_delimiter.append(True)
            chain_of_thoughts.append(cot_text)
        else:
            # No delimiter found in tokens: use entire response as chain-of-thought
            cot_log_prob = torch.sum(log_probs[i, :])
            answer_log_prob = torch.tensor(0.0, device=log_probs.device)  # No answer part
            
            # Decode full response only when needed
            if response is not None:
                chain_of_thoughts.append(response)  # Use original response string
            else:
                # Decode tokens to get response text
                response_text = tokenizer.decode(tokens, skip_special_tokens=True)
                chain_of_thoughts.append(response_text)
            has_delimiter.append(False)
        
        cot_log_probs.append(cot_log_prob)
        answer_log_probs.append(answer_log_prob)
    
    # Convert to tensors for vectorized computation
    cot_log_probs_tensor = torch.stack(cot_log_probs)  # [n]
    answer_log_probs_tensor = torch.stack(answer_log_probs)  # [n]
    
    # Step 2: Compute π_θ(a*|x,c_j) for each chain-of-thought c_j
    # where a* is the ground truth answer
    answer_log_probs_gt = []  # Log probs for ground truth answer given each CoT
    
    for i, cot in enumerate(chain_of_thoughts):
        # Construct prompt: question + CoT + delimiter 
        if has_delimiter[i]:
            prompt_with_cot = question + cot + delimiter
        else:
            # If no delimiter in response, still try to use the "CoT" part
            prompt_with_cot = question + cot + delimiter
        
        # Tokenize the prompt + ground truth answer
        full_input = prompt_with_cot + ground_truth_answer
        input_ids = tokenizer.encode(full_input, return_tensors="pt").to(device)
        
        # Get log probabilities from model
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]
            log_probs_full = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
        
        # Find where the ground truth answer starts in the tokenized sequence
        prompt_tokens = tokenizer.encode(prompt_with_cot, add_special_tokens=False)
        answer_tokens = tokenizer.encode(ground_truth_answer, add_special_tokens=False)
        
        # Extract log probabilities for ground truth answer tokens
        answer_start_pos = len(prompt_tokens)
        answer_log_prob_sum = 0.0
        
        for j, token_id in enumerate(answer_tokens):
            if answer_start_pos + j < log_probs_full.shape[1]:
                # Get log prob for this specific token
                token_log_prob = log_probs_full[0, answer_start_pos + j - 1, token_id]  # -1 because logits are shifted
                answer_log_prob_sum += token_log_prob.item()
        
        answer_log_probs_gt.append(torch.tensor(answer_log_prob_sum, device=device))
    
    answer_log_probs_tensor = torch.stack(answer_log_probs_gt)  # [n]
    
    # Step 2b: Vectorized advantage computation using ground truth answer probabilities
    # A_i = log(1/n * sum_j π_θ(a*|x,c_j)) - v_i
    # where v_i = log(1/(n-1) * sum_{j!=i} π_θ(a*|x,c_j))
    
    log_mean_prob = torch.logsumexp(answer_log_probs_tensor, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32, device=device))
    
    # Calculate v_i = log(1/(n-1) * sum_{j!=i} π_θ(a*|x,c_j)) for all i
    if n > 1:
        # Create mask to exclude each i-th element
        indices = torch.arange(n, device=device)
        mask = indices.unsqueeze(1) != indices.unsqueeze(0)  # [n, n]
        
        # Broadcast answer_log_probs_tensor and apply mask
        masked_log_probs = answer_log_probs_tensor.unsqueeze(0).expand(n, -1)  # [n, n]
        masked_log_probs = torch.where(mask, masked_log_probs, torch.tensor(float('-inf'), device=device))
        
        # Compute logsumexp for each row (excluding i-th element)
        v_i = torch.logsumexp(masked_log_probs, dim=1) - torch.log(torch.tensor(n-1, dtype=torch.float32, device=device))  # [n]
    else:
        v_i = torch.tensor(float('-inf'), device=device).expand(n)
    
    # A_i = log(1/n * sum_j π_θ(a*|x,c_j)) - v_i (vectorized)
    A_tensor = log_mean_prob - v_i  # [n]
    
    # Step 3: Calculate tilde_A_i = clip(A_i / std(A), -1, 1) (vectorized)
    A_std = torch.std(A_tensor)
    if A_std > 1e-8:  # Avoid division by zero
        tilde_A_i = torch.clamp(A_tensor / A_std, -1.0, 1.0)
    else:
        tilde_A_i = torch.zeros_like(A_tensor)
    
    # Step 4: Vectorized format advantages computation
    has_delimiter_tensor = torch.tensor(has_delimiter, dtype=torch.bool, device=device)
    A_i_format_tensor = torch.where(has_delimiter_tensor, 0.0, -format_penalty)
    
    # Step 5: Vectorized format advantages normalization
    format_mean = torch.mean(A_i_format_tensor)
    format_std = torch.std(A_i_format_tensor)
    
    if format_std > 1e-8:
        tilde_A_i_ref = (A_i_format_tensor - format_mean) / format_std
    else:
        tilde_A_i_ref = torch.zeros_like(A_i_format_tensor)
    
    return (
        tilde_A_i.to(device), 
        tilde_A_i_ref.to(device), 
        cot_log_probs_tensor.to(device), 
        answer_log_probs_tensor.to(device)  # Now contains ground truth answer log probs
    )


def compute_jepo_advantages_batched(
    questions_responses: List[List[str]] = None,  # Optional - List of questions, each with multiple responses
    questions_log_probs: List[torch.Tensor],  # List of log prob tensors for each question
    questions_response_tokens: List[List[List[int]]],  # List of tokenized responses for each question
    questions: List[str],  # List of original questions
    ground_truth_answers: List[str],  # List of ground truth answers for each question
    tokenizer,
    delimiter: str,
    format_penalty: float,
    model,
    device: torch.device
) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Batched version of compute_jepo_advantages for multiple questions
    
    Args:
        questions_responses: List of [responses for question_i]
        questions_log_probs: List of [log_probs tensor for question_i] 
        questions_response_tokens: List of [tokenized responses for question_i]
        questions: List of original questions x_i
        ground_truth_answers: List of ground truth answers a*_i for each question
        tokenizer: Tokenizer
        delimiter: String delimiter
        format_penalty: Penalty for missing delimiter
        model: Policy model for computing ground truth probabilities
        device: Device
        
    Returns:
        Lists of tensors for each question:
        - tilde_A_i_list: Clipped advantages for each question
        - tilde_A_i_ref_list: Format advantages for each question  
        - cot_log_probs_list: CoT log probs for each question
        - answer_log_probs_list: Ground truth answer log probs for each question
    """
    # Handle case where questions_responses is None
    if questions_responses is None:
        num_questions = len(questions_response_tokens)
    else:
        num_questions = len(questions_responses)
    
    tilde_A_i_list = []
    tilde_A_i_ref_list = []
    cot_log_probs_list = []
    answer_log_probs_list = []
    
    # Process all questions in batch
    for q_idx in range(num_questions):
        responses = questions_responses[q_idx] if questions_responses is not None else None
        log_probs = questions_log_probs[q_idx] 
        response_tokens = questions_response_tokens[q_idx]
        
        # Compute advantages for this question using ground truth answer
        tilde_A_i, tilde_A_i_ref, cot_log_probs, answer_log_probs = compute_jepo_advantages(
            responses=responses,
            log_probs=log_probs,
            response_tokens=response_tokens,
            tokenizer=tokenizer,
            delimiter=delimiter,
            format_penalty=format_penalty,
            ground_truth_answer=ground_truth_answers[q_idx],
            model=model,
            question=questions[q_idx],
            device=device
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
    questions_ref_log_probs: List[torch.Tensor],
    questions_current_log_probs: List[torch.Tensor],
    beta_supp: float,
    beta_kl: float
) -> Dict[str, torch.Tensor]:
    """
    Batched version of jepo_loss for multiple questions using vectorized operations
    
    Args:
        questions_*: Lists of tensors for each question
        beta_supp: Suppression coefficient
        beta_kl: KL coefficient
        
    Returns:
        Dictionary with aggregated loss components
    """
    if not questions_cot_log_probs:
        return {
            "total_loss": torch.tensor(0.0),
            "pg_loss": torch.tensor(0.0),
            "supp_loss": torch.tensor(0.0),
            "kl_loss": torch.tensor(0.0),
            "advantages_mean": torch.tensor(0.0),
            "advantages_std": torch.tensor(0.0),
            "tilde_A_i_mean": torch.tensor(0.0),
            "tilde_A_i_ref_mean": torch.tensor(0.0)
        }
    
    # Stack all tensors for vectorized computation
    all_cot_log_probs = torch.cat(questions_cot_log_probs, dim=0)  # [total_samples, cot_seq_len]
    all_tilde_A_i = torch.cat(questions_tilde_A_i, dim=0)  # [total_samples]
    all_tilde_A_i_ref = torch.cat(questions_tilde_A_i_ref, dim=0)  # [total_samples]
    all_ref_log_probs = torch.cat(questions_ref_log_probs, dim=0)  # [total_samples, seq_len]
    all_current_log_probs = torch.cat(questions_current_log_probs, dim=0)  # [total_samples, seq_len]
    
    # Vectorized computation of combined advantages
    combined_advantages = all_tilde_A_i + all_tilde_A_i_ref
    combined_advantages = combined_advantages.detach()  # Detach to avoid backprop through advantages
    
    # Vectorized policy gradient loss (grad1 component)
    # Handle different sequence lengths by using masked_mean if needed
    if all_cot_log_probs.dim() == 1:
        # If already summed per response
        pg_loss = -torch.mean(combined_advantages * all_cot_log_probs)
    else:
        # If per-token log probs, sum across sequence dimension first
        cot_log_probs_summed = torch.sum(all_cot_log_probs, dim=-1)  # [total_samples]
        pg_loss = -torch.mean(combined_advantages * cot_log_probs_summed)
    
    # Vectorized suppression loss (grad2 component) 
    # Compute per-question suppression terms
    supp_losses = []
    for answer_log_probs in questions_answer_log_probs:
        if answer_log_probs.dim() == 1:
            # Already summed per response
            mean_answer_log_prob = torch.logsumexp(answer_log_probs, dim=0) - torch.log(torch.tensor(len(answer_log_probs), dtype=torch.float32))
        else:
            # Sum across sequence dimension first, then compute mean
            answer_log_probs_summed = torch.sum(answer_log_probs, dim=-1)
            mean_answer_log_prob = torch.logsumexp(answer_log_probs_summed, dim=0) - torch.log(torch.tensor(len(answer_log_probs_summed), dtype=torch.float32))
        supp_losses.append(mean_answer_log_prob)
    
    supp_loss = -beta_supp * torch.mean(torch.stack(supp_losses))
    
    # Vectorized KL divergence loss (grad3 component) using PPO-style computation
    from verl.trainer.ppo import core_algos
    kl_div = core_algos.kl_penalty(all_current_log_probs, all_ref_log_probs, kl_penalty="kl")
    kl_loss = beta_kl * torch.mean(kl_div)
    
    # Total loss
    total_loss = pg_loss + supp_loss + kl_loss
    
    return {
        "total_loss": total_loss,
        "pg_loss": pg_loss,
        "supp_loss": supp_loss,
        "kl_loss": kl_loss,
        "advantages_mean": torch.mean(combined_advantages),
        "advantages_std": torch.std(combined_advantages),
        "tilde_A_i_mean": torch.mean(all_tilde_A_i),
        "tilde_A_i_ref_mean": torch.mean(all_tilde_A_i_ref)
    }


def compute_jepo_gradients(
    chain_of_thought_log_probs: torch.Tensor,
    answer_log_probs: torch.Tensor,
    tilde_A_i: torch.Tensor,
    tilde_A_i_ref: torch.Tensor,
    ref_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    beta_supp: float,
    beta_kl: float
) -> Dict[str, torch.Tensor]:
    """
    Compute JEPO gradients using vectorized operations like PPO
    
    Args:
        chain_of_thought_log_probs: Log probs for chain-of-thought tokens [n, cot_seq_len]
        answer_log_probs: Log probs for answer tokens [n, ans_seq_len]  
        tilde_A_i: Clipped advantages [n]
        tilde_A_i_ref: Normalized format advantages [n]
        ref_log_probs: Reference model log probabilities [n, seq_len]
        current_log_probs: Current model log probabilities [n, seq_len]
        beta_supp: Coefficient for suppression gradient
        beta_kl: Coefficient for KL divergence gradient
        
    Returns:
        Dictionary containing gradient components
    """
    n = tilde_A_i.shape[0]
    
    # grad1: 1/n * sum_i ((tilde_A_i + tilde_A_i_ref) * grad_theta log pi_theta(c_i|x))
    combined_advantages = tilde_A_i + tilde_A_i_ref
    
    # Vectorized computation for chain-of-thought gradients
    # Broadcast advantages to match log_probs shape and compute weighted gradients
    if chain_of_thought_log_probs.dim() == 2:
        # [n, seq_len] case
        expanded_advantages = combined_advantages.unsqueeze(-1)  # [n, 1]
        weighted_cot_grads = expanded_advantages * chain_of_thought_log_probs  # [n, seq_len]
        grad1 = torch.mean(weighted_cot_grads, dim=0)  # [seq_len]
    else:
        # [n] case (already summed)
        grad1 = torch.mean(combined_advantages * chain_of_thought_log_probs)
    
    # grad2: grad_theta log(1/n * sum_i pi_theta(a|x,c_i))
    # This is the gradient of the log of the mean probability
    if answer_log_probs.dim() == 1:
        # Already summed per response
        mean_answer_log_prob = torch.logsumexp(answer_log_probs, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32))
    else:
        # Sum across sequence dimension first
        answer_log_probs_summed = torch.sum(answer_log_probs, dim=-1)
        mean_answer_log_prob = torch.logsumexp(answer_log_probs_summed, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32))
    
    grad2 = mean_answer_log_prob
    
    # grad3: KL divergence gradient using PPO-style computation
    from verl.trainer.ppo import core_algos
    kl_div = core_algos.kl_penalty(current_log_probs, ref_log_probs, kl_penalty="kl")
    grad3 = torch.mean(kl_div, dim=0 if kl_div.dim() > 1 else None)
    
    # Combine gradients: grad1 + beta_supp * grad2 - beta_kl * grad3
    total_gradient = grad1 + beta_supp * grad2 - beta_kl * grad3
    
    return {
        'grad1': grad1,
        'grad2': grad2, 
        'grad3': grad3,
        'total_gradient': total_gradient,
        'combined_advantages': combined_advantages
    }


def jepo_loss(
    chain_of_thought_log_probs: torch.Tensor,
    answer_log_probs: torch.Tensor,
    tilde_A_i: torch.Tensor,
    tilde_A_i_ref: torch.Tensor,
    ref_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    beta_supp: float,
    beta_kl: float
) -> Dict[str, torch.Tensor]:
    """
    Compute JEPO loss components using PPO-style vectorized operations
    
    Returns:
        Dictionary with loss components and metrics
    """
    n = tilde_A_i.shape[0]
    
    # Policy gradient loss for chain-of-thought (grad1 component)
    combined_advantages = tilde_A_i + tilde_A_i_ref
    combined_advantages = combined_advantages.detach()  # Detach to avoid backprop through advantages
    
    # Handle different tensor shapes for log probs
    if chain_of_thought_log_probs.dim() == 1:
        # Already summed per response
        pg_loss = -torch.mean(combined_advantages * chain_of_thought_log_probs)
    else:
        # Per-token log probs, sum across sequence dimension first
        cot_log_probs_summed = torch.sum(chain_of_thought_log_probs, dim=-1)
        pg_loss = -torch.mean(combined_advantages * cot_log_probs_summed)
    
    # Suppression loss (grad2 component)
    if answer_log_probs.dim() == 1:
        # Already summed per response
        mean_answer_log_prob = torch.logsumexp(answer_log_probs, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32))
    else:
        # Sum across sequence dimension first
        answer_log_probs_summed = torch.sum(answer_log_probs, dim=-1)
        mean_answer_log_prob = torch.logsumexp(answer_log_probs_summed, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32))
    
    supp_loss = -beta_supp * mean_answer_log_prob
    
    # KL divergence loss (grad3 component) using PPO-style computation
    from verl.trainer.ppo import core_algos
    kl_div = core_algos.kl_penalty(current_log_probs, ref_log_probs, kl_penalty="kl")
    kl_loss = beta_kl * torch.mean(kl_div)
    
    # Total loss
    total_loss = pg_loss + supp_loss + kl_loss
    
    return {
        'total_loss': total_loss,
        'pg_loss': pg_loss,
        'supp_loss': supp_loss,
        'kl_loss': kl_loss,
        'advantages_mean': torch.mean(combined_advantages),
        'advantages_std': torch.std(combined_advantages),
        'tilde_A_i_mean': torch.mean(tilde_A_i),
        'tilde_A_i_ref_mean': torch.mean(tilde_A_i_ref)
    }