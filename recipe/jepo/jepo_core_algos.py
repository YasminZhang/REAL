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
    responses: List[str],
    log_probs: torch.Tensor,
    response_tokens: List[List[int]],
    tokenizer,
    delimiter: str,
    format_penalty: float,
    pi_theta: torch.Tensor,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute JEPO advantages based on the algorithm described in jepo.md
    
    Args:
        responses: List of n response strings
        log_probs: Log probabilities for each response [n, seq_len]
        response_tokens: List of tokenized responses for each response
        tokenizer: Tokenizer to decode tokens and find delimiter positions
        delimiter: String used to split chain-of-thought from response
        format_penalty: Penalty p for responses without delimiter
        pi_theta: Policy probabilities [n, vocab_size]
        device: Device to place tensors on
        
    Returns:
        tilde_A_i: Clipped advantages for chain-of-thought
        tilde_A_i_ref: Normalized format advantages
        cot_log_probs_tensor: Chain-of-thought log probabilities [n]
        answer_log_probs_tensor: Answer log probabilities [n]
    """
    n = len(responses)
    
    # Step 1: Split responses and find token boundaries for CoT, delimiter, and answer
    chain_of_thoughts = []
    has_delimiter = []
    cot_log_probs = []  # Log probs for chain-of-thought tokens only
    answer_log_probs = []  # Log probs for answer tokens only
    
    for i, (response, tokens) in enumerate(zip(responses, response_tokens)):
        if delimiter in response:
            # Find delimiter position in the response string
            cot_text = response.split(delimiter)[0]
            
            # Find delimiter token positions
            delimiter_tokens = tokenizer.encode(delimiter, add_special_tokens=False)
            delimiter_start_pos = None
            
            # Search for delimiter token sequence in the response tokens
            for j in range(len(tokens) - len(delimiter_tokens) + 1):
                if tokens[j:j+len(delimiter_tokens)] == delimiter_tokens:
                    delimiter_start_pos = j
                    break
            
            if delimiter_start_pos is not None:
                # Extract log probs for chain-of-thought part (before delimiter)
                cot_log_prob = torch.sum(log_probs[i, :delimiter_start_pos])
                # Extract log probs for answer part (after delimiter)
                delimiter_end_pos = delimiter_start_pos + len(delimiter_tokens)
                answer_log_prob = torch.sum(log_probs[i, delimiter_end_pos:])
                has_delimiter.append(True)
            else:
                # Fallback: use entire sequence if delimiter not found in tokens
                cot_log_prob = torch.sum(log_probs[i, :])
                answer_log_prob = torch.tensor(0.0, device=log_probs.device)  # No answer part
                has_delimiter.append(False)
            
            chain_of_thoughts.append(cot_text)
        else:
            # No delimiter: use entire response as chain-of-thought
            cot_log_prob = torch.sum(log_probs[i, :])
            answer_log_prob = torch.tensor(0.0, device=log_probs.device)  # No answer part
            chain_of_thoughts.append(response)
            has_delimiter.append(False)
        
        cot_log_probs.append(cot_log_prob)
        answer_log_probs.append(answer_log_prob)
    
    # Step 2: Calculate A_i for each response using chain-of-thought log probs
    A_values = []
    
    # Convert to tensors for easier computation
    cot_log_probs_tensor = torch.stack(cot_log_probs)  # [n]
    answer_log_probs_tensor = torch.stack(answer_log_probs)  # [n]
    
    for i in range(n):
        # Calculate log(1/n * sum_j pi_theta(a|x,c_j)) using chain-of-thought log probs
        log_mean_prob = torch.logsumexp(cot_log_probs_tensor, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32))
        
        # Calculate v_i = log(1/(n-1) * sum_{j!=i} pi_theta(a|x,c_j))
        other_cot_log_probs = torch.cat([cot_log_probs_tensor[:i], cot_log_probs_tensor[i+1:]], dim=0)
        if len(other_cot_log_probs) > 0:
            v_i = torch.logsumexp(other_cot_log_probs, dim=0) - torch.log(torch.tensor(n-1, dtype=torch.float32))
        else:
            v_i = torch.tensor(float('-inf'))
        
        # A_i = log(1/n * sum_j pi_theta(a|x,c_j)) - v_i
        A_i = log_mean_prob - v_i
        A_values.append(A_i)
    
    A_tensor = torch.stack(A_values)
    
    # Step 3: Calculate tilde_A_i = clip(A_i / std(A), -1, 1)
    A_std = torch.std(A_tensor)
    if A_std > 1e-8:  # Avoid division by zero
        tilde_A_i = torch.clamp(A_tensor / A_std, -1.0, 1.0)
    else:
        tilde_A_i = torch.zeros_like(A_tensor)
    
    # Step 4: Calculate format advantages
    A_i_format = []
    for has_delim in has_delimiter:
        if has_delim:
            A_i_format.append(0.0)
        else:
            A_i_format.append(-format_penalty)
    
    A_i_format_tensor = torch.tensor(A_i_format, device=device, dtype=torch.float32)
    
    # Step 5: Normalize format advantages
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
        answer_log_probs_tensor.to(device)
    )


def compute_jepo_advantages_batched(
    questions_responses: List[List[str]],  # List of questions, each with multiple responses
    questions_log_probs: List[torch.Tensor],  # List of log prob tensors for each question
    questions_response_tokens: List[List[List[int]]],  # List of tokenized responses for each question
    tokenizer,
    delimiter: str,
    format_penalty: float,
    device: torch.device
) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Batched version of compute_jepo_advantages for multiple questions
    
    Args:
        questions_responses: List of [responses for question_i]
        questions_log_probs: List of [log_probs tensor for question_i] 
        questions_response_tokens: List of [tokenized responses for question_i]
        tokenizer: Tokenizer
        delimiter: String delimiter
        format_penalty: Penalty for missing delimiter
        device: Device
        
    Returns:
        Lists of tensors for each question:
        - tilde_A_i_list: Clipped advantages for each question
        - tilde_A_i_ref_list: Format advantages for each question  
        - cot_log_probs_list: CoT log probs for each question
        - answer_log_probs_list: Answer log probs for each question
    """
    num_questions = len(questions_responses)
    
    tilde_A_i_list = []
    tilde_A_i_ref_list = []
    cot_log_probs_list = []
    answer_log_probs_list = []
    
    # Process all questions in batch
    for q_idx in range(num_questions):
        responses = questions_responses[q_idx]
        log_probs = questions_log_probs[q_idx] 
        response_tokens = questions_response_tokens[q_idx]
        
        # Dummy pi_theta for this question (not used in current implementation)
        pi_theta = torch.exp(log_probs)
        
        # Compute advantages for this question
        tilde_A_i, tilde_A_i_ref, cot_log_probs, answer_log_probs = compute_jepo_advantages(
            responses=responses,
            log_probs=log_probs,
            response_tokens=response_tokens,
            tokenizer=tokenizer,
            delimiter=delimiter,
            format_penalty=format_penalty,
            pi_theta=pi_theta,
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
    Batched version of jepo_loss for multiple questions
    
    Args:
        questions_*: Lists of tensors for each question
        beta_supp: Suppression coefficient
        beta_kl: KL coefficient
        
    Returns:
        Dictionary with aggregated loss components
    """
    num_questions = len(questions_cot_log_probs)
    
    total_grad1_loss = 0.0
    total_grad2_loss = 0.0  
    total_grad3_loss = 0.0
    total_samples = 0
    
    for q_idx in range(num_questions):
        # Compute loss for this question
        question_loss = jepo_loss(
            chain_of_thought_log_probs=questions_cot_log_probs[q_idx],
            answer_log_probs=questions_answer_log_probs[q_idx],
            tilde_A_i=questions_tilde_A_i[q_idx],
            tilde_A_i_ref=questions_tilde_A_i_ref[q_idx],
            ref_log_probs=questions_ref_log_probs[q_idx],
            current_log_probs=questions_current_log_probs[q_idx],
            beta_supp=beta_supp,
            beta_kl=beta_kl
        )
        
        # Weight by number of responses in this question
        num_responses = len(questions_tilde_A_i[q_idx])
        total_grad1_loss += question_loss["grad1_loss"] * num_responses
        total_grad2_loss += question_loss["grad2_loss"] * num_responses
        total_grad3_loss += question_loss["grad3_loss"] * num_responses
        total_samples += num_responses
    
    # Average across all samples
    if total_samples > 0:
        avg_grad1_loss = total_grad1_loss / total_samples
        avg_grad2_loss = total_grad2_loss / total_samples  
        avg_grad3_loss = total_grad3_loss / total_samples
        total_loss = avg_grad1_loss + beta_supp * avg_grad2_loss - beta_kl * avg_grad3_loss
    else:
        avg_grad1_loss = torch.tensor(0.0)
        avg_grad2_loss = torch.tensor(0.0)
        avg_grad3_loss = torch.tensor(0.0)
        total_loss = torch.tensor(0.0)
    
    return {
        "grad1_loss": avg_grad1_loss,
        "grad2_loss": avg_grad2_loss, 
        "grad3_loss": avg_grad3_loss,
        "total_loss": total_loss
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
    Compute JEPO gradients according to the algorithm
    
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
    
    # Compute gradient for chain-of-thought terms
    grad1_terms = []
    for i in range(n):
        cot_grad = combined_advantages[i] * chain_of_thought_log_probs[i]
        grad1_terms.append(cot_grad)
    
    grad1 = torch.mean(torch.stack(grad1_terms), dim=0)
    
    # grad2: grad_theta log(1/n * sum_i pi_theta(a|x,c_i))
    # This is the gradient of the log of the mean probability
    mean_answer_log_prob = torch.logsumexp(answer_log_probs, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32))
    grad2 = mean_answer_log_prob
    
    # grad3: KL divergence gradient
    kl_div = current_log_probs - ref_log_probs
    grad3 = torch.mean(kl_div, dim=0)
    
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
    Compute JEPO loss components
    
    Returns:
        Dictionary with loss components and metrics
    """
    n = tilde_A_i.shape[0]
    
    # Policy gradient loss for chain-of-thought (grad1 component)
    combined_advantages = tilde_A_i + tilde_A_i_ref
    pg_loss = -torch.mean(combined_advantages.unsqueeze(-1) * chain_of_thought_log_probs)
    
    # Suppression loss (grad2 component)
    mean_answer_log_prob = torch.logsumexp(answer_log_probs, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32))
    supp_loss = -beta_supp * torch.mean(mean_answer_log_prob)
    
    # KL divergence loss (grad3 component)
    kl_loss = beta_kl * F.kl_div(current_log_probs, ref_log_probs, log_target=True, reduction='batchmean')
    
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