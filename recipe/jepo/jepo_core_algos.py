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
    
    def add(self, prompt: str, answer: str, responses: List[str]):
        """Add a batch of incorrect responses to buffer"""
        if len(self.buffer) >= self.max_size:
            # Remove oldest entry when buffer is full
            self.buffer.pop(0)
        
        self.buffer.append({
            'prompt': prompt,
            'answer': answer,
            'responses': responses
        })
    
    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size
    
    def clear(self):
        self.buffer.clear()
    
    def get_batch(self) -> List[Dict[str, Any]]:
        return self.buffer.copy()


def compute_jepo_advantages(
    responses: List[str],
    log_probs: torch.Tensor,
    delimiter: str,
    format_penalty: float,
    pi_theta: torch.Tensor,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute JEPO advantages based on the algorithm described in jepo.md
    
    Args:
        responses: List of n response strings
        log_probs: Log probabilities for each response [n, seq_len]
        delimiter: String used to split chain-of-thought from response
        format_penalty: Penalty p for responses without delimiter
        pi_theta: Policy probabilities [n, vocab_size]
        device: Device to place tensors on
        
    Returns:
        tilde_A_i: Clipped advantages for chain-of-thought
        tilde_A_i_ref: Normalized format advantages
    """
    n = len(responses)
    
    # Step 1: Split responses and extract chain-of-thought
    chain_of_thoughts = []
    has_delimiter = []
    
    for response in responses:
        if delimiter in response:
            cot = response.split(delimiter)[0]
            has_delimiter.append(True)
        else:
            cot = response
            has_delimiter.append(False)
        chain_of_thoughts.append(cot)
    
    # Step 2: Calculate A_i for each response
    A_values = []
    
    # Use mean of log_probs across sequence dimension for each response
    response_log_probs = torch.mean(log_probs, dim=-1)  # [n]
    
    for i in range(n):
        # Calculate log(1/n * sum_j pi_theta(a|x,c_j))
        log_mean_prob = torch.logsumexp(response_log_probs, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32))
        
        # Calculate v_i = log(1/(n-1) * sum_{j!=i} pi_theta(a|x,c_j))
        other_response_log_probs = torch.cat([response_log_probs[:i], response_log_probs[i+1:]], dim=0)
        if len(other_response_log_probs) > 0:
            v_i = torch.logsumexp(other_response_log_probs, dim=0) - torch.log(torch.tensor(n-1, dtype=torch.float32))
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
    
    return tilde_A_i.to(device), tilde_A_i_ref.to(device)


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