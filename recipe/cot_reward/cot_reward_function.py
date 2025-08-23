"""
CoT Reward Function Implementation

This module implements the Chain-of-Thought reward function that calculates
the reward based on likelihood ratios π_θ(a|c,x) / π_θ(a|x), where:
- a: ground truth answer  
- c: chain of thought (CoT) reasoning
- x: question/prompt
- π_θ: current model being trained

The log probabilities are calculated during rollouts and passed through extra_info
to avoid expensive model initialization in the reward function.
"""

import torch
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class CoTRewardConfig:
    """Configuration for CoT reward calculation."""
    delimiter: str = "\\boxed{"  # Delimiter to split CoT from answer
    min_cot_length: int = 10  # Minimum CoT length to be considered valid
    max_ratio: float = 10.0  # Maximum allowed ratio to prevent extreme values
    log_rewards: bool = True  # Whether to log reward calculations


def split_response_by_delimiter(response: str, delimiter: str = "\\boxed{") -> Tuple[str, str]:
    """
    Split response by delimiter into CoT and answer parts.
    
    Args:
        response: Full model response containing CoT and answer
        delimiter: Delimiter to split on
        
    Returns:
        Tuple of (cot_part, answer_part)
    """
    # Find the delimiter
    delimiter_pos = response.find(delimiter)
    
    if delimiter_pos == -1:
        # No delimiter found, treat entire response as CoT
        return response, ""
        
    # Split at delimiter
    cot_part = response[:delimiter_pos].strip()
    answer_part = response[delimiter_pos:].strip()
    
    return cot_part, answer_part


def extract_ground_truth_answer(ground_truth: Any) -> str:
    """
    Extract the answer from ground truth, handling various formats.
    """
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("answer", "")
        
    if isinstance(ground_truth, (list, tuple)) and len(ground_truth) > 0:
        ground_truth = ground_truth[0]
        
    gt_str = str(ground_truth).strip()
    
    if "\\boxed{" in gt_str:
        # Extract content from \boxed{}
        match = re.search(r"\\boxed\{([^}]+)\}", gt_str)
        if match:
            return match.group(1).strip()
            
    return gt_str


def calculate_log_prob_with_context(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str, 
    context: str, 
    answer: str,
    device: str = "cuda"
) -> float:
    """
    Calculate log probability of answer given question and context.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        question: The original question/prompt
        context: The context (CoT or empty for baseline)
        answer: The ground truth answer
        device: Device to run computation on
        
    Returns:
        Log probability of the answer
    """
    try:
        if context:
            full_prompt = f"{question}\n{context}\n{answer}"
            # We want P(answer | question, context)
            prompt_part = f"{question}\n{context}\n"
        else:
            full_prompt = f"{question}\n{answer}"
            # We want P(answer | question)  
            prompt_part = f"{question}\n"
            
        # Tokenize
        prompt_tokens = tokenizer.encode(prompt_part, return_tensors="pt", add_special_tokens=False)
        full_tokens = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
        
        # Get answer tokens (the part we want to compute probability for)
        answer_tokens = full_tokens[:, prompt_tokens.shape[1]:]
        
        if answer_tokens.shape[1] == 0:
            return float('-inf')  # No answer tokens
            
        # Move to device
        full_tokens = full_tokens.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=full_tokens)
            logits = outputs.logits
            
        # Get logits for the answer portion
        # logits shape: [batch_size, sequence_length, vocab_size]
        answer_logits = logits[:, prompt_tokens.shape[1]-1:-1, :]  # -1 to align with targets
        
        # Calculate log probabilities
        log_probs = torch.log_softmax(answer_logits, dim=-1)
        
        # Extract log probabilities for actual answer tokens
        answer_log_probs = []
        for i, token_id in enumerate(answer_tokens[0]):
            if i < log_probs.shape[1]:
                token_log_prob = log_probs[0, i, token_id.item()].item()
                answer_log_probs.append(token_log_prob)
                
        # Sum log probabilities (log of product = sum of logs)
        total_log_prob = sum(answer_log_probs) if answer_log_probs else float('-inf')
        
        return total_log_prob
        
    except Exception as e:
        print(f"Error calculating log probability: {e}")
        return float('-inf')


def compute_cot_log_probs(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    response: str,
    ground_truth: str,
    delimiter: str = "\\boxed{",
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute the log probabilities needed for CoT reward calculation.
    This should be called during rollouts and the results stored in DataProto.
    
    Args:
        model: The current model
        tokenizer: Model tokenizer
        question: Original question/prompt
        response: Model response containing CoT
        ground_truth: Ground truth answer
        delimiter: Delimiter to split CoT from answer
        device: Device for computation
        
    Returns:
        Dictionary containing log probabilities and metadata
    """
    try:
        # Split response into CoT and answer parts
        cot_part, _ = split_response_by_delimiter(response, delimiter)
        
        # Extract ground truth answer
        gt_answer = extract_ground_truth_answer(ground_truth)
        
        if not gt_answer:
            return {
                "log_prob_with_cot": float('-inf'),
                "log_prob_without_cot": float('-inf'),
                "cot_length": 0,
                "has_valid_gt": False,
                "error": "No ground truth available"
            }
            
        # Calculate log P(a|c,x) - probability of answer given CoT and question
        log_prob_with_cot = calculate_log_prob_with_context(
            model, tokenizer, question, cot_part, gt_answer, device
        )
        
        # Calculate log P(a|x) - probability of answer given only question  
        log_prob_without_cot = calculate_log_prob_with_context(
            model, tokenizer, question, "", gt_answer, device
        )
        
        return {
            "log_prob_with_cot": log_prob_with_cot,
            "log_prob_without_cot": log_prob_without_cot,
            "cot_length": len(cot_part),
            "has_valid_gt": True,
            "cot_part": cot_part,
            "gt_answer": gt_answer
        }
        
    except Exception as e:
        return {
            "log_prob_with_cot": float('-inf'),
            "log_prob_without_cot": float('-inf'),
            "cot_length": 0,
            "has_valid_gt": False,
            "error": str(e)
        }


def cot_reward_function(
    data_source: str, 
    solution_str: str, 
    ground_truth: str, 
    extra_info: Dict = None
) -> float:
    """
    CoT reward function that calculates likelihood ratio from pre-computed log probabilities.
    
    The log probabilities should be computed during rollouts and passed through extra_info
    to avoid expensive model operations in the reward function.
    
    Args:
        data_source: Dataset source identifier
        solution_str: Model's full response containing CoT and answer
        ground_truth: Ground truth answer
        extra_info: Dictionary containing pre-computed log probabilities:
            - "cot_log_probs": Dict with log_prob_with_cot, log_prob_without_cot, etc.
        
    Returns:
        Reward value based on likelihood ratio π_θ(a|c,x) / π_θ(a|x)
    """
    config = CoTRewardConfig()
    
    # Fast path: allow passing ratio directly via extra_info["ratio"]
    if extra_info and isinstance(extra_info, dict) and "ratio" in extra_info:
        ratio = float(extra_info["ratio"])
        # Clamp to safe range
        ratio = max(0.0, min(ratio, config.max_ratio))
        if config.log_rewards:
            print(f"CoT reward (direct ratio) = {ratio:.4f}")
        return ratio

    # Otherwise, fallback to nested cot_log_probs structure
    if not extra_info or "cot_log_probs" not in extra_info:
        if config.log_rewards:
            print("Warning: No pre-computed CoT log probabilities found in extra_info")
        return 0.0
        
    cot_data = extra_info["cot_log_probs"]
    
    # Check if computation was successful
    if not cot_data.get("has_valid_gt", False):
        if config.log_rewards:
            print(f"Invalid ground truth or computation error: {cot_data.get('error', 'Unknown error')}")
        return 0.0
        
    # If ratio is provided in nested dict, use it directly
    if "ratio" in cot_data:
        ratio = float(cot_data["ratio"]) if cot_data["ratio"] is not None else 0.0
        ratio = max(0.0, min(ratio, config.max_ratio))
        if config.log_rewards:
            print(f"CoT reward (nested ratio) = {ratio:.4f}")
        return ratio

    log_prob_with_cot = cot_data.get("log_prob_with_cot", float('-inf'))
    log_prob_without_cot = cot_data.get("log_prob_without_cot", float('-inf'))
    cot_length = cot_data.get("cot_length", 0)
    # Check for valid log probabilities
    if log_prob_with_cot == float('-inf') or log_prob_without_cot == float('-inf'):
        if config.log_rewards:
            print("Invalid log probabilities computed")
        return 0.0
        
    # Check minimum CoT length requirement
    if cot_length < config.min_cot_length:
        if config.log_rewards:
            print(f"CoT too short: {cot_length} < {config.min_cot_length}")
        return 0.0
        
    # Calculate log ratio: log(P(a|c,x) / P(a|x)) = log(P(a|c,x)) - log(P(a|x))
    log_ratio = log_prob_with_cot - log_prob_without_cot
    
    # Convert to probability ratio
    ratio = torch.exp(torch.tensor(log_ratio)).item()
    
    # Clamp to reasonable range to avoid extreme values
    ratio = max(0.0, min(ratio, config.max_ratio))
    
    if config.log_rewards:
        print(f"CoT reward calculated: log_ratio={log_ratio:.4f}, ratio={ratio:.4f}, cot_length={cot_length}")
    
    return ratio


# Alternative simple version for testing without model access
def simple_cot_reward_function(
    data_source: str, 
    solution_str: str, 
    ground_truth: str, 
    extra_info: Dict = None
) -> float:
    """
    Simplified CoT reward function for testing that doesn't require model access.
    Just checks if the response contains reasoning before the delimiter.
    """
    delimiter = "\\boxed{"
    
    # Split response by delimiter
    delimiter_pos = solution_str.find(delimiter)
    
    if delimiter_pos == -1:
        return 0.0  # No delimiter found
        
    cot_part = solution_str[:delimiter_pos].strip()
    
    # Simple heuristic: reward based on length and content of reasoning
    if len(cot_part) > 100:  # Substantial reasoning
        return 1.0
    elif len(cot_part) > 50:  # Moderate reasoning
        return 0.7
    elif len(cot_part) > 10:  # Some reasoning
        return 0.3
    else:
        return 0.0  # Little to no reasoning
