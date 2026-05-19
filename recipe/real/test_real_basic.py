#!/usr/bin/env python3
"""
Basic test script for REAL algorithm components
"""

import torch
import numpy as np
from real_core_algos import (
    REALConfig, 
    REALBuffer, 
    compute_real_advantages,
    real_loss
)

def test_jepo_buffer():
    """Test REAL buffer functionality"""
    print("Testing REAL buffer...")
    
    buffer = REALBuffer(max_size=3)
    assert not buffer.is_full()
    
    # Add some test data
    buffer.add("What is 2+2?", "4", ["Let me think.\n\n4", "2+2=4", "I think it's 4"])
    buffer.add("What is 3+3?", "6", ["Let me think.\n\n6", "3+3=6", "I think it's 6"])
    assert not buffer.is_full()
    
    buffer.add("What is 4+4?", "8", ["Let me think.\n\n8", "4+4=8", "I think it's 8"])
    assert buffer.is_full()
    
    # Adding another should remove the oldest
    buffer.add("What is 5+5?", "10", ["Let me think.\n\n10", "5+5=10", "I think it's 10"])
    assert buffer.is_full()
    assert len(buffer.buffer) == 3
    
    print("✓ REAL buffer test passed")

def test_jepo_advantages():
    """Test REAL advantage computation"""
    print("Testing REAL advantages...")
    
    # Mock data
    responses = [
        "Let me think step by step.\n\nThe answer is 4",
        "I need to calculate this.\n\nThe answer is 4", 
        "This is easy.\n\nThe answer is 4"
    ]
    
    # Mock log probabilities (batch_size=3, seq_len=10)
    log_probs = torch.randn(3, 10)
    
    # Mock tokenizer
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False, return_tensors=None):
            # Simple mock: return list of token IDs based on text length
            if return_tensors == "pt":
                return torch.tensor([[i for i in range(len(text.split()))]])
            return [i for i in range(len(text.split()))]
    
    tokenizer = MockTokenizer()
    
    # Mock model
    class MockModel:
        def __call__(self, input_ids, labels=None):
            class MockOutput:
                def __init__(self, seq_len, vocab_size=1000):
                    self.logits = torch.randn(1, seq_len, vocab_size)
            return MockOutput(input_ids.shape[1])
    
    model = MockModel()
    
    # Mock response tokens
    response_tokens = [
        tokenizer.encode(response) for response in responses
    ]
    
    tilde_A_i, tilde_A_i_ref, cot_log_probs, answer_log_probs = compute_real_advantages(
        responses=responses,
        log_probs=log_probs,
        response_tokens=response_tokens,
        tokenizer=tokenizer,
        delimiter="\n\n",
        format_penalty=0.1,
        ground_truth_answer="4",
        model=model,
        question="What is 2+2?",
        device=torch.device('cpu')
    )
    
    assert tilde_A_i.shape == (3,)
    assert tilde_A_i_ref.shape == (3,)
    assert cot_log_probs.shape == (3,)
    assert answer_log_probs.shape == (3,)
    assert torch.all(torch.abs(tilde_A_i) <= 1.0)  # Should be clipped to [-1, 1]
    
    print("✓ REAL advantages test passed")

def test_jepo_loss():
    """Test REAL loss computation"""
    print("Testing REAL loss...")
    
    # Mock data
    batch_size, cot_seq_len, ans_seq_len = 3, 5, 8
    
    cot_log_probs = torch.randn(batch_size, cot_seq_len)
    ans_log_probs = torch.randn(batch_size, ans_seq_len)
    tilde_A_i = torch.randn(batch_size)
    tilde_A_i_ref = torch.randn(batch_size)
    ref_log_probs = torch.randn(batch_size, cot_seq_len + ans_seq_len)
    current_log_probs = torch.randn(batch_size, cot_seq_len + ans_seq_len)
    
    loss_dict = real_loss(
        chain_of_thought_log_probs=cot_log_probs,
        answer_log_probs=ans_log_probs,
        tilde_A_i=tilde_A_i,
        tilde_A_i_ref=tilde_A_i_ref,
        ref_log_probs=ref_log_probs,
        current_log_probs=current_log_probs,
        beta_supp=1.0,
        beta_kl=0.1
    )
    
    # Check that all expected keys are present
    expected_keys = ['total_loss', 'pg_loss', 'supp_loss', 'kl_loss', 
                     'advantages_mean', 'advantages_std', 'tilde_A_i_mean', 'tilde_A_i_ref_mean']
    
    for key in expected_keys:
        assert key in loss_dict, f"Missing key: {key}"
        assert isinstance(loss_dict[key], torch.Tensor), f"Key {key} should be a tensor"
    
    print("✓ REAL loss test passed")

def test_jepo_config():
    """Test REAL configuration"""
    print("Testing REAL config...")
    
    config = REALConfig()
    assert config.delimiter == "\n\n"
    assert config.format_penalty == 0.1
    assert config.beta_supp == 1.0
    assert config.beta_kl == 0.1
    assert config.buffer_size == 1000
    assert config.real_steps == 5
    
    # Test custom config
    custom_config = REALConfig(
        delimiter="###",
        format_penalty=0.2,
        buffer_size=500
    )
    assert custom_config.delimiter == "###"
    assert custom_config.format_penalty == 0.2
    assert custom_config.buffer_size == 500
    
    print("✓ REAL config test passed")

def main():
    """Run all tests"""
    print("Running REAL basic tests...\n")
    
    test_jepo_config()
    test_jepo_buffer()
    test_jepo_advantages()
    test_jepo_loss()
    
    print("\n🎉 All REAL tests passed!")

if __name__ == "__main__":
    main()