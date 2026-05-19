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
REAL Ray Trainer - combines GRPO workflow with REAL algorithm
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.torch_functional import masked_mean

from .real_core_algos import (REALBuffer, REALConfig, compute_real_advantages,
                              compute_real_gradients, real_loss)


class RayREALTrainer(RayPPOTrainer):
    """
    REAL Trainer that extends PPO/GRPO with REAL algorithm
    
    Integrates with the existing ReLIFT workflow by:
    1. Performing standard GRPO training
    2. Replacing the SFT update step with REAL training when buffer conditions are met
    3. Using the same buffer logic as ReLIFT but applying REAL algorithm instead of SFT
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize REAL components
        self.real_config = REALConfig(
            delimiter=getattr(self.config.algorithm, 'real_delimiter', '\n\n'),
            format_penalty=getattr(self.config.algorithm, 'real_format_penalty', 0.1),
            beta_supp=getattr(self.config.algorithm, 'real_beta_supp', 1.0),
            beta_kl=getattr(self.config.algorithm, 'real_beta_kl', 0.1),
            buffer_size=getattr(self.config.algorithm, 'real_buffer_size', 1000),
            real_steps=getattr(self.config.algorithm, 'real_steps', 5),
            epochs=getattr(self.config.algorithm, 'real_epochs', 1),
            mini_batch_size_per_gpu=getattr(self.config.algorithm, 'real_mini_batch_size_per_gpu', 8),
            micro_batch_size_per_gpu=getattr(self.config.algorithm, 'real_micro_batch_size_per_gpu', 1),
            num_response_per_question=getattr(self.config.algorithm, 'real_num_response_per_question', 8),
            accum_steps=getattr(self.config.algorithm, 'real_accum_steps', 4),
            responses_micro_batch_size=getattr(self.config.algorithm, 'real_responses_micro_batch_size', 8),
            # +algorithm.real_use_regression_reward=True \
            # +algorithm.real_use_last_token_as_answer=True \
            # +algorithm.real_answer_token_length=1 \
            # +algorithm.real_store_last_token_probs=True \
            # +algorithm.real_use_format_adv=${real_use_format_adv} \
            # +algorithm.real_use_log_prob_loss=${real_use_log_prob_loss} \
            # +algorithm.real_use_extra_loss=${real_use_extra_loss} \
            # +algorithm.real_use_cot_loss=${real_use_cot_loss} \
            # +algorithm.real_normalize_advantages=${real_normalize_advantages} \
                
            # add the new configs above
            use_regression_reward=getattr(self.config.algorithm, 'real_use_regression_reward', True),
            use_last_token_as_answer=getattr(self.config.algorithm, 'real_use_last_token_as_answer', True),
            answer_token_length=getattr(self.config.algorithm, 'real_answer_token_length', 1),
            store_last_token_probs=getattr(self.config.algorithm, 'real_store_last_token_probs', True),
            use_format_adv=getattr(self.config.algorithm, 'real_use_format_adv', False),
            use_log_prob_loss=getattr(self.config.algorithm, 'real_use_log_prob_loss', False),
            use_extra_loss=getattr(self.config.algorithm, 'real_use_extra_loss', False),
            use_cot_loss=getattr(self.config.algorithm, 'real_use_cot_loss', False),
            normalize_advantages=getattr(self.config.algorithm, 'real_normalize_advantages', False),
            use_l2_loss=getattr(self.config.algorithm, 'real_use_l2_loss', False),    
                
            
        )
        
        self.real_buffer = REALBuffer(self.real_config.buffer_size)
        self.real_metrics = defaultdict(list)
        
        # Enable REAL mode if configured
        self.use_real = getattr(self.config.algorithm, 'use_real', True)
        
         
        
    def _check_all_responses_incorrect(self, rewards: torch.Tensor) -> bool:
        """Check if all responses in batch have reward = 0"""
        return torch.all(rewards == 0).item()
    
    def _extract_responses_from_batch(self, data_batch) -> List[str]:
        """Extract response strings from data batch"""
        responses = []
        for i in range(len(data_batch)):
            response_ids = data_batch[i].batch["responses"]
            prompt_length = data_batch[i].batch["prompts"].shape[-1]
            valid_response_length = data_batch[i].batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]
            responses.append(response_str)
        
        return responses
    
    def _extract_prompt_and_answer(self, data_batch) -> tuple[str, str]:
        """Extract prompt and ground truth answer from first item in batch"""
        first_item = data_batch[0]
        
        # Extract prompt
        prompt_ids = first_item.batch["prompts"]
        valid_prompt_length = first_item.batch["attention_mask"][:prompt_ids.shape[-1]].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        
        # Extract ground truth answer
        answer = first_item.non_tensor_batch["reward_model"]["ground_truth"]
        
        return prompt_str, answer
    
    def _perform_jepo_training_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform one REAL training step on buffered data
        
        Args:
            batch_data: Dictionary containing prompt, answer, and responses
            
        Returns:
            Dictionary of training metrics
        """
        prompt = batch_data['prompt']
        answer = batch_data['answer']
        responses = batch_data['responses']
        
        # Tokenize responses and get model outputs
        response_tokens = []
        chain_of_thought_tokens = []
        
        for response in responses:
            # Tokenize full response
            tokens = self.tokenizer.encode(response, return_tensors='pt')
            response_tokens.append(tokens)
            
            # Split by delimiter to get chain-of-thought part
            if self.real_config.delimiter in response:
                cot_text = response.split(self.real_config.delimiter)[0]
            else:
                cot_text = response
            cot_tokens = self.tokenizer.encode(cot_text, return_tensors='pt')
            chain_of_thought_tokens.append(cot_tokens)
        
        # Get model forward passes
        with torch.no_grad():
            current_log_probs = []
            ref_log_probs = []
            answer_log_probs = []
            cot_log_probs = []
            
            for i, (resp_tokens, cot_tokens) in enumerate(zip(response_tokens, chain_of_thought_tokens)):
                # Get current model probabilities
                curr_outputs = self.actor_rollout_ref.forward(resp_tokens)
                curr_logits = curr_outputs.logits
                curr_log_prob = torch.log_softmax(curr_logits, dim=-1)
                current_log_probs.append(curr_log_prob)
                
                # Get reference model probabilities
                ref_outputs = self.ref_policy.forward(resp_tokens)  
                ref_logits = ref_outputs.logits
                ref_log_prob = torch.log_softmax(ref_logits, dim=-1)
                ref_log_probs.append(ref_log_prob)
                
                # Extract answer portion (after chain-of-thought)
                cot_length = cot_tokens.shape[-1]
                answer_portion = curr_log_prob[:, cot_length:]
                answer_log_probs.append(answer_portion)
                
                # Chain-of-thought portion
                cot_portion = curr_log_prob[:, :cot_length]  
                cot_log_probs.append(cot_portion)
        
        # Convert to tensors
        current_log_probs = torch.stack(current_log_probs)
        ref_log_probs = torch.stack(ref_log_probs)
        answer_log_probs = torch.stack(answer_log_probs)
        cot_log_probs = torch.stack(cot_log_probs)
        
        # Prepare response tokens for REAL
        response_token_lists = []
        for tokens in response_tokens:
            response_token_lists.append(tokens.squeeze(0).tolist())  # Remove batch dim and convert to list
        
        # Compute REAL advantages using ground truth answer
        tilde_A_i, tilde_A_i_ref, cot_log_probs_jepo, answer_log_probs_jepo = compute_real_advantages(
            responses=responses,
            log_probs=answer_log_probs,
            response_tokens=response_token_lists,
            tokenizer=self.tokenizer,
            delimiter=self.real_config.delimiter,
            format_penalty=self.real_config.format_penalty,
            ground_truth_answer=answer,
            model=self.actor_rollout_ref.module,
            question=prompt,
            device=self.device
        )
        
        # Compute REAL loss using the ground truth answer log probs
        loss_dict = real_loss(
            chain_of_thought_log_probs=cot_log_probs,
            answer_log_probs=answer_log_probs_jepo,  # Use ground truth answer log probs from REAL
            tilde_A_i=tilde_A_i,
            tilde_A_i_ref=tilde_A_i_ref,
            ref_log_probs=ref_log_probs,
            current_log_probs=current_log_probs,
            beta_supp=self.real_config.beta_supp,
            beta_kl=self.real_config.beta_kl
        )
        
        # Backward pass and optimization
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        
        # Update model parameters
        self.actor_rollout_ref.optimizer.step()
        self.actor_rollout_ref.optimizer.zero_grad()
        
        # Convert tensors to floats for logging
        metrics = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                metrics[f'real_{key}'] = value.item()
            else:
                metrics[f'real_{key}'] = value
                
        return metrics
    
    def _run_real_training(self) -> Dict[str, float]:
        """
        Run REAL training on all buffered data
        
        Returns:
            Aggregated training metrics
        """
        if not self.real_buffer.is_full():
            return {}
        
        print(f"Running REAL training with {len(self.real_buffer.buffer)} batches...")
        
        all_metrics = defaultdict(list)
        
        # Perform REAL steps on buffered data
        for step in range(self.real_config.real_steps):
            step_metrics = defaultdict(list)
            
            for batch_data in self.real_buffer.get_batch():
                metrics = self._perform_jepo_training_step(batch_data)
                
                for key, value in metrics.items():
                    step_metrics[key].append(value)
            
            # Average metrics across all batches in this step
            for key, values in step_metrics.items():
                avg_value = np.mean(values)
                all_metrics[f'{key}_step_{step}'].append(avg_value)
                all_metrics[key].append(avg_value)
        
        # Clear buffer after training
        self.real_buffer.clear()
        
        # Return averaged metrics
        final_metrics = {}
        for key, values in all_metrics.items():
            final_metrics[key] = np.mean(values)
        
        return final_metrics
    
    def update_policy(self, data):
        """
        Override the update_policy method to integrate REAL training
        """
        # First, run standard GRPO update
        metrics = super().update_policy(data)
        
        # Check if all responses are incorrect (reward = 0)
        rewards = data.batch.get('rewards', None)
        if rewards is not None and self._check_all_responses_incorrect(rewards):
            # Extract data for buffer
            responses = self._extract_responses_from_batch(data)
            prompt, answer = self._extract_prompt_and_answer(data)
            
            # Add to REAL buffer
            self.real_buffer.add(prompt, answer, responses)
            
            print(f"Added batch to REAL buffer. Buffer size: {len(self.real_buffer.buffer)}/{self.real_buffer.max_size}")
        
        # If buffer is full, run REAL training
        if self.real_buffer.is_full():
            real_metrics = self._run_real_training()
            metrics.update(real_metrics)
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Override to include REAL metrics in logging"""
        # Separate REAL metrics
        real_metrics = {k: v for k, v in metrics.items() if k.startswith('real_')}
        other_metrics = {k: v for k, v in metrics.items() if not k.startswith('real_')}
        
        # Log standard metrics
        super().log_metrics(other_metrics, step)
        
        # Log REAL metrics separately
        if real_metrics:
            print(f"REAL Metrics at step {step}:")
            for key, value in real_metrics.items():
                print(f"  {key}: {value}")
                # Store for later analysis
                self.real_metrics[key].append(value)