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
JEPO Ray Trainer - combines GRPO workflow with JEPO algorithm
"""

from typing import Dict, List, Any, Optional
import torch
import numpy as np
from collections import defaultdict

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.utils.torch_functional import masked_mean
from .jepo_core_algos import (
    JEPOConfig, 
    JEPOBuffer, 
    compute_jepo_advantages,
    compute_jepo_gradients,
    jepo_loss
)


class RayJEPOTrainer(RayPPOTrainer):
    """
    JEPO Trainer that extends PPO/GRPO with JEPO algorithm
    
    Integrates with the existing ReLIFT workflow by:
    1. Performing standard GRPO training
    2. Replacing the SFT update step with JEPO training when buffer conditions are met
    3. Using the same buffer logic as ReLIFT but applying JEPO algorithm instead of SFT
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize JEPO components
        self.jepo_config = JEPOConfig(
            delimiter=getattr(self.config.algorithm, 'jepo_delimiter', '\n\n'),
            format_penalty=getattr(self.config.algorithm, 'jepo_format_penalty', 0.1),
            beta_supp=getattr(self.config.algorithm, 'jepo_beta_supp', 1.0),
            beta_kl=getattr(self.config.algorithm, 'jepo_beta_kl', 0.1),
            buffer_size=getattr(self.config.algorithm, 'jepo_buffer_size', 1000),
            jepo_steps=getattr(self.config.algorithm, 'jepo_steps', 5)
        )
        
        self.jepo_buffer = JEPOBuffer(self.jepo_config.buffer_size)
        self.jepo_metrics = defaultdict(list)
        
        # Enable JEPO mode if configured
        self.use_jepo = getattr(self.config.algorithm, 'use_jepo', True)
        
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
        Perform one JEPO training step on buffered data
        
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
            if self.jepo_config.delimiter in response:
                cot_text = response.split(self.jepo_config.delimiter)[0]
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
        
        # Compute JEPO advantages
        tilde_A_i, tilde_A_i_ref = compute_jepo_advantages(
            responses=responses,
            log_probs=answer_log_probs,
            delimiter=self.jepo_config.delimiter,
            format_penalty=self.jepo_config.format_penalty,
            pi_theta=current_log_probs,
            device=self.device
        )
        
        # Compute JEPO loss
        loss_dict = jepo_loss(
            chain_of_thought_log_probs=cot_log_probs,
            answer_log_probs=answer_log_probs,
            tilde_A_i=tilde_A_i,
            tilde_A_i_ref=tilde_A_i_ref,
            ref_log_probs=ref_log_probs,
            current_log_probs=current_log_probs,
            beta_supp=self.jepo_config.beta_supp,
            beta_kl=self.jepo_config.beta_kl
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
                metrics[f'jepo_{key}'] = value.item()
            else:
                metrics[f'jepo_{key}'] = value
                
        return metrics
    
    def _run_jepo_training(self) -> Dict[str, float]:
        """
        Run JEPO training on all buffered data
        
        Returns:
            Aggregated training metrics
        """
        if not self.jepo_buffer.is_full():
            return {}
        
        print(f"Running JEPO training with {len(self.jepo_buffer.buffer)} batches...")
        
        all_metrics = defaultdict(list)
        
        # Perform JEPO steps on buffered data
        for step in range(self.jepo_config.jepo_steps):
            step_metrics = defaultdict(list)
            
            for batch_data in self.jepo_buffer.get_batch():
                metrics = self._perform_jepo_training_step(batch_data)
                
                for key, value in metrics.items():
                    step_metrics[key].append(value)
            
            # Average metrics across all batches in this step
            for key, values in step_metrics.items():
                avg_value = np.mean(values)
                all_metrics[f'{key}_step_{step}'].append(avg_value)
                all_metrics[key].append(avg_value)
        
        # Clear buffer after training
        self.jepo_buffer.clear()
        
        # Return averaged metrics
        final_metrics = {}
        for key, values in all_metrics.items():
            final_metrics[key] = np.mean(values)
        
        return final_metrics
    
    def update_policy(self, data):
        """
        Override the update_policy method to integrate JEPO training
        """
        # First, run standard GRPO update
        metrics = super().update_policy(data)
        
        # Check if all responses are incorrect (reward = 0)
        rewards = data.batch.get('rewards', None)
        if rewards is not None and self._check_all_responses_incorrect(rewards):
            # Extract data for buffer
            responses = self._extract_responses_from_batch(data)
            prompt, answer = self._extract_prompt_and_answer(data)
            
            # Add to JEPO buffer
            self.jepo_buffer.add(prompt, answer, responses)
            
            print(f"Added batch to JEPO buffer. Buffer size: {len(self.jepo_buffer.buffer)}/{self.jepo_buffer.max_size}")
        
        # If buffer is full, run JEPO training
        if self.jepo_buffer.is_full():
            jepo_metrics = self._run_jepo_training()
            metrics.update(jepo_metrics)
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Override to include JEPO metrics in logging"""
        # Separate JEPO metrics
        jepo_metrics = {k: v for k, v in metrics.items() if k.startswith('jepo_')}
        other_metrics = {k: v for k, v in metrics.items() if not k.startswith('jepo_')}
        
        # Log standard metrics
        super().log_metrics(other_metrics, step)
        
        # Log JEPO metrics separately
        if jepo_metrics:
            print(f"JEPO Metrics at step {step}:")
            for key, value in jepo_metrics.items():
                print(f"  {key}: {value}")
                # Store for later analysis
                self.jepo_metrics[key].append(value)