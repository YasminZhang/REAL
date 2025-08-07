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
JEPO Actor that extends DataParallelPPOActor with JEPO-specific loss computation
"""

import logging
import os
import sys
import torch

# Add the parent directory to path to import jepo_core_algos
sys.path.append('/home/aiscuser/jepo/recipe/jepo')

from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.workers.actor.dp_actor import DataParallelPPOActor
from jepo_core_algos import compute_jepo_advantages, jepo_loss

__all__ = ["JEPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class JEPOActor(DataParallelPPOActor):
    """
    JEPO Actor that extends DataParallelPPOActor to use JEPO loss computation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initialized JEPO Actor")

    def update_policy(self, data: DataProto):
        """
        Override update_policy to use JEPO loss computation
        """
        # Make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        
        # Extract JEPO configuration from metadata
        jepo_config = data.meta_info.get("jepo_config", {})
        delimiter = jepo_config.get("delimiter", "\n\n")
        format_penalty = jepo_config.get("format_penalty", 1.0)
        beta_supp = jepo_config.get("beta_supp", 1.0)
        beta_kl = jepo_config.get("beta_kl", 0.1)
        
        # Check if this is JEPO mode with pre-computed data
        use_jepo = data.meta_info.get("use_jepo", False)
        jepo_data = data.meta_info.get("jepo_data", None)
        
        if use_jepo and jepo_data is not None:
            # JEPO mode: work directly with buffered data
            return self._update_policy_jepo_mode(jepo_data, temperature, delimiter, format_penalty, beta_supp, beta_kl)

        # Standard PPO mode
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask", 
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    from verl.utils.seqlen_balancing import prepare_dynamic_batch
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # Forward pass to get current log probabilities
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=True
                    )

                    # JEPO Actor: All data is pre-filtered to be incorrect responses only
                    response_tokens_tensor = model_inputs.get("responses", None)  # These are already tokens
                    uids = model_inputs.get("uid", [])
                    prompts = model_inputs.get("prompts", [])
                    ground_truths = model_inputs.get("ground_truth", [])
                    
                    if response_tokens_tensor is None or len(response_tokens_tensor) == 0:
                        # Fall back to standard PPO if no responses
                        logger.warning("No responses found in JEPO actor, falling back to PPO")
                        from verl.trainer.ppo.core_algos import get_policy_loss_fn
                        policy_loss_fn = get_policy_loss_fn(self.config.policy_loss.get("loss_mode", "vanilla"))
                        advantages = model_inputs["advantages"]
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                            config=self.config,
                        )
                        policy_loss = pg_loss
                        micro_batch_metrics.update({
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        })
                    else:
                        # Apply JEPO to all data (pre-filtered by trainer)
                        try:
                            import numpy as np
                            from jepo_core_algos import compute_jepo_advantages_batched, jepo_loss_batched
                            
                            # Group by UID (all responses should be incorrect)
                            unique_uids = np.unique(uids)
                            
                            # Collect all questions for batched processing
                            jepo_questions_responses = []
                            jepo_questions_log_probs = []
                            jepo_questions_response_tokens = []
                            jepo_questions_old_log_probs = []
                            jepo_questions_prompts = []
                            jepo_questions_ground_truths = []
                            
                            # Extract tokenizer once
                            tokenizer = getattr(self.actor_module, 'tokenizer', None)
                            if tokenizer is None:
                                tokenizer = getattr(self.config, 'tokenizer', None)
                            
                            if tokenizer is None:
                                raise ValueError("Tokenizer not found - cannot process JEPO")
                            
                            for uid in unique_uids:
                                uid_mask = torch.tensor([u == uid for u in uids], dtype=torch.bool, device=log_prob.device)
                                uid_indices = torch.where(uid_mask)[0]
                                
                                if len(uid_indices) > 1:  # Need multiple responses for JEPO
                                    # Get response tokens directly (already encoded)
                                    uid_response_tokens_tensor = response_tokens_tensor[uid_mask]
                                    uid_log_prob = log_prob[uid_mask]
                                    uid_old_log_prob = old_log_prob[uid_mask]
                                    
                                    # Convert tensor tokens to list of token lists (no string decoding needed)
                                    response_tokens = []
                                    for i in range(uid_response_tokens_tensor.shape[0]):
                                        tokens = uid_response_tokens_tensor[i].cpu().tolist()
                                        # Remove padding tokens (typically 0 or tokenizer.pad_token_id)
                                        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                                            tokens = [t for t in tokens if t != tokenizer.pad_token_id]
                                        response_tokens.append(tokens)
                                    
                                    # Extract prompt and ground truth for this UID (should be same for all responses with same UID)
                                    uid_prompt = prompts[uid_indices[0]] if prompts else ""
                                    uid_ground_truth = ground_truths[uid_indices[0]] if ground_truths else ""
                                    
                                    # Pass None for responses since JEPO algorithm now works directly with tokens
                                    jepo_questions_responses.append(None)  # No longer needed - algorithm works with tokens
                                    jepo_questions_log_probs.append(uid_log_prob)
                                    jepo_questions_response_tokens.append(response_tokens)
                                    jepo_questions_old_log_probs.append(uid_old_log_prob)
                                    jepo_questions_prompts.append(uid_prompt)
                                    jepo_questions_ground_truths.append(uid_ground_truth)
                            
                            if jepo_questions_responses:
                                # Batched JEPO computation
                                tilde_A_i_list, tilde_A_i_ref_list, cot_log_probs_list, answer_log_probs_list = compute_jepo_advantages_batched(
                                    questions_responses=jepo_questions_responses,
                                    questions_log_probs=jepo_questions_log_probs,
                                    questions_response_tokens=jepo_questions_response_tokens,
                                    questions=jepo_questions_prompts,
                                    ground_truth_answers=jepo_questions_ground_truths,
                                    tokenizer=tokenizer,
                                    delimiter=delimiter,
                                    format_penalty=format_penalty,
                                    model=self.actor_module,
                                    device=log_prob.device
                                )
                                
                                # Batched JEPO loss computation
                                jepo_loss_components = jepo_loss_batched(
                                    questions_cot_log_probs=cot_log_probs_list,
                                    questions_answer_log_probs=answer_log_probs_list,
                                    questions_tilde_A_i=tilde_A_i_list,
                                    questions_tilde_A_i_ref=tilde_A_i_ref_list,
                                    questions_ref_log_probs=jepo_questions_old_log_probs,
                                    questions_current_log_probs=jepo_questions_log_probs,
                                    beta_supp=beta_supp,
                                    beta_kl=beta_kl
                                )
                                
                                policy_loss = jepo_loss_components["total_loss"]
                                
                                # Add JEPO-specific metrics
                                total_samples = sum(len(responses) for responses in jepo_questions_responses)
                                micro_batch_metrics.update({
                                    "jepo_actor/grad1_loss": jepo_loss_components["grad1_loss"].detach().item() * loss_scale_factor,
                                    "jepo_actor/grad2_loss": jepo_loss_components["grad2_loss"].detach().item() * loss_scale_factor,
                                    "jepo_actor/grad3_loss": jepo_loss_components["grad3_loss"].detach().item() * loss_scale_factor,
                                    "jepo_actor/total_loss": jepo_loss_components["total_loss"].detach().item() * loss_scale_factor,
                                    "jepo_actor/jepo_samples": total_samples,
                                    "jepo_actor/num_jepo_questions": len(jepo_questions_responses),
                                })
                            else:
                                # No valid JEPO groups, fallback to zero loss
                                policy_loss = torch.tensor(0.0, device=log_prob.device)
                                micro_batch_metrics.update({
                                    "jepo_actor/no_valid_groups": 1,
                                })
                                
                        except Exception as e:
                            logger.error(f"JEPO computation failed: {e}, using zero loss")
                            policy_loss = torch.tensor(0.0, device=log_prob.device)
                            micro_batch_metrics.update({
                                "jepo_actor/computation_failed": 1,
                            })

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    
                    loss.backward()
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"jepo_actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        return metrics
    
    def _update_policy_jepo_mode(self, jepo_data: dict, temperature: float, delimiter: str, 
                                 format_penalty: float, beta_supp: float, beta_kl: float):
        """
        Handle JEPO training with pre-computed buffered data
        """
        try:
            # Extract data from the buffer
            responses = jepo_data.get('responses', [])
            prompt = jepo_data.get('prompt', '')
            answer = jepo_data.get('answer', '')
            
            if not responses:
                logger.warning("No responses found in JEPO data")
                return {"jepo_actor/no_responses": 1}
            
            # Extract tokenizer
            tokenizer = getattr(self.actor_module, 'tokenizer', None) or getattr(self.config, 'tokenizer', None)
            if tokenizer is None:
                logger.error("Tokenizer not found for JEPO mode")
                return {"jepo_actor/no_tokenizer": 1}
            
            # Prepare inputs for the model
            # Create input prompts for each response to compute log probs
            full_texts = [prompt + response for response in responses]
            
            # Tokenize all texts
            tokenized_inputs = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = tokenized_inputs['input_ids'].to(self.actor_module.device)
            attention_mask = tokenized_inputs['attention_mask'].to(self.actor_module.device)
            
            # Get log probabilities from the model
            with torch.no_grad():
                outputs = self.actor_module(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # Convert logits to log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Extract response tokens and their log probs
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
            
            response_tokens_list = []
            response_log_probs_list = []
            
            for i, response in enumerate(responses):
                response_tokens = tokenizer.encode(response, add_special_tokens=False)
                response_tokens_list.append(response_tokens)
                
                # Extract log probs for this response (after prompt)
                seq_len = attention_mask[i].sum().item()
                response_start = min(prompt_len, seq_len - len(response_tokens))
                response_end = min(seq_len, response_start + len(response_tokens))
                
                # Get log probs for the response tokens
                if response_end > response_start:
                    response_log_probs = log_probs[i, response_start:response_end-1]  # Shift by 1 for next token prediction
                    response_log_probs_list.append(response_log_probs.sum())
                else:
                    response_log_probs_list.append(torch.tensor(0.0, device=log_probs.device))
            
            # Stack log probs into tensor
            response_log_probs_tensor = torch.stack(response_log_probs_list)
            
            # Call JEPO algorithm
            from jepo_core_algos import compute_jepo_advantages, jepo_loss
            
            tilde_A_i, tilde_A_i_ref, cot_log_probs, answer_log_probs = compute_jepo_advantages(
                log_probs=response_log_probs_tensor.unsqueeze(-1),  # Add sequence dimension
                response_tokens=response_tokens_list,
                tokenizer=tokenizer,
                delimiter=delimiter,
                format_penalty=format_penalty,
                ground_truth_answer=answer,
                model=self.actor_module,
                question=prompt,
                device=self.actor_module.device,
                responses=responses
            )
            
            # Compute JEPO loss (simplified version)
            total_loss = jepo_loss(
                cot_log_probs=cot_log_probs,
                answer_log_probs=answer_log_probs,
                tilde_A_i=tilde_A_i,
                tilde_A_i_ref=tilde_A_i_ref,
                beta_supp=beta_supp,
                beta_kl=beta_kl
            )
            
            # Backward pass
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            grad_norm = self._optimizer_step()
            
            # Return metrics
            return {
                "jepo_actor/total_loss": total_loss.detach().item(),
                "jepo_actor/grad_norm": grad_norm.detach().item(),
                "jepo_actor/num_responses": len(responses)
            }
            
        except Exception as e:
            logger.error(f"JEPO mode training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"jepo_actor/jepo_mode_failed": 1}