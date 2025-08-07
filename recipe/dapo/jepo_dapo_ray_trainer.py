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
JEPO-enhanced DAPO Trainer that integrates JEPO algorithm with DAPO workflow
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Dict, List, Any
import torch
import numpy as np
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.trainer.ppo.ray_trainer import Role

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

# Import JEPO components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'jepo'))

from jepo_core_algos import (
    JEPOConfig,
    JEPOBuffer,
    compute_jepo_advantages,
    jepo_loss
)


class RayJEPODAPOTrainer(RayDAPOTrainer):
    """
    JEPO-enhanced DAPO Trainer that integrates JEPO algorithm with standard DAPO training
    
    Workflow:
    1. Perform standard DAPO training (GRPO with advantage estimation)
    2. Track responses where all answers are incorrect (reward = 0)
    3. When JEPO buffer reaches threshold, perform JEPO training steps
    4. JEPO training replaces/augments the standard actor update for incorrect response batches
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize parent DAPO trainer without modifying workers
        # JEPO logic will be handled in the trainer methods, not by replacing workers
        super().__init__(*args, **kwargs)
        
        print(f"JEPO-DAPO Trainer initialized with actor class: {self.role_worker_mapping.get(Role.ActorRollout, 'Unknown')}")
        
        # Initialize JEPO components
        self.jepo_config = JEPOConfig(
            delimiter=getattr(self.config.algorithm, 'jepo_delimiter', '\n\n'),
            format_penalty=getattr(self.config.algorithm, 'jepo_format_penalty', 0.1),
            beta_supp=getattr(self.config.algorithm, 'jepo_beta_supp', 1.0),
            beta_kl=getattr(self.config.algorithm, 'jepo_beta_kl', 0.1),
            buffer_size=getattr(self.config.algorithm, 'jepo_buffer_size', 100),
            jepo_steps=getattr(self.config.algorithm, 'jepo_steps', 5)
        )
        
        self.jepo_buffer = JEPOBuffer(self.jepo_config.buffer_size)
        self.jepo_metrics = defaultdict(list)
        
        # Enable JEPO mode
        self.use_jepo = getattr(self.config.algorithm, 'use_jepo', True)
        self.jepo_update_frequency = getattr(self.config.algorithm, 'jepo_update_frequency', 10)
        
        print(f"JEPO-DAPO Trainer initialized with buffer_size={self.jepo_config.buffer_size}, use_jepo={self.use_jepo}")
    
    def _extract_responses_from_batch(self, data_batch) -> List[str]:
        """Extract response strings from data batch"""
        responses = []
        for i in range(len(data_batch)):
            response_ids = data_batch[i].batch["responses"]
            # Get the valid response length using attention mask
            prompt_length = data_batch[i].batch["prompts"].shape[-1]
            full_length = data_batch[i].batch["attention_mask"].shape[-1]
            response_length = full_length - prompt_length
            valid_response_length = data_batch[i].batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if eos_token and response_str.endswith(eos_token):
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
    
    def _perform_jepo_training_step(self, jepo_batch: DataProto) -> Dict[str, float]:
        """
        Perform one JEPO training step using the worker group
        
        Args:
            jepo_batch: DataProto containing buffered data with all-incorrect responses
                      - prompts are under batch.batch["prompts"] (tokens)
                      - responses are under batch.batch["responses"] (tokens)  
                      - ground_truth_answer is under batch.non_tensor_batch["reward_model"]["ground_truth"] (str)
            
        Returns:
            Dictionary of training metrics
        """
        # Extract prompt and ground truth for JEPO metadata
        # Note: prompts and responses are tokens, ground_truth is string
        prompts_tokens = jepo_batch.batch.get("prompts", [])
        ground_truth_data = jepo_batch.non_tensor_batch.get("reward_model", {})
        ground_truth_answer = ground_truth_data.get("ground_truth", "")
        
        # For JEPO, we need the first prompt as string (convert from tokens if needed)
        if len(prompts_tokens) > 0:
            # Assuming we have a tokenizer available
            if hasattr(self, 'tokenizer'):
                prompt_str = self.tokenizer.decode(prompts_tokens[0], skip_special_tokens=True)
            else:
                prompt_str = str(prompts_tokens[0])  # Fallback
        else:
            prompt_str = ""
        
        # Add JEPO-specific configuration and metadata to the existing DataProto
        jepo_batch.meta_info.update({
            "jepo_config": {
                "delimiter": self.jepo_config.delimiter,
                "format_penalty": self.jepo_config.format_penalty,
                "beta_supp": self.jepo_config.beta_supp,
                "beta_kl": self.jepo_config.beta_kl,
            },
            "use_jepo": True,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
            "jepo_prompt": prompt_str,
            "jepo_answer": ground_truth_answer if isinstance(ground_truth_answer, str) else str(ground_truth_answer),
        })
        
        # Call the JEPO-specific actor update with the properly formatted DataProto
        actor_output = self.actor_rollout_wg.jepo_update_actor(jepo_batch)
        
        # Extract metrics
        metrics = {}
        if "metrics" in actor_output.meta_info:
            actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            for key, value in actor_metrics.items():
                metrics[f"jepo_{key}"] = value
        
        return metrics
    
    def _group_by_uid_and_compute_metrics(self, batch, metric_name: str = "token_level_scores"):
        """
        Reusable UID grouping logic adapted from standard DAPO
        Groups data by UID and computes metrics for each group
        """
        from collections import defaultdict
        import numpy as np
        
        uids = batch.non_tensor_batch.get("uid", None)
        if uids is None:
            return {}, {}
            
        # Extract the specified metric (rewards by default)
        if metric_name in batch.batch:
            metric_data = batch.batch[metric_name]
            if metric_data.dim() > 1:  # Token-level rewards
                metric_vals = metric_data.sum(-1).cpu().numpy()
            else:
                metric_vals = metric_data.cpu().numpy()
        else:
            print(f"Warning: {metric_name} not found in batch")
            return {}, {}
        
        # Group metric values by UID (adapted from DAPO logic)
        uid_to_metric_vals = defaultdict(list)
        for uid, metric_val in zip(uids, metric_vals):
            uid_to_metric_vals[uid].append(metric_val)
        
        # Compute statistics for each UID group
        uid_to_stats = {}
        for uid, vals in uid_to_metric_vals.items():
            uid_to_stats[uid] = {
                'values': vals,
                'mean': np.mean(vals),
                'std': np.std(vals),
                'all_zero': np.all(np.array(vals) == 0),
                'all_nonzero': np.all(np.array(vals) != 0),
                'count': len(vals)
            }
        
        return uid_to_metric_vals, uid_to_stats
    
    def _extract_prompt_and_answer_from_uid_data(self, batch_data: Dict, non_tensor_data: Dict) -> tuple[str, str]:
        """Extract prompt and answer from UID-specific data"""
        try:
            # Try to get from non-tensor data first
            if 'prompts' in non_tensor_data and len(non_tensor_data['prompts']) > 0:
                prompt = non_tensor_data['prompts'][0]  # Take first prompt (should be same for all responses)
            else:
                prompt = "Unknown prompt"
                
            if 'answers' in non_tensor_data and len(non_tensor_data['answers']) > 0:
                answer = non_tensor_data['answers'][0]  # Take first answer (should be same for all responses)
            else:
                answer = "Unknown answer"
                
            return prompt, answer
        except Exception as e:
            print(f"Error extracting prompt/answer from UID data: {e}")
            return "Unknown prompt", "Unknown answer"

    def _run_jepo_training(self, all_incorrect_batch: DataProto) -> Dict[str, float]:
        """
        Run JEPO training on DataProto batch object
        
        Args:
            all_incorrect_batch: DataProto containing all incorrect responses
        
        Returns:
            Aggregated training metrics
        """

        batch_size = len(all_incorrect_batch.batch["prompts"])
        print(f"Running JEPO training with {batch_size} samples...")

        all_metrics = defaultdict(list)
        
        # Perform JEPO steps on the provided batch
        for step in range(self.jepo_config.jepo_steps):
            metrics = self._perform_jepo_training_step(all_incorrect_batch)
            
            # Store metrics for this step
            for key, value in metrics.items():
                all_metrics[f'{key}_step_{step}'].append(value)
                all_metrics[key].append(value)
        
        # Return averaged metrics
        final_metrics = {}
        for key, values in all_metrics.items():
            if values:
                final_metrics[key] = np.mean(values)
        
        return final_metrics
    
    def fit(self):
        """
        Enhanced DAPO training loop with JEPO integration
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_prompt_in_jepo_buffer = 0
        num_gen_batches = 0
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # Track whether we've added this generation batch to JEPO buffer
                    added_to_jepo_buffer = False
                    
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                        
                        # Add to JEPO buffer when filtering is disabled
                        if self.use_jepo and self.config.trainer.critic_warmup <= self.global_steps:
                            self._check_and_buffer_incorrect_responses(new_batch)
                            added_to_jepo_buffer = True
                    else:  # Filtering logic (same as original DAPO)
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        prompt_uid2metric_mean = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
                            prompt_uid2metric_mean[prompt_uid] = np.mean(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        all_incorrect_uids = [
                            uid
                            for uid, mean in prompt_uid2metric_mean.items()
                            if mean == 0 # Only works when we use acc as filter metrics. need to be change for other metrics.
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)
                        num_prompt_in_jepo_buffer += len(all_incorrect_uids)

                        kept_traj_idxs = []
                        all_incorrect_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                            if traj_from_prompt_uid in all_incorrect_uids:
                                all_incorrect_traj_idxs.append(idx)

                        # Add to JEPO buffer for each generation batch before continuing
                        if self.use_jepo and self.config.trainer.critic_warmup <= self.global_steps and not added_to_jepo_buffer:
                            print(f"Solve None: {len(all_incorrect_traj_idxs)}")
                            print(f"Solve Partial: {len(kept_traj_idxs)}")
                            all_incorrect_new_batch = new_batch[all_incorrect_traj_idxs]
                            all_incorrect_batch = all_incorrect_new_batch.copy() if all_incorrect_batch is None else DataProto.concat([all_incorrect_batch, all_incorrect_new_batch])
                            added_to_jepo_buffer = True
                            
                            # Perform JEPO training if buffer is full
                            if num_prompt_in_jepo_buffer >= self.jepo_config.buffer_size:
                                jepo_metrics = self._run_jepo_training(all_incorrect_batch[:self.jepo_config.buffer_size])
                                all_incorrect_batch = None # clear the jepo batch
                                num_prompt_in_jepo_buffer = 0
                                if jepo_metrics:
                                    print(f"JEPO training completed with metrics: {jepo_metrics}")

                        new_batch = new_batch[kept_traj_idxs]
                        #all_incorrect_batch = new_batch[all_incorrect_traj_idxs]

                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Standard DAPO Training ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup and standard actor update
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # === JEPO Buffer Management ===
                    # Check for all-incorrect UIDs and add to JEPO buffer for the final batch (if not already added)
                    if self.use_jepo and self.config.trainer.critic_warmup <= self.global_steps and not added_to_jepo_buffer:
                        self._check_and_buffer_incorrect_responses(batch)

                    # === JEPO Training Step ===
                    
                    # Perform JEPO training when buffer conditions are met
                    buffer_size_met = len(self.jepo_buffer.buffer) >= self.jepo_config.buffer_size
                    frequency_met = self.global_steps % self.jepo_update_frequency == 0
                    
                    print(f"JEPO Training Check - Step {self.global_steps}: Buffer={len(self.jepo_buffer.buffer)}/{self.jepo_config.buffer_size}, Buffer_size_met={buffer_size_met}, Frequency_met={frequency_met}")
                    
                    if (self.use_jepo and 
                        len(self.jepo_buffer.buffer) > 0 and 
                        (buffer_size_met or frequency_met)):
                        
                        print(f"🚀 STARTING JEPO TRAINING at step {self.global_steps} with {len(self.jepo_buffer.buffer)} buffered samples")
                        with marked_timer("jepo_training", timing_raw, "purple"):
                            jepo_metrics = self._run_jepo_training()
                            metrics.update(jepo_metrics)
                        print(f"✅ JEPO TRAINING COMPLETED - Metrics: {list(jepo_metrics.keys()) if jepo_metrics else 'None'}")

                    # validation
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1