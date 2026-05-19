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
REAL-enhanced DAPO Trainer that integrates REAL algorithm with DAPO workflow
"""

import os
# Import REAL components
import sys
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           reduce_metrics)
from verl.trainer.ppo.ray_trainer import (AdvantageEstimator, Role,
                                          apply_kl_penalty, compute_advantage,
                                          compute_response_mask)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'real'))

from real_core_algos import REALConfig


class RayREALDAPOTrainer(RayDAPOTrainer):
    """
    REAL-enhanced DAPO Trainer that integrates REAL algorithm with standard DAPO training
    
    Workflow:
    1. Perform standard DAPO training (GRPO with advantage estimation)
    2. Track responses where all answers are incorrect (reward = 0)
    3. When REAL buffer reaches threshold, perform REAL training steps
    4. REAL training replaces/augments the standard actor update for incorrect response batches
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize parent DAPO trainer without modifying workers
        # REAL logic will be handled in the trainer methods, not by replacing workers
        super().__init__(*args, **kwargs)
        
        print(f"REAL-DAPO Trainer initialized with actor class: {self.role_worker_mapping.get(Role.ActorRollout, 'Unknown')}")
        
        # Initialize REAL components
        self.real_config = REALConfig(
            delimiter=getattr(self.config.algorithm, 'real_delimiter', '\n\n'),
            format_penalty=getattr(self.config.algorithm, 'real_format_penalty', 0.1),
            beta_supp=getattr(self.config.algorithm, 'real_beta_supp', 1.0),
            beta_kl=getattr(self.config.algorithm, 'real_beta_kl', 0.1),
            buffer_size=getattr(self.config.algorithm, 'real_buffer_size', 100),
            real_steps=getattr(self.config.algorithm, 'real_steps', 5),
            epochs=getattr(self.config.algorithm, 'real_epochs', 1),
            mini_batch_size_per_gpu=getattr(self.config.algorithm, 'real_mini_batch_size_per_gpu', 8),
            micro_batch_size_per_gpu=getattr(self.config.algorithm, 'real_micro_batch_size_per_gpu', 1),
            responses_micro_batch_size=getattr(self.config.algorithm, 'real_responses_micro_batch_size', 8),
            num_response_per_question=getattr(self.config.actor_rollout_ref.rollout, 'n', 1),
            accum_steps=getattr(self.config.algorithm, 'real_accum_steps', 4),
            data_type=getattr(self.config.algorithm, 'real_data_type', 'partial_incorrect'), # partial, all, incorrect, partial_incorrect,
        )
        
        self.real_metrics = defaultdict(list)
        
        # Enable REAL mode
        self.use_real = getattr(self.config.algorithm, 'use_real', True)
        self.real_update_frequency = getattr(self.config.algorithm, 'real_update_frequency', 10)
        
        print(f"REAL-DAPO Trainer initialized with buffer_size={self.real_config.buffer_size}, use_real={self.use_real}")
    
    def _perform_real_training_step(self, real_batch: DataProto) -> Dict[str, float]:
        """
        Perform one REAL training step using the worker group
        
        Args:
            real_batch: DataProto containing buffered data with all-incorrect responses
                      - prompts are under batch.batch["prompts"] (tokens)
                      - responses are under batch.batch["responses"] (tokens)  
                      - ground_truth_answer is under batch.non_tensor_batch["reward_model"]["ground_truth"] (str)
            
        Returns:
            Dictionary of training metrics
        """
        # Add full REAL config to the batch meta_info so REALActor can consume it
        # Note: keep key names in sync with verl/workers/actor/real_actor.py
        real_batch.meta_info["real_config"] = {
            "delimiter": self.real_config.delimiter,
            "format_penalty": self.real_config.format_penalty,
            "beta_supp": self.real_config.beta_supp,
            "beta_supp_extra": getattr(self.config.algorithm, 'real_beta_supp_extra', 0.001),
            "beta_kl": self.real_config.beta_kl,
            # training/loop settings read by REALActor
            "epochs": self.real_config.epochs,
            "mini_batch_size_per_gpu": self.real_config.mini_batch_size_per_gpu,
            "micro_batch_size_per_gpu": self.real_config.micro_batch_size_per_gpu,
            "responses_micro_batch_size": self.real_config.responses_micro_batch_size,
            # token cap per GPU used by REALActor when use_dynamic_bsz
            "ppo_max_token_len_per_gpu": getattr(
                self.config.algorithm,
                'real_ppo_max_token_len_per_gpu',
                getattr(self.config.actor_rollout_ref.actor, 'ppo_max_token_len_per_gpu', 16384),
            ),
            "accum_steps": self.real_config.accum_steps,
            "num_response_per_question": self.real_config.num_response_per_question,
            # REAL-specific knobs to keep REALActor self-contained
            "entropy_coeff": getattr(self.config.algorithm, 'real_entropy_coeff', getattr(self.config.actor_rollout_ref.actor, 'entropy_coeff', 0.0)),
            "loss_agg_mode": getattr(self.config.algorithm, 'real_loss_agg_mode', getattr(self.config.actor_rollout_ref.actor, 'loss_agg_mode', 'token-mean')),
            "use_dynamic_bsz": getattr(self.config.algorithm, 'real_use_dynamic_bsz', getattr(self.config.actor_rollout_ref.actor, 'use_dynamic_bsz', True)),
            "use_dynamic_balancer": getattr(self.config.algorithm, 'real_use_dynamic_balancer', False),
            # Debug hooks
            "dummy_forward_rank": getattr(self.config.algorithm, 'real_dummy_forward_rank', -1),
            "dummy_forward_value": getattr(self.config.algorithm, 'real_dummy_forward_value', 0.0),
            "dummy_forward_min_seq": getattr(self.config.algorithm, 'real_dummy_forward_min_seq', 0),
            # Progress bar output options
            "show_all_rank_pbar_to_file": getattr(self.config.algorithm, 'real_show_all_rank_pbar_to_file', False),
            "pbar_file_dir": getattr(self.config.algorithm, 'real_pbar_file_dir', 'user_logs'),
            # Suffix-anchor delimiter matching config (optional)
            "delimiter_suffix_anchor": getattr(self.config.algorithm, 'real_delimiter_suffix_anchor', True),
            "delimiter_suffix_min_len": getattr(self.config.algorithm, 'real_delimiter_suffix_min_len', 2),
            # add more REAL-specific config here as needed
            "use_regression_reward": getattr(self.config.algorithm, 'real_use_regression_reward', False),
            "use_last_token_as_answer": getattr(self.config.algorithm, 'real_use_last_token_as_answer', True),
            "answer_token_length": getattr(self.config.algorithm, 'real_answer_token_length', 1),
            "store_last_token_probs": getattr(self.config.algorithm, 'real_store_last_token_probs', True),
            "use_format_adv": getattr(self.config.algorithm, 'real_use_format_adv', False),
            "use_log_prob_loss": getattr(self.config.algorithm, 'real_use_log_prob_loss', False),
            "use_extra_loss": getattr(self.config.algorithm, 'real_use_extra_loss', False),
            "use_cot_loss": getattr(self.config.algorithm, 'real_use_cot_loss', False),
            "normalize_advantages": getattr(self.config.algorithm, 'real_normalize_advantages', False),
            "use_l2_loss": getattr(self.config.algorithm, 'real_use_l2_loss', False),
            "data_type": getattr(self.config.algorithm, 'real_data_type', 'partial_incorrect'), # partial, all, incorrect, partial_incorrect
            "use_prob_as_reward": getattr(self.config.algorithm, 'real_use_prob_as_reward', False),
            "model_name": getattr(self.config.algorithm, 'model_name', 'unknown_model'),
            "use_rloo": getattr(self.config.algorithm, 'real_use_rloo', False),
        }
        
        # Call the REAL-specific actor update with the properly formatted DataProto
        actor_output = self.actor_rollout_wg.real_update_actor(real_batch)
        
        # Extract metrics
        metrics = {}
        if "metrics" in actor_output.meta_info:
            actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            for key, value in actor_metrics.items():
                metrics[f"real_{key}"] = value
        
        return metrics


    def _run_real_training(self, all_incorrect_batch: DataProto) -> Dict[str, float]:
        """
        Run REAL training on DataProto batch object
        
        Args:
            all_incorrect_batch: DataProto containing all incorrect responses
        
        Returns:
            Aggregated training metrics
        """

        batch_size = len(all_incorrect_batch.batch["prompts"])
        print(f"Running REAL training with {batch_size} samples...")
        
        all_metrics = defaultdict(list)
        
        # Perform REAL steps on the provided batch
        for step in range(self.real_config.real_steps):
            # Process batch in micro batches to avoid OOM
            step_metrics = defaultdict(list)
            
            metrics = self._perform_real_training_step(all_incorrect_batch)
                
            # Collect metrics from this micro batch
            for key, value in metrics.items():
                step_metrics[key].append(value)
            
            # Average metrics across all micro batches for this step
            for key, values in step_metrics.items():
                if values:
                    step_avg = np.mean(values)
                    all_metrics[f'{key}_step_{step}'].append(step_avg)
                    all_metrics[key].append(step_avg)
        
        # Return averaged metrics
        final_metrics = {}
        for key, values in all_metrics.items():
            if values:
                final_metrics[key] = np.mean(values)
        
        return final_metrics
    
    def fit(self):
        """
        Enhanced DAPO training loop with REAL integration
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
        num_prompt_in_real_buffer = 0
        num_gen_batches = 0
        all_incorrect_batch = None  # Buffer for all incorrect responses
        
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

                    # Collect the sequence reward for each trajectory
                    metric_name = self.config.algorithm.filter_groups.metric
                    if metric_name == "seq_final_reward":
                        new_batch.non_tensor_batch["seq_final_reward"] = (
                            new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        )
                    elif metric_name == "seq_reward":
                        new_batch.non_tensor_batch["seq_reward"] = (
                            new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                        )
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
                    # Prompts where all responses are incorrect (acc == 0 across n samples)
                    # NOTE: For accuracy-style metrics in {0,1}, the “all incorrect” case means mean == 0.
                    # Using ">= 0" incorrectly included almost all prompts and diluted REAL training.
                    if self.config.algorithm.use_grpo: # if use grpo, then use all incorrect question for real
                        all_incorrect_uids = [
                            uid for uid, mean in prompt_uid2metric_mean.items() if np.isclose(mean, 0.0)
                        ]
                    else: # if there is no grpo, then only use real with all questions
                        all_incorrect_uids = [
                            uid for uid, mean in prompt_uid2metric_mean.items() if mean >= 0
                        ]
                    # Prompts where all responses are correct (acc == 1 across n samples)
                    all_correct_uids = [
                        uid for uid, mean in prompt_uid2metric_mean.items() if np.isclose(mean, 1.0)
                    ]
                    # Partial solves: neither all-correct nor all-incorrect
                    all_prompt_uids = list(prompt_uid2metric_mean.keys())
                    partial_uids = [
                        uid for uid in all_prompt_uids if uid not in all_correct_uids and uid not in all_incorrect_uids
                    ]

                    all_incorrect_uids_ = [
                            uid for uid, mean in prompt_uid2metric_mean.items() if np.isclose(mean, 0.0)
                        ]
                    partial_uids_ = [
                        uid for uid in all_prompt_uids if uid not in all_correct_uids and uid not in all_incorrect_uids_
                    ]
                    
                    num_prompt_in_batch += len(kept_prompt_uids)
                    
                    # Log solve stats for this gen batch
                    metrics["real_buffer/solve_all"] = len(all_correct_uids)
                    metrics["real_buffer/solve_none"] = len(all_incorrect_uids_)
                    metrics["real_buffer/solve_partial"] = len(partial_uids_)
                    metrics["real_buffer/total_prompts"] = len(all_prompt_uids)

                    kept_traj_idxs = []
                    all_incorrect_traj_idxs = []
                    all_partial_traj_idxs = []
                    all_correct_traj_idxs = []
                    
                    for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                        if traj_from_prompt_uid in kept_prompt_uids:
                            kept_traj_idxs.append(idx)
                        if traj_from_prompt_uid in all_incorrect_uids_:
                            all_incorrect_traj_idxs.append(idx)
                        if traj_from_prompt_uid in partial_uids_:
                            all_partial_traj_idxs.append(idx)
                        if traj_from_prompt_uid in all_correct_uids:
                            all_correct_traj_idxs.append(idx)
                        
                            
                

                    # Add to REAL buffer for each generation batch before continuing
                    if self.use_real:
                        print(f"Solve None: {len(all_incorrect_uids_)}")
                        print(f"Solve Partial: {len(partial_uids_)}")
                        print(f"Solve All: {len(all_correct_uids)}")
                        
                        print('data_type:', self.real_config.data_type)
                        
                        if self.real_config.data_type == "partial_incorrect":
                            all_incorrect_new_batch = new_batch[all_incorrect_traj_idxs + all_partial_traj_idxs]
                            num_prompt_in_real_buffer += len(all_incorrect_uids_) + len(partial_uids_)
                        elif self.real_config.data_type == "all":
                            all_incorrect_new_batch = new_batch
                            num_prompt_in_real_buffer += len(all_prompt_uids)
                        elif self.real_config.data_type == "incorrect":
                            all_incorrect_new_batch = new_batch[all_incorrect_traj_idxs]
                            num_prompt_in_real_buffer += len(all_incorrect_uids_)
                        elif self.real_config.data_type == "partial":
                            all_incorrect_new_batch = new_batch[all_partial_traj_idxs]
                            num_prompt_in_real_buffer += len(partial_uids_)
                        elif self.real_config.data_type == "partial_correct":
                            all_incorrect_new_batch = new_batch[all_partial_traj_idxs + all_correct_traj_idxs]
                            num_prompt_in_real_buffer += len(partial_uids_) + len(all_correct_uids)
                        else:
                            raise ValueError(f"Unknown real_data_type: {self.real_config.data_type}")
                        
                    
                        
                        
                        print(f"Total prompts in real buffer: {num_prompt_in_real_buffer}")
                        
                         
                        all_incorrect_batch = deepcopy(all_incorrect_new_batch) if all_incorrect_batch is None else DataProto.concat([all_incorrect_batch, all_incorrect_new_batch])
                        
                        

                        # Perform REAL training if buffer is full
                        if num_prompt_in_real_buffer >= self.real_config.buffer_size:
                            # Truncate all_incorrect_batch to be divisible by dp_size before any operations
                            dp_size = self.actor_rollout_wg.world_size
                            current_size = len(all_incorrect_batch)
                            remainder = current_size % dp_size
                            if remainder != 0:
                                truncate_size = current_size - remainder
                                print(f"[REAL Buffer Truncation] Original size: {current_size}, Truncated to: {truncate_size}, Remainder removed: {remainder}, dp_size: {dp_size}")
                                all_incorrect_batch = all_incorrect_batch[:truncate_size]
                            else:
                                print(f"[REAL Buffer Truncation] No truncation needed. Batch size {current_size} is divisible by dp_size={dp_size}")
                            
                            if self.use_reference_policy:
                                with marked_timer("ref", timing_raw, "olive"):
                                    if not self.ref_in_actor:
                                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(all_incorrect_batch)
                                    else:
                                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(all_incorrect_batch)
                                    all_incorrect_batch = all_incorrect_batch.union(ref_log_prob)
                            real_metrics = self._run_real_training(all_incorrect_batch[:self.real_config.buffer_size*self.config.actor_rollout_ref.rollout.n])
                            metrics.update(real_metrics)
                            print(f"✅ REAL TRAINING COMPLETED - Metrics: {list(real_metrics.keys()) if real_metrics else 'None'}")
                            all_incorrect_batch = None # clear the real batch
                            num_prompt_in_real_buffer = 0
                            if real_metrics:
                                print(f"REAL training completed with metrics: {real_metrics}")
                    
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # Filtering logic (same as original DAPO)
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
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
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
                    if self.config.algorithm.use_grpo:
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with marked_timer("update_actor", timing_raw, "red"):
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                    # TODO: Skip this part for now
                    # # Perform REAL training when frequency is hit
                    # frequency_met = self.global_steps % self.real_update_frequency == 0
                    # if (self.use_real and all_incorrect_batch is not None and frequency_met):
                    #     real_metrics = self._run_real_training(all_incorrect_batch)
                    #     metrics.update(real_metrics)
                    #     print(f"✅ REAL TRAINING COMPLETED - Metrics: {list(real_metrics.keys()) if real_metrics else 'None'}")
                    #     all_incorrect_batch = None # clear the real batch
                    #     num_prompt_in_real_buffer = 0
                    #     if real_metrics:
                    #         print(f"REAL training completed with metrics: {real_metrics}")

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
