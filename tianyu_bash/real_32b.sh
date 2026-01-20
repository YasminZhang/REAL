#!/usr/bin/env bash
set -xeuo pipefail

# add eval

project_name='REAL_32B'
#exp_name='deepscaler-1.5b-2k-format-test-g1-delimiter-token-math'
exp_name="32B_r64_alpha128_5e-6"

adv_estimator=grpo

lr=5e-6
use_kl_in_reward=False
kl_coef=0.01
use_kl_loss=True
kl_loss_coef=0.01
clip_ratio_low=0.2
clip_ratio_high=0.2




max_prompt_length=2048
max_response_length=1024
enable_overlong_buffer=False
overlong_buffer_len=1024
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Adjusted for 1.5B model - smaller batch sizes
train_prompt_bsz=256

train_prompt_mini_bsz=64

# DAPO
# don't do filter.
enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

# JEPO specific parameters
use_jepo=True
use_grpo=False
jepo_delimiter=" So the overall score is " # no use
jepo_format_penalty=1

#######################################################################
n_resp_per_prompt=32
jepo_lr=5e-8 # Qwen -> 1e-6, 5e-7, Mistral -> 5e-8, lora = full-finetuning * 10 
jepo_beta_supp=1.0 # lambda
jepo_beta_supp_extra=0.000 # beta
jepo_beta_kl=0.000
jepo_entropy_coeff=0.000
jepo_use_format_adv=False

jepo_use_extra_loss=False
jepo_use_log_prob_loss=True
jepo_use_l2_loss=True

jepo_normalize_advantages=True # keep it as it is
jepo_use_cot_loss=True
jepo_data_type="partial" # partial, all, incorrect, partial_incorrect, partial_correct
jepo_use_prob_as_reward=True # keep it as it is
jepo_use_rloo=False # if True, please set use_extra_loss to False
jepo_update_freq=10 # Qwen -> 20
##########################################################################

jepo_buffer_size=${train_prompt_bsz} # number of questions
jepo_steps=1
jepo_update_frequency=100000
jepo_epochs=3
jepo_use_dynamic_bsz=True
jepo_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
jepo_mini_batch_size_per_gpu=128 # responses per gpu
jepo_micro_batch_size_per_gpu=64 # responses per gpu

jepo_responses_micro_batch_size=1024 # this param will be ignored
jepo_accum_steps=1 # this is also ignored

# Ray - single node setup for 1.5B
NNODES=2
NGPUS_PER_NODE=8

# Use 1.5B model
# MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_PATH="/blob/v-tianyuchen/Projects/jepo/ckpts/JEPO_token/Regression-warmup/global_step_100_hf"
# MODEL_PATH="yasiz/Mistral-7b-v0.2-Instruct-TRACT-copy"
# MODEL_PATH="/blob/v-tianyuchen/Projects/jepo/ckpts/JEPO_token_Dec9/Regression-warmup-Qwen3-1.7B/global_step_1170/huggingface"
# MODEL_PATH="Qwen/Qwen3-1.7B"
# MODEL_PATH="/blob/v-tianyuchen/Projects/jepo/ckpts/JEPO_token_Dec13/Qwen3-1.7B-RAFT/global_step_100/huggingface"
# MODEL_PATH="/blob/v-tianyuchen/Projects/jepo/ckpts/JEPO_token_Dec13/Qwen3-8B-RAFT_epoch5/global_step_1950/huggingface"
# MODEL_PATH="/blob/v-tianyuchen/Projects/jepo/ckpts/JEPO_token_Dec13/Qwen3-8B-RAFT_epoch5_stage2/global_step_1950/huggingface"
# MODEL_PATH="yasiz/Llama-3.1-8B-Instruct-TRACT-copy"
# MODEL_PATH=" /blob/v-tianyuchen/Projects/jepo/ckpts/JEPO_token_Dec13/Qwen3-8B-RAFT_epoch5/global_step_100/huggingface"
MODEL_PATH="Qwen/Qwen3-32B"
CKPTS_DIR="/blob/v-tianyuchen/Projects/jepo/ckpts/${project_name}/${exp_name}"
TRAIN_FILE=/blob/v-tianyuchen/Projects/jepo/jepo_dataset/train.parquet
TEST_FILE=/blob/v-tianyuchen/Projects/jepo/jepo_dataset/feedback_ood_test/test.parquet


# Algorithm
temperature=1
top_p=0.9
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.9
repetition_penalty=1.0

# Performance Related Parameter - adjusted for 1.5B model
sp_size=1  # Single sequence parallel for smaller model
use_dynamic_bsz=True
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
offload=True  # Keep offload for memory efficiency
gen_tp=2  # Single tensor parallel for 1.5B
fsdp_size=-1  # Auto FSDP size

extra_val_files=\"/blob/v-tianyuchen/Projects/jepo/jepo_dataset/feedback_ood_test/test.parquet,/blob/v-tianyuchen/Projects/jepo/jepo_dataset/flask/test.parquet,/blob/v-tianyuchen/Projects/jepo/jepo_dataset/mt_bench/test.parquet,/blob/v-tianyuchen/Projects/jepo/jepo_dataset/vicuna/test.parquet\"

# Use JEPO-DAPO recipe
python3 -m recipe.dapo.main_jepo_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    +data.extra_val_files="${extra_val_files}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.use_jepo=${use_jepo} \
    algorithm.jepo_delimiter="${jepo_delimiter}" \
    algorithm.jepo_format_penalty=${jepo_format_penalty} \
    algorithm.jepo_beta_supp=${jepo_beta_supp} \
    +algorithm.jepo_beta_supp_extra=${jepo_beta_supp_extra} \
    algorithm.jepo_beta_kl=${jepo_beta_kl} \
    algorithm.jepo_buffer_size=${jepo_buffer_size} \
    algorithm.jepo_steps=${jepo_steps} \
    algorithm.jepo_update_frequency=${jepo_update_frequency} \
    +algorithm.jepo_use_regression_reward=True \
    +algorithm.jepo_use_last_token_as_answer=True \
    +algorithm.jepo_answer_token_length=1 \
    +algorithm.jepo_store_last_token_probs=True \
    +algorithm.jepo_use_format_adv=${jepo_use_format_adv} \
    +algorithm.jepo_use_log_prob_loss=${jepo_use_log_prob_loss} \
    +algorithm.jepo_use_extra_loss=${jepo_use_extra_loss} \
    +algorithm.jepo_use_cot_loss=${jepo_use_cot_loss} \
    +algorithm.jepo_normalize_advantages=${jepo_normalize_advantages} \
    +algorithm.jepo_use_l2_loss=${jepo_use_l2_loss} \
    +algorithm.jepo_use_prob_as_reward=${jepo_use_prob_as_reward} \
    +algorithm.jepo_use_rloo=${jepo_use_rloo} \
    +algorithm.model_name="${MODEL_PATH}" \
    +algorithm.jepo_data_type=${jepo_data_type} \
    +algorithm.jepo_epochs=${jepo_epochs} \
    +algorithm.jepo_mini_batch_size_per_gpu=${jepo_mini_batch_size_per_gpu} \
    +algorithm.jepo_micro_batch_size_per_gpu=${jepo_micro_batch_size_per_gpu} \
    +algorithm.jepo_responses_micro_batch_size=${jepo_responses_micro_batch_size} \
    +algorithm.jepo_delimiter_suffix_anchor=False \
    +algorithm.jepo_delimiter_suffix_min_len=2 \
    +algorithm.jepo_accum_steps=${jepo_accum_steps} \
    +algorithm.jepo_loss_agg_mode=${loss_agg_mode} \
    +algorithm.jepo_entropy_coeff=${jepo_entropy_coeff} \
    +algorithm.jepo_use_dynamic_bsz=${jepo_use_dynamic_bsz} \
    +algorithm.jepo_use_dynamic_balancer=False \
    +algorithm.use_grpo=${use_grpo} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.actor.model_name="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.layered_summon=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.model_name="${MODEL_PATH}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    +algorithm.jepo_ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.load_format="safetensors" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.jepo_actor.optim.lr=${jepo_lr} \
    +actor_rollout_ref.jepo_actor.optim.warmup_style=constant \
    +actor_rollout_ref.jepo_actor.optim.warmup_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.repetition_penalty=${repetition_penalty} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=${jepo_update_freq} \
    trainer.save_freq=${jepo_update_freq} \
    trainer.total_epochs=500 \
    trainer.total_training_steps=5000 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.validation_data_dir="${CKPTS_DIR}/validations" \
    custom_reward_function.path="recipe/dapo/deepscaler_reward.py" \
    custom_reward_function.name=deepscaler_reward_fn \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128.0 \
