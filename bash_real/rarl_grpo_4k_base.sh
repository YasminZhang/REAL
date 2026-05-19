#!/usr/bin/env bash
set -xeuo pipefail

# add eval

project_name='JEPO_token'
#exp_name='deepscaler-1.5b-2k-format-test-g1-delimiter-token-math'
exp_name="Regression-Base-TRACT${1}"

adv_estimator=grpo

lr=1e-7
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
n_resp_per_prompt=4
train_prompt_mini_bsz=64

# DAPO
# don't do filter.
enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

# JEPO specific parameters
use_real=True
use_grpo=False
real_delimiter=" So the overall score is "
real_format_penalty=1

real_lr=1e-8
real_beta_supp=1
real_beta_kl=0.000
real_entropy_coeff=0.000
real_use_format_adv=False


real_buffer_size=${train_prompt_bsz} # number of questions
real_steps=1
real_update_frequency=100000
real_epochs=3
real_use_dynamic_bsz=True
real_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 8))
real_mini_batch_size_per_gpu=128 # responses per gpu
real_micro_batch_size_per_gpu=64 # responses per gpu

real_responses_micro_batch_size=1024 # this param will be ignored
real_accum_steps=1 # this is also ignored

# Ray - single node setup for 1.5B
NNODES=1
NGPUS_PER_NODE=8

# Use 1.5B model
# MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_PATH="/blob/v-tianyuchen/Projects/jepo/ckpts/JEPO_token/Regression-warmup/global_step_100_hf"
MODEL_PATH="yasiz/Mistral-7b-v0.2-Instruct-TRACT-copy"
CKPTS_DIR="/blob/v-tianyuchen/Projects/jepo/ckpts/${project_name}/${exp_name}"
TRAIN_FILE=data/feedback_collection_for_base/train_sample.parquet
TEST_FILE=data/feedback_bench_for_base/train.parquet


# Algorithm
temperature=1
top_p=0.9
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.9
repetition_penalty=1.0

# Performance Related Parameter - adjusted for 1.5B model
sp_size=1  # Single sequence parallel for smaller model
use_dynamic_bsz=True
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 8))
offload=True  # Keep offload for memory efficiency
gen_tp=1  # Single tensor parallel for 1.5B
fsdp_size=-1  # Auto FSDP size

extra_val_files=\"tract_data/feedback_collection_ood_for_base/train.parquet,tract_data/flask_eval_for_base/train.parquet,tract_data/mt_bench_for_base/train.parquet,tract_data/vicuna_eval_for_base/train.parquet\"

# Use JEPO-DAPO recipe
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m recipe.dapo.main_real_dapo \
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
    algorithm.use_real=${use_real} \
    algorithm.real_delimiter="${real_delimiter}" \
    algorithm.real_format_penalty=${real_format_penalty} \
    algorithm.real_beta_supp=${real_beta_supp} \
    algorithm.real_beta_kl=${real_beta_kl} \
    algorithm.real_buffer_size=${real_buffer_size} \
    algorithm.real_steps=${real_steps} \
    algorithm.real_update_frequency=${real_update_frequency} \
    +algorithm.real_config.use_regression_reward=True \
    +algorithm.real_config.use_last_token_as_answer=True \
    +algorithm.real_config.answer_token_length=1 \
    +algorithm.real_config.store_last_token_probs=True \
    +algorithm.real_config.use_format_adv=${real_use_format_adv} \
    +algorithm.real_epochs=${real_epochs} \
    +algorithm.real_mini_batch_size_per_gpu=${real_mini_batch_size_per_gpu} \
    +algorithm.real_micro_batch_size_per_gpu=${real_micro_batch_size_per_gpu} \
    +algorithm.real_responses_micro_batch_size=${real_responses_micro_batch_size} \
    +algorithm.real_delimiter_suffix_anchor=False \
    +algorithm.real_delimiter_suffix_min_len=2 \
    +algorithm.real_accum_steps=${real_accum_steps} \
    +algorithm.real_loss_agg_mode=${loss_agg_mode} \
    +algorithm.real_entropy_coeff=${real_entropy_coeff} \
    +algorithm.real_use_dynamic_bsz=${real_use_dynamic_bsz} \
    +algorithm.real_use_dynamic_balancer=False \
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
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    +algorithm.real_ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=64.0 \
    actor_rollout_ref.rollout.load_format="safetensors" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.real_actor.optim.lr=${real_lr} \
    +actor_rollout_ref.real_actor.optim.warmup_style=constant \
    +actor_rollout_ref.real_actor.optim.warmup_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
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
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.total_epochs=500 \
    trainer.total_training_steps=5000 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=5 \
    custom_reward_function.path="recipe/dapo/deepscaler_reward.py" \
    custom_reward_function.name=deepscaler_reward_fn
