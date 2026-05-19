#!/usr/bin/env bash
# Launches a REAL RL training run on a single 8-GPU node.
# Usage: bash run_real.sh <experiment_name>
#   <experiment_name> is used for the wandb run name and the checkpoint subdir.
set -xeuo pipefail

# ---------- Experiment identity ----------
project_name='REAL_Ablation'
exp_name="${1}"

# ---------- Core RL algorithm (no use for real training) ----------
# GRPO-style advantage estimator; KL is applied as a loss term, not in the reward.
adv_estimator=grpo
lr=1e-7
use_kl_in_reward=False
kl_coef=0.01
use_kl_loss=True
kl_loss_coef=0.01
clip_ratio_low=0.2
clip_ratio_high=0.2

# ---------- Sequence lengths ----------
max_prompt_length=2048
max_response_length=1024
# Overlong-buffer penalty is off; responses past max_response_length are simply truncated.
enable_overlong_buffer=False
overlong_buffer_len=1024
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"


# ---------- DAPO filter groups ----------
# Disabled here: we do NOT filter generation batches by accuracy.
enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

# ---------- REAL mode ----------
use_real=True # don't change
use_grpo=False # don't change
real_delimiter=" So the overall score is " # no use
real_format_penalty=1 # no use

#############################################################################################
# ---------- REAL hyperparameters ----------
# lr scale rule of thumb: Qwen ~1e-6; Mistral ~5e-8; LoRA = ~10x full-finetune lr.
#############################################################################################

# ---------- Batch sizes (sized for a 1.5B–7B model on 8 GPUs) ----------
train_prompt_bsz=256
train_prompt_mini_bsz=32 # could be 64.

# ---------- REAL hyperparameters (important for training)----------
n_resp_per_prompt=16 # could be 8 for a quick test
real_lr=5e-8                #Qwen -> 1e-6, Mistral -> 5e-8, lora = full-finetuning * 10 
real_beta_supp=1.0          # lambda (support loss weight), recommend to set to 1 for better performance
real_beta_supp_extra=0.00  # beta (extra support loss weight), recommend to set to 0 for faster and reasonable performance
real_beta_kl=0.000
real_entropy_coeff=0.000
real_update_freq=10             # eval/save cadence; Qwen runs typically use 20
val_before_train=True          # whether to run validation before training
real_use_rloo=False             # if you want to try RLOO baselines, set to True

# default settings for real training
real_normalize_advantages=True
real_use_cot_loss=True
real_data_type="partial"        # one of: partial, all, incorrect, partial_incorrect, partial_correct

# some hard-coded flags for ablations
real_use_prob_as_reward=True
real_use_l2_loss=True
real_use_log_prob_loss=True
real_use_format_adv=False
real_use_extra_loss=True   # if real_use_rloo=True, keep this False


# ---------- REAL buffer / batching ----------
# Dynamic batching: the per-GPU "responses" counts below cap micro-batches by token budget.
real_buffer_size=${train_prompt_bsz}     # number of questions
real_steps=1
real_update_frequency=100000
real_epochs=3
real_use_dynamic_bsz=True
real_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 8))   # unused — kept for parity
real_mini_batch_size_per_gpu=128         # responses per gpu
real_micro_batch_size_per_gpu=64         # responses per gpu
real_responses_micro_batch_size=1024     # ignored at runtime
real_accum_steps=1                       # ignored at runtime

# ---------- Distributed setup ----------
NNODES=1
NGPUS_PER_NODE=8

# ---------- Model & data paths ----------
# NOTE: MODEL_PATH below is the downloaded REAL FSDP checkpoint. vLLM/HF
# from_pretrained will not load it directly — run the verl FSDP -> HF
# conversion first if you haven't already.
# Paths are relative to the jepo project root — run this script from there:
#   bash bash_real/run_real.sh <exp_name>
DATA_DIR=./data/real_data/real_dataset
MODEL_PATH=./ckpts/Mistral-7b-v0.2-Instruct-TRACT-copy
CKPTS_DIR=./outputs/${project_name}/${exp_name}
TRAIN_FILE=${DATA_DIR}/train.parquet
TEST_FILE=${DATA_DIR}/feedback_ood_test/test.parquet
# Extra eval sets reported alongside the primary TEST_FILE.
extra_val_files=\"${DATA_DIR}/feedback_ood_test/test.parquet,${DATA_DIR}/flask/test.parquet,${DATA_DIR}/mt_bench/test.parquet,${DATA_DIR}/vicuna/test.parquet\"

# ---------- Generation (vLLM rollout) ----------
temperature=1
top_p=0.9
top_k=-1                # 0 for HF rollout, -1 for vLLM
val_top_p=0.9
repetition_penalty=1.0

# ---------- Performance / parallelism ----------
sp_size=1               # Ulysses sequence parallel (1 = off)
use_dynamic_bsz=True
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 8))
offload=True            # FSDP param/optimizer offload to CPU for memory headroom
gen_tp=1                # vLLM tensor parallel per replica
fsdp_size=-1            # auto

# ---------- Launch ----------
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
    +algorithm.real_beta_supp_extra=${real_beta_supp_extra} \
    algorithm.real_beta_kl=${real_beta_kl} \
    algorithm.real_buffer_size=${real_buffer_size} \
    algorithm.real_steps=${real_steps} \
    algorithm.real_update_frequency=${real_update_frequency} \
    +algorithm.real_use_regression_reward=True \
    +algorithm.real_use_last_token_as_answer=True \
    +algorithm.real_answer_token_length=1 \
    +algorithm.real_store_last_token_probs=True \
    +algorithm.real_use_format_adv=${real_use_format_adv} \
    +algorithm.real_use_log_prob_loss=${real_use_log_prob_loss} \
    +algorithm.real_use_extra_loss=${real_use_extra_loss} \
    +algorithm.real_use_cot_loss=${real_use_cot_loss} \
    +algorithm.real_normalize_advantages=${real_normalize_advantages} \
    +algorithm.real_use_l2_loss=${real_use_l2_loss} \
    +algorithm.real_use_prob_as_reward=${real_use_prob_as_reward} \
    +algorithm.real_use_rloo=${real_use_rloo} \
    +algorithm.model_name="${MODEL_PATH}" \
    +algorithm.real_data_type=${real_data_type} \
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
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.actor.model_name="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.model_name="${MODEL_PATH}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    +algorithm.real_ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
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
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${real_update_freq} \
    trainer.save_freq=${real_update_freq} \
    trainer.total_epochs=500 \
    trainer.total_training_steps=5000 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.validation_data_dir="${CKPTS_DIR}/validations" \
    custom_reward_function.path="recipe/dapo/deepscaler_reward.py" \
    custom_reward_function.name=deepscaler_reward_fn
