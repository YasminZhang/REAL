set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_gemma_7b.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi


project_name='JEPO_token'
 
exp_name="Regression-warmup"
nproc_per_node=8
save_path="/blob/v-tianyuchen/Projects/jepo/ckpts/${project_name}/${exp_name}"

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    verl/trainer/fsdp_sft_trainer.py \
    data.train_files=/home/aiscuser/jepo/data/feedback_collection_for_base_warmup/train.parquet \
    data.val_files=/home/aiscuser/jepo/data/feedback_bench_for_base_warmup/train.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.max_length=3000 \
    data.micro_batch_size_per_gpu=16 \
    model.partial_pretrain="mistralai/Mistral-7B-Instruct-v0.2" \
    optim.lr=1e-5 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.total_epochs=5 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.logger='["console","wandb"]' $@