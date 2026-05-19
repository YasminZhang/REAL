#!/bin/bash
set -x

# Usage: bash generate_samples.sh <checkpoint_path>
# Example: bash generate_samples.sh /blob/v-tianyuchen/Projects/real/ckpts/real_token/Regression-warmup/global_step_100

if [ "$#" -lt 1 ]; then
    echo "Usage: generate_samples.sh <checkpoint_path>"
    echo "Example: generate_samples.sh ./Projects/real/ckpts/real_token/Regression-warmup/global_step_100"
    exit 1
fi

checkpoint_path=$1
nproc_per_node=8
output_dir="/home/aiscuser/real/outputs/generation_samples"
mkdir -p $output_dir

# Generate unique output filename with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
output_file="$output_dir/samples_$(basename $checkpoint_path)_$timestamp.parquet"

echo "Generating samples from checkpoint: $checkpoint_path"
echo "Output will be saved to: $output_file"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    verl/trainer/main_generation.py \
    data.path=/home/aiscuser/real/data/feedback_bench_for_base_warmup/train.parquet \
    data.prompt_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.n_samples=3 \
    data.output_path=$output_file \
    data.batch_size=32 \
    model.path=$checkpoint_path \
    rollout.name=vllm \
    rollout.temperature=0.7 \
    rollout.top_k=50 \
    rollout.top_p=0.9 \
    rollout.prompt_length=1536 \
    rollout.response_length=512 \
    rollout.dtype=bfloat16 \
    rollout.gpu_memory_utilization=0.5

echo "Generation completed. Results saved to: $output_file"