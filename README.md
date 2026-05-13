# REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge

REAL is a principled reinforcement learning framework designed to optimize **regression rewards** for LLM-as-a-Judge tasks. Unlike standard RL methods that rely on binary rewards (e.g., 0-1 accuracy), REAL explicitly models the **ordinal structure** inherent in numeric scoring, recognizing that predicting 4 is significantly better than predicting 1 when the ground truth is 5.

**Paper**: [REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge](https://arxiv.org/abs/2603.17145) (ICML 2026)

**Authors**: Yasi Zhang, Tianyu Chen, Mingyuan Zhou, Oscar Leong, Ying Nian Wu, Michal Lukasik

## Key Features

- **Regression-Aware Reward**: Optimizes a policy-dependent regression loss that captures the ordinal structure of evaluation scores, proven to be optimal for correlation metrics (Pearson, Spearman)
- **Generalized Policy Gradient**: Handles policy-dependent rewards via a generalized gradient estimator that naturally decomposes into two complementary components:
  1. **CoT Exploration** — policy gradient over Chain-of-Thought trajectories weighted by regression-aware rewards
  2. **Prediction Refinement** — regression-aware supervision on the final numeric score via standard backpropagation
- **RLOO Stabilization**: Uses the leave-one-out baseline for variance reduction without requiring a learned value function
- **RAIL Inference**: Computes expected value over digit tokens for regression-aware predictions at inference time
- **Multi-scale Support**: Validated across 8B and 32B model scales with ready-to-use training scripts
- **Built on verl**: Leverages the [verl](https://github.com/volcengine/verl) framework for efficient FSDP/vLLM integration and Ray distributed training

## Main Results

On Qwen3-32B, REAL achieves gains of **+8.40 Pearson** and **+7.20 Spearman** correlation over the SFT baseline, and **+18.30/+11.20** over the base model across four LLM-as-a-Judge benchmarks.

| Method | Paradigm | FB Bench (r) | FLASK (r) | Vic. Bench (r) | MT Bench (r) | Avg Pearson |
|--------|----------|-------------|-----------|---------------|-------------|-------------|
| Base | — | 63.4 | 54.3 | 50.8 | 42.5 | 52.7 |
| RAFT | SFT | 85.4 | 52.1 | 51.9 | 61.1 | 62.6 |
| **REAL** | **RL** | **91.1** | **58.9** | **65.1** | **68.9** | **71.0** |

## Environment Setup

Create and activate the `real` conda environment:

```bash
conda create -n real python=3.12 -y
source activate real

pip install -e .
pip3 install -e .[vllm]
pip install flash-attn==2.8.1 --no-build-isolation
pip install "transformers<4.54.0"
pip install "ray[default]"
pip install ray==2.38
conda install -c conda-forge rdma-core
```

See [azure/env_setup.sh](azure/env_setup.sh) for the full setup script.

## Quick Start

Run the main training script:

```bash
bash bash_real/qwen_rarl_grpo_4k_sft.sh <experiment_name>
```

This launches REAL training on 8 GPUs (single node) using vLLM for rollout generation. The training entry point is `recipe.dapo.main_jepo_dapo`.

 

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_resp_per_prompt` | 8 | Number of responses sampled per prompt |
| `max_prompt_length` | 2048 | Maximum prompt token length |
| `max_response_length` | 1024 | Maximum response token length |
| `train_prompt_bsz` | 256 | Training prompt batch size |
| `loss_agg_mode` | `token-mean` | Token-level loss aggregation |
| `clip_ratio_low/high` | 0.2 | PPO-style clipping ratios |
| `kl_loss_coef` | 0.01 | KL divergence regularization coefficient |
| `offload` | True | FSDP CPU offloading for memory efficiency |

## Project Structure

```
.
├── bash_real/                  # Training scripts (8B models, single-node)
├── bash_real_32b/              # Training scripts (32B models, multi-node Ray)
├── recipe/
│   └── dapo/                   # Training entry points and configs
│       ├── main_jepo_dapo.py   # Main training entry
│       ├── jepo_dapo_ray_trainer.py  # Ray trainer
│       ├── deepscaler_reward.py      # Reward function
│       └── config/             # Hydra config files
├── verl/                       # Core verl framework code
├── azure/                      # Environment setup scripts
├── data/                       # Data utilities
├── examples/                   # Example scripts and data preprocessing
└── tools/                      # Utility tools
```

## Scripts

### `bash_real/` — 8B Model Training Scripts

| Script | Description |
|--------|-------------|
| `qwen_rarl_grpo_4k_sft.sh` | **Main script** — REAL training from SFT checkpoint |
| `rarl_grpo_4k_sft.sh` | REAL + GRPO from SFT checkpoint |
| `rarl_grpo_4k_base.sh` | REAL + GRPO from base model |
| `dapo_4k.sh` / `dapo_8k.sh` | DAPO baseline (4k/8k context) |
| `grpo_1k.sh` / `grpo_4k.sh` | GRPO baseline (1k/4k context) |
| `sft.sh` | Supervised fine-tuning |
| `generate_samples.sh` | Sample generation for evaluation |

### `bash_real_32b/` — 32B Model Training Scripts

All 32B scripts use Ray for multi-node distributed training (2×8 A100 GPUs).

| Script | Description |
|--------|-------------|
| `real_32b.sh` | Base 32B training |
| `real_32b_ray_submit.sh` | 32B Ray submit baseline |
| `real_32b_ray_submit_from_raft.sh` | Init from RAFT checkpoint |
| `real_32b_ray_submit_from_raft_full_lr1e-6.sh` | From RAFT, full fine-tuning, lr=1e-6 |
| `real_32b_ray_submit_from_raft_r128.sh` | From RAFT, LoRA rank=128 |
| `real_32b_ray_submit_from_tract_full_lr*.sh` | From TRACT, full fine-tuning, various lr |
| `real_test.sh` | 32B evaluation script |

## Datasets

- **Training**: Feedback Collection (~100K pointwise samples with fine-grained score rubrics)
- **Evaluation**:
  - Feedback Bench (in-domain) — 1K rubrics, 200 instructions
  - FLASK (out-of-domain) — 200 prompts, 12 rubrics, 2K responses
  - Vicuna Bench (out-of-domain) — 80 prompts, 320 responses
  - MT Bench (out-of-domain) — 80 multi-turn prompts, 320 responses

## Citation

```bibtex
@inproceedings{zhang2026real,
  title     = {REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge},
  author    = {Zhang, Yasi and Chen, Tianyu and Zhou, Mingyuan and Leong, Oscar and Wu, Ying Nian and Lukasik, Michal},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
