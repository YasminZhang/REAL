# REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge

REAL is a principled reinforcement learning framework designed to optimize **regression rewards** for LLM-as-a-Judge tasks. Unlike standard RL methods that rely on binary rewards (e.g., 0-1 accuracy), REAL explicitly models the **ordinal structure** inherent in numeric scoring, recognizing that predicting 4 is significantly better than predicting 1 when the ground truth is 5.

**Paper**: [REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge](https://arxiv.org/abs/2603.17145) (ICML 2026)

**Authors**: [Yasi Zhang](https://yasminzhang.github.io/)*, [Tianyu Chen](https://tianyucodings.github.io/)*, Mingyuan Zhou, Oscar Leong, Ying Nian Wu, [Michal Lukasik](https://mlukasik.github.io/)

## Key Features

- **Regression-Aware Reward**: Optimizes a policy-dependent regression loss that captures the ordinal structure of evaluation scores, proven to be optimal for correlation metrics (Pearson, Spearman)
- **Generalized Policy Gradient**: Handles policy-dependent rewards via a generalized gradient estimator that naturally decomposes into two complementary components:
  1. **CoT Exploration** — policy gradient over Chain-of-Thought trajectories weighted by regression-aware rewards
  2. **Prediction Refinement** — regression-aware supervision on the final numeric score via standard backpropagation
- **RLOO Stabilization**: Uses the leave-one-out baseline for variance reduction without requiring a learned value function
- **RAIL Inference**: Computes expected value over digit tokens for regression-aware predictions at inference time
- **Multi-scale Support**: Validated across 8B and 32B model scales with ready-to-use training scripts
- **Built on verl**: Leverages the [verl](https://github.com/volcengine/verl) framework for efficient FSDP/vLLM integration and Ray distributed training

## Key Takeaway
```
In RL, the probability assigned to the answer token offers a richer, more informative reward signal than binary accuracy alone.
```

## Main Results

On Qwen3-32B, REAL achieves gains of **+8.40 Pearson** and **+7.20 Spearman** correlation over the SFT baseline, and **+18.30/+11.20** over the base model across four LLM-as-a-Judge benchmarks.


| Method   | Paradigm | FB Bench (r) | FLASK (r) | Vic. Bench (r) | MT Bench (r) | Avg Pearson |
| -------- | -------- | ------------ | --------- | -------------- | ------------ | ----------- |
| Base     | —        | 63.4         | 54.3      | 50.8           | 42.5         | 52.7        |
| RAFT     | SFT      | 85.4         | 52.1      | 51.9           | 61.1         | 62.6        |
| **REAL** | **RL**   | **91.1**     | **58.9**  | **65.1**       | **68.9**     | **71.0**    |


## Environment Setup

Create and activate the `real` conda environment:

```bash
conda create -n real python=3.12 -y
conda activate real

pip install -e .
pip3 install -e .[vllm]
pip install flash-attn==2.8.1 --no-build-isolation
pip install "transformers<4.54.0"
pip install "ray[default]"
pip install ray==2.38
conda install -c conda-forge rdma-core
```

Or run [azure/env_setup.sh](azure/env_setup.sh) under the main folder for the full setup script.


You can login wandb first:
```
wandb login
```


## Datasets

- **Training**: Feedback Collection (~100K pointwise samples with fine-grained score rubrics)
- **Evaluation**:
  - Feedback Bench (in-domain, new rubrics) — 1K rubrics, 200 instructions, 1K responses
  - FLASK (out-of-domain) — 200 prompts, 12 rubrics, 2K responses
  - Vicuna Bench (out-of-domain) — 80 prompts, 320 responses
  - MT Bench (out-of-domain) — 80 multi-turn prompts, 320 responses


### Option 1 — Download the prepared parquet files (recommended)

All preprocessed train/eval splits are mirrored on the Hugging Face Hub at
[yasiz/real_data](https://huggingface.co/datasets/yasiz/real_data):

```bash
huggingface-cli download yasiz/real_data \
    --repo-type dataset \
    --local-dir ./data/real_data
```

This produces the following layout (paths used by the training scripts):


| File                                          | Purpose                         |
| --------------------------------------------- | ------------------------------- |
| `real_dataset/feedback_ood_test/test.parquet` | Feedback Bench (in-domain eval) |
| `real_dataset/flask/test.parquet`             | FLASK (OOD eval)                |
| `real_dataset/mt_bench/test.parquet`          | MT Bench (OOD eval)             |
| `real_dataset/vicuna/test.parquet`            | Vicuna Bench (OOD eval)         |


Update `TRAIN_FILE`, `TEST_FILE`, and `extra_val_files` in your launch script
(e.g. [bash_real/run_real.sh](bash_real/run_real.sh)) to point at this directory.

If you want to SFT (e.g. RAFT/TRACT), use:


| File                                  | Purpose             |
| ------------------------------------- | ------------------- |
| `real_dataset_sft/train.parquet`      | SFT warmup training |
| `real_dataset_sft/test/train.parquet` | SFT validation      |


### Option 2 — Regenerate from source

The scripts in `[data/](data)` rebuild the parquet files from the original
Prometheus releases on Hugging Face:

```bash
# RL training set (Feedback-Collection)
python data/collection.py     --local_dir ./data/feedback_collection_for_base
# SFT warmup set (Feedback-Collection, SFT-formatted)
python data/collection_sft.py --local_dir ./data/feedback_collection_for_base_warmup
# In-domain eval set (Feedback-Bench)
python data/bench.py          --local_dir ./data/feedback_bench_for_base
python data/bench_sft.py      --local_dir ./data/feedback_bench_for_base_warmup
```

## Model Checkpoints

All trained checkpoints and the baselines used in the paper are hosted on the Hub
under the `yasiz/` namespace.


| Checkpoint                                                                                                      | Method              | Base model               | Size   |
| --------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------ | ------ |
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)                                                       | Base                | —                        | ~62 GB |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)                                                         | Base                | —                        | ~16 GB |
| [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)               | Base                | —                        | ~14 GB |
| [yasiz/Qwen3-32B-REAL](https://huggingface.co/yasiz/Qwen3-32B-REAL)                                          | REAL (ours)         | Qwen3-32B                | 393 GB |
| [yasiz/Qwen3-8B-REAL](https://huggingface.co/yasiz/Qwen3-8B-REAL)                                            | REAL (ours)         | Qwen3-8B                 | 98 GB  |
| [yasiz/Mistral-7b-v0.2-Instruct-REAL](https://huggingface.co/yasiz/Mistral-7b-v0.2-Instruct-REAL)             | REAL (ours)         | Mistral-7B-v0.2-Instruct | 101 GB |
| [yasiz/Qwen3-32B-RAFT](https://huggingface.co/yasiz/Qwen3-32B-RAFT)                                           | RAFT (SFT baseline) | Qwen3-32B                | 66 GB  |
| [yasiz/Qwen3-8B-RAFT](https://huggingface.co/yasiz/Qwen3-8B-RAFT)                                             | RAFT (SFT baseline) | Qwen3-8B                 | 33 GB  |
| [yasiz/Qwen3-8B-TRACT](https://huggingface.co/yasiz/Qwen3-8B-TRACT)                                           | TRACT baseline      | Qwen3-8B                 | 33 GB  |
| [yasiz/Mistral-7b-v0.2-Instruct-TRACT-copy](https://huggingface.co/yasiz/Mistral-7b-v0.2-Instruct-TRACT-copy) | TRACT baseline      | Mistral-7B-v0.2-Instruct | 29 GB  |


Download a single checkpoint:

```bash
huggingface-cli download yasiz/Mistral-7b-v0.2-Instruct-TRACT-copy --local-dir ./ckpts/Mistral-7b-v0.2-Instruct-TRACT-copy
```

Then point `MODEL_PATH` in your launch script (e.g. [bash_real/run_real.sh](bash_real/run_real.sh))
at the local directory.

> The REAL checkpoints (and `Qwen3-32B-RAFT`) are exported in raw FSDP shard
> format from training. Use its subfolder: ./ckpts/Qwen3-8B-REAL/actor/huggingface instead.

## Quick Start

Run the main training script:

```bash
bash bash_real/run_real.sh <experiment_name>
```

This launches REAL training on 8 GPUs (single node) using vLLM for rollout generation. The training entry point is `recipe.dapo.main_jepo_dapo`.

If you want to run 32B models:

```bash
bash bash_real/run_real_32B.sh <experiment_name>
```

A quick sanity check is that in the printing:

Expected values should be reasonable decimals, instead of numbers very close to 0 (that means the digit token's index might be wrong.) For example:
```
Expected values (first 5): [1.8960844 2.8807998 4.1740913 2.6792083 2.679181]
```

## Training Configuration

All defaults below are taken from [`bash_real/run_real.sh`](bash_real/run_real.sh). Edit the corresponding bash variable to change a value — every entry is wired straight through to a Hydra override on the `python3 -m recipe.dapo.main_real_dapo` line.



### Sequence lengths

| Variable                  | Default | Description                                                                    |
| ------------------------- | ------- | ------------------------------------------------------------------------------ |
| `max_prompt_length`       | `2048`  | Truncate/pad prompts to this many tokens (`data.truncation='left'`).           |
| `max_response_length`     | `1024`  | Hard cap on generated tokens. Beyond this, responses are simply truncated.     |

### Batch sizes & DAPO filter groups

| Variable                | Default | Description                                                                       |
| ----------------------- | ------- | --------------------------------------------------------------------------------- |
| `train_prompt_bsz`      | `256`   | Number of distinct prompts per global training batch.                             |
| `train_prompt_mini_bsz` | `64`    | PPO mini-batch size (used by the standard PPO update path).                       |
| `n_resp_per_prompt`     | `8`     | vLLM rollouts per prompt (also the LOO group size).                               |


### REAL — hyperparameters

| Variable             | Default | Description                                                                                                                 |
| -------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------- |
| `real_lr`            | `5e-8`  | REAL actor LR. Rule of thumb: **Qwen ~1e-6**, **Mistral ~5e-8**, **LoRA ≈ 10× full-finetune LR**.                            |
| `real_beta_supp`     | `1.0`   | λ — weight on the support (log-likelihood) loss. Recommended `1.0` for best correlation.                                    |
| `real_beta_supp_extra` | `0.0` | β — weight on the L2 + log-likelihood extra-loss bundle. `0.0` is faster and gives reasonable results. 0.01 gives the best performance.                        |
| `real_beta_kl`       | `0.0`   | KL coefficient on the original rollout policy (off by default).                                                             |
| `real_entropy_coeff` | `0.0`   | Entropy regularization (off by default).                                                                                    |
| `real_update_freq`   | `10`    | Eval/save cadence (steps). Qwen models run typically use `20`.                                                                    |
| `val_before_train`   | `True`  | Run validation once before any training step (sanity-check baseline metrics).                                               |



## Core Scripts

The files below are the ones you'll most often touch when running or extending REAL.

### Environment

- **[`azure/env_setup.sh`](azure/env_setup.sh)** — One-shot setup script that creates the `real` conda env and installs verl, vLLM, flash-attn 2.8.1, `transformers<4.54`, and Ray 2.38. Equivalent to the steps under [Environment Setup](#environment-setup).

### Launch scripts (entry points)

- **[`bash_real/sft.sh`](bash_real/sft.sh)** — SFT warmup launcher (RAFT / TRACT style). Wraps `verl/trainer/fsdp_sft_trainer.py` on `n` GPUs via `torchrun --standalone`, training a base model on `real_dataset_sft/`. Use this to produce the warm-start checkpoint that REAL then RL-finetunes.
- **[`bash_real/run_real.sh`](bash_real/run_real.sh)** — Main 8-GPU REAL launcher (Mistral-7B / Qwen3-8B scale). Drives the `recipe.dapo.main_real_dapo` entry point with all `algorithm.real_*` Hydra overrides. Edit `MODEL_PATH`, `DATA_DIR`, and the REAL hyperparameters (`real_lr`, `real_beta_supp`, `real_use_*_loss`, etc.) here.
- **[`bash_real/real_32b_ray_submit_from_raft_full_lr1e-6.sh`](bash_real/real_32b_ray_submit_from_raft_full_lr1e-6.sh)** — Multi-node Ray-submit launcher for the 32B-scale REAL run starting from the `Qwen3-32B-RAFT` checkpoint, with `lr=1e-6` and FSDP sharded across nodes. Use as the template for any 32B run.

### Training core

- **[`verl/trainer/ppo/ray_trainer.py`](verl/trainer/ppo/ray_trainer.py)** — `RayPPOTrainer`: the Ray-based single-controller PPO loop that REAL subclasses. Owns rollout generation (vLLM), reference/critic log-prob computation, advantage estimation, the actor-update call into the worker group, and the val/save cadence. The REAL trainer (`recipe/dapo/real_dapo_ray_trainer.py`) overrides `fit()` here to inject the REAL teacher-forced update.
- **[`verl/workers/actor/dp_actor.py`](verl/workers/actor/dp_actor.py)** — `DataParallelPPOActor`: the FSDP actor worker. Implements `_forward_micro_batch` (with the regression branch that computes `E[digit] = Σ p(k) · k` over the digit tokens for the last position), `compute_log_prob` (teacher-forced log-probs for rollouts and the reference policy), and `update_policy` (standard PPO/GRPO update). REAL's actor inherits from this.
- **[`verl/workers/actor/real_actor.py`](verl/workers/actor/real_actor.py)** — `REALActor`: subclasses `DataParallelPPOActor` and replaces `update_policy` with the REAL objective. `_precompute_adv_w_with_verl` runs Stage 1 (no-grad teacher-forced forward over the prompt + ground-truth answer per question to read `E[digit]` and the last-token log-prob) and Stage 2 (per-UID leave-one-out advantages from the regression reward `R = -(E[digit] - y)²` and the accuracy reward `R = p(y)`). `update_policy` then runs a second pass with grad enabled to combine the CoT policy-gradient loss with the regression supervision terms (`l2_loss`, `log_likelihood_loss`) weighted by `beta_supp`/`beta_supp_extra`, and steps the optimizer.


```
Feel free to raise an issue on Github if there's any questions on the paper or the code. 
```



## Citation

```bibtex
@inproceedings{zhang2026real,
  title     = {REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge},
  author    = {Zhang, Yasi and Chen, Tianyu and Zhou, Mingyuan and Leong, Oscar and Wu, Ying Nian and Lukasik, Michal},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

