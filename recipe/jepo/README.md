# JEPO (Just Exploration with Policy Optimization)

JEPO is a training recipe that combines the GRPO workflow with a specialized algorithm for handling cases where all responses to a question are incorrect.

## Algorithm Overview

The JEPO algorithm follows this workflow:

1. **Buffer Management**: Maintain a buffer with maximum length B
2. **Standard GRPO Training**: Perform typical GRPO training steps
3. **Incorrect Response Handling**: For questions where all responses are incorrect (reward = 0):
   - Record the question (x), answer (a), and n responses
   - Add to the JEPO buffer
4. **JEPO Training**: When buffer is full, perform JEPO training steps

## Core Algorithm

For each batch of responses where all have reward = 0:

1. **Split Responses**: Use a delimiter (hyperparameter) to split each response into:
   - Chain-of-thought part (c_j)
   - Answer part

2. **Compute Advantages**: Calculate A_i for each response i:
   ```
   A_i = log(1/n * Σ π_θ(a|x,c_j)) - v_i
   where v_i = log(1/(n-1) * Σ_{j≠i} π_θ(a|x,c_j))
   ```

3. **Normalize Advantages**: 
   ```
   tilde_A_i = clip(A_i / std(A), -1, 1)
   ```

4. **Format Rewards**: Calculate format advantages:
   - A_i_format = 0 if delimiter present in response
   - A_i_format = -p if delimiter absent (p is hyperparameter)
   - Normalize: tilde_A_i_ref = (A_i_format - mean) / std

5. **Compute Gradients**:
   - grad1 = 1/n * Σ((tilde_A_i + tilde_A_i_ref) * ∇_θ log π_θ(c_i|x))
   - grad2 = ∇_θ log(1/n * Σ π_θ(a|x,c_i))
   - grad3 = ∇_θ KL(π_θ(.|x), π_ref(.|x))
   - Final gradient = grad1 + β_supp * grad2 - β * grad3

## Configuration

Key hyperparameters:
- `jepo_delimiter`: String delimiter to split chain-of-thought (default: "\n\n")
- `jepo_format_penalty`: Penalty p for responses without delimiter (default: 0.1)
- `jepo_beta_supp`: Coefficient for suppression gradient (default: 1.0)
- `jepo_beta_kl`: Coefficient for KL divergence gradient (default: 0.1)
- `jepo_buffer_size`: Maximum buffer size (default: 1000)
- `jepo_steps`: Number of JEPO training steps when buffer is full (default: 5)

## Usage

```bash
cd recipe/jepo
python main_jepo.py --config-name jepo_trainer \
    actor_rollout_ref.model.path=/path/to/model \
    data.train_files=[/path/to/train/data] \
    data.val_files=[/path/to/val/data]
```