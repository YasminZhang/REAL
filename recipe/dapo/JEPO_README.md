# JEPO-DAPO Integration

This directory contains the JEPO (Just Exploration with Policy Optimization) algorithm integrated with the DAPO (Data-Augmented Policy Optimization) training pipeline.

## Overview

JEPO enhances the standard DAPO workflow by adding specialized training for cases where all model responses to a question are incorrect (reward = 0). Instead of just discarding these "failure" cases, JEPO:

1. **Buffers incorrect responses** from questions where all n responses have reward = 0
2. **Performs standard GRPO training** for normal cases
3. **Applies JEPO algorithm** to learn from the buffered incorrect responses

## Algorithm Integration

### Standard DAPO Flow
```
Question → Generate n responses → Compute rewards → GRPO training
```

### JEPO-Enhanced Flow  
```
Question → Generate n responses → Compute rewards
    ↓
All rewards = 0? → Buffer responses → Continue DAPO training
    ↓                    ↓
   No                   Yes (periodically)
    ↓                    ↓
GRPO training         JEPO training on buffered data
```

## Key Components

### 1. JEPO Core Algorithm (`jepo_core_algos.py`)
- **JEPOConfig**: Configuration for JEPO hyperparameters
- **JEPOBuffer**: Buffer to store incorrect response batches
- **compute_jepo_advantages**: Compute advantages using the JEPO algorithm
- **jepo_loss**: JEPO loss computation with all gradient components

### 2. JEPO-DAPO Trainer (`jepo_dapo_ray_trainer.py`)
- **RayJEPODAPOTrainer**: Extends `RayDAPOTrainer` with JEPO functionality
- Detects when all responses have reward = 0
- Buffers these cases for later JEPO training
- Performs JEPO training steps periodically

### 3. Configuration (`config/jepo_dapo_trainer.yaml`)
JEPO-specific parameters:
```yaml
algorithm:
  use_jepo: True                    # Enable JEPO functionality  
  jepo_delimiter: "\n\n"            # Split chain-of-thought from answer
  jepo_format_penalty: 0.1          # Penalty for missing delimiter
  jepo_beta_supp: 1.0               # Coefficient for suppression gradient
  jepo_beta_kl: 0.1                 # Coefficient for KL divergence gradient
  jepo_buffer_size: 100             # Buffer size for incorrect responses
  jepo_steps: 5                     # Number of JEPO training steps
  jepo_update_frequency: 10         # Run JEPO every N steps when buffer has data
```

## JEPO Algorithm Details

For buffered incorrect responses, JEPO computes:

1. **Split responses** by delimiter into chain-of-thought (c_i) and answer parts
2. **Compute advantages**: 
   - A_i = log(1/n ∑ π_θ(a|x,c_j)) - v_i
   - tilde_A_i = clip(A_i / std(A), -1, 1)
3. **Format advantages** based on delimiter presence
4. **Gradient computation**:
   - grad1: Policy gradient for chain-of-thought
   - grad2: Suppression gradient for answers  
   - grad3: KL divergence gradient
   - Total: grad1 + β_supp×grad2 - β_kl×grad3

## Usage

### Basic Training
```bash
cd recipe/dapo
python main_jepo_dapo.py \
    data.train_files=/path/to/train/data \
    data.val_files=/path/to/val/data \
    algorithm.use_jepo=True \
    algorithm.jepo_buffer_size=100
```

### Full Training Script
```bash
cd recipe/dapo
./test_jepo_dapo_7b_math.sh
```

## Files Structure

```
recipe/dapo/
├── jepo_dapo_ray_trainer.py      # JEPO-enhanced DAPO trainer
├── main_jepo_dapo.py             # Main training script
├── config/
│   └── jepo_dapo_trainer.yaml    # Configuration file
├── test_jepo_dapo_7b_math.sh     # Test script for 7B model
└── JEPO_README.md                # This documentation
```

## Metrics and Monitoring

The JEPO-DAPO trainer logs additional metrics:
- `jepo/buffer_size`: Current buffer size
- `jepo/buffer_full`: Whether buffer is full
- `jepo_*`: JEPO-specific training metrics
- `timing/jepo_training`: Time spent on JEPO training

## Benefits

1. **Utilizes failure cases**: Instead of discarding incorrect responses, learns from them
2. **Improves chain-of-thought**: Encourages proper reasoning structure with delimiter
3. **Maintains DAPO benefits**: Keeps all advantages of standard DAPO training
4. **Minimal overhead**: Only activates when needed (all responses incorrect)

## Comparison with Standard DAPO

| Aspect | Standard DAPO | JEPO-DAPO |
|--------|---------------|-----------|
| Correct responses | GRPO training | GRPO training |  
| Mixed responses | GRPO training | GRPO training |
| All incorrect | Discarded | Buffered → JEPO training |
| Chain-of-thought | Not explicitly encouraged | Encouraged via format rewards |
| Training steps | n | n + periodic JEPO steps |