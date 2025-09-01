Here’s the updated JEPO spec reflecting our decisions.

Setup

Inputs: question x, ground-truth answer tokens a, n sampled responses per question.
Delimiter: token-level string; allow suffix partial match; pick the last match.
Split: response i → c_i (tokens before delimiter; empty if none), delimiter tokens, then teacher-forced a.
Stop-grad: all advantages and weights are detached scalars.
Answer Likelihoods

Context: evaluate under πθ(a | x, c_i, delimiter).
Sequence log-likelihood: ℓ_i = sum of answer-token log-probs (no length normalization).
JEPO Advantage (CoT branch)

Per question over its n responses:
log_mean = log((1/n) Σ_j exp(ℓ_j))
v_i = log((1/(n−1)) Σ_{j≠i} exp(ℓ_j)) [leave-one-out]
A_i = log_mean − v_i
Normalize: Â_i = clip(A_i / std(A), −1, 1) across the n responses.
Format Advantage

Raw: fmt_i = 0 if has delimiter else −p (p > 0 provided as input).
Normalize: ~fmt_i = fmt_i − mean(fmt) across the n responses (no std).
Gradients

Grad1 (CoT + delimiter):
Tokens: all tokens in c_i ⊕ delimiter (delimiter included).
Advantage per response: 1{has_delim_i}·Â_i + ~fmt_i.
Aggregation: normalize by sequence length (CoT + delimiter) via token-mean.
Grad2 (Answer support, mixture gradient of log-mean):
Weights: w_i = softmax(ℓ_i) over the n responses for the question.
Tokens: answer tokens only; per-token advantage = β_supp · w_i.
Aggregation: normalize by answer length via token-mean.
Grad3 (KL penalty):
Same as dp_actor.py over the original sampled responses (token-level KL).
Coefficient: β.
Total Update

Gradient: ∇ = Grad1 + Grad2 − β·∇KL
Scalars: β_supp scales Grad2; β scales KL.
Workflow

Buffer: for now include all responses to validate; later run JEPO only on all-incorrect questions.
Re-sample: for all-incorrect in GRPO, re-sample as in DAPO.
Sanity check: start with p = 100 (positive input; code applies −p for missing delimiter) and monitor frac_has_delimiter.
Defaults

β_supp: 0–0.01 (small)
β (KL): 0.001
p: 100 for initial formatting push