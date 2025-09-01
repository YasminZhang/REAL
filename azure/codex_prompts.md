Algorithm

Inputs: question x, n responses, ground-truth answer tokens a.
Split: response i → CoT c_i (tokens before delimiter), delimiter tokens, teacher-forced a.
Likelihoods: ℓ_i = log πθ(a | x, c_i, delimiter) = sum of answer-token log-probs (no length averaging).
JEPO advantage:
log_mean = log((1/n) Σ_j exp(ℓ_j))
v_i = log((1/(n−1)) Σ_{j≠i} exp(ℓ_j))
A_i = log_mean − v_i; Â_i = clip(A_i / std(A), −1, 1) per question; detach.
Format advantage:
fmt_i = 0 if has delimiter else −p; ~fmt_i = fmt_i − mean(fmt) per question; detach.
Gradients:
Grad1 (CoT+delimiter): per-token advantage = 1{has_delim_i}·Â_i + ~fmt_i; aggregation = seq-mean-token-mean (normalizes by CoT+delimiter length).
Grad2 (Answer mixture): w_i = softmax(ℓ_i) per question; per-token advantage = β_supp · w_i; aggregation = token-mean (normalizes by answer length). Do not mask Grad2 by format.
KL (original responses): β · KL(πθ(.|x), π_ref(.|x)) computed token-wise on original sampled responses, not teacher-forced.
Total update: ∇ = Grad1 + Grad2 − β·∇KL.
Delimiter: token-level, allow suffix partial match, take the last match.
Defaults: β_supp ∈ [0, 0.01] (start small), β = 1e−3, p = 100 for sanity check (later 10).
Implementation Keys

Token-level split:
Encode delimiter once; find last match per response with optional suffix-anchor fallback; record has_delimiter, cot_start, answer_start.
Truncate CoT so that cot + delimiter + a fits the response window.
Teacher-forced packs:
Build per-response tensors: input_ids = prompt ⊕ cot ⊕ delimiter ⊕ a; track cot_start_positions and answer_start_positions.
Pad to batch; keep attention_mask, position_ids. Store in DataProto for reuse.
Batched log-probs (no special “responses_micro_batch_size”):
Use self._forward_micro_batch for all log-prob evals (same path as dp_actor.py).
Compute answer ℓ_i in batches by slicing rows into arbitrary-sized chunks to fit memory; no dependency on per-prompt grouping during this step.
Numerical stability: compute log_mean via logsumexp; LOO term via stable lse_others using d = ℓ − lse_all, lse_others = lse_all + log((-expm1(d)).clamp_min(1e-12)).
Precompute per-question stats:
For each question’s group of responses, compute ℓ_i (sum over answer tokens), Â_i, ~fmt_i, and mixture weights w_i = softmax(ℓ_i). Detach and attach to data.batch as jepo_adv_raw, format_adv, jepo_weights, has_delimiter.
Training loop (per step over all responses):
Build A_vec = 1{has_delim}·Â_i + ~fmt_i.
CoT forward: batch rows in chunks; feed prefixes up to answer_start with responses = CoT-only span; get token log-probs; losses via GPG with advantages = A_vec per token; aggregation = seq-mean-token-mean.
Answer forward: batch rows in chunks; feed full inputs with responses = answer span; losses via GPG with advantages = β_supp·w_i per token; aggregation = token-mean.
KL: use batch["ref_log_prob"] (precomputed), compute actor log-probs on original sampled responses (not teacher-forced), then kl_penalty(...) and agg_loss(..., "token-mean") scaled by β.
Masking behavior:
Grad1: A_i term masked by has_delim; ~fmt_i always applied.
Grad2: unmasked (as agreed).
Length normalization:
CoT+delimiter branch normalized by its own length via seq-mean-token-mean.
Answer branch normalized by answer length via token-mean.
Detach everywhere:
Â_i, ~fmt_i, w_i: scalars with stop-gradient.
Metrics to log:
frac_has_delimiter, num_has_delimiter, format_penalty (p), format_adv_max, jepo_advs_mean/std, cot_log_probs_mean, log_mean_answer_probs_mean, beta_supp, beta_kl, kl_loss.
Edge handling:
No CoT (length 0): skip CoT compute; only format and Grad2 contribute.
No answer tokens or padding-only rows: skip safely.
Distributed: allreduce frac_has_delimiter across ranks for a global view.