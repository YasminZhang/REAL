#!/usr/bin/env python3
"""
Minimal script to compare log-prob computation methods:

- actor-like (teacher-forced on prompt+response; shift-and-gather)
- forward without prompt context (response-only)
- optional no-shift variant for illustration

Usage:
  python tools/real_compare_logprob.py \
    --model gpt2 \
    --prompt "What is 2+3?" \
    --response "Let's think step by step. The answer is 5." \
    [--device cuda]

This highlights two common pitfalls that cause inconsistencies:
  1) Missing prompt conditioning (feeding only the response tokens)
  2) Missing shift when gathering token log-probs
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.utils.torch_functional import logprobs_from_logits
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import ActorConfig
import torch.distributed as dist


def actor_like_logprob(model, tokenizer, prompt: str, response: str, device: str = "cpu"):
    """Actor-style: teacher-forced on prompt+response; shift-and-gather on the response segment only."""
    model.eval()
    with torch.no_grad():
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        resp_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        input_ids = torch.cat([prompt_ids, resp_ids], dim=1)  # [1, P+R]
        attn = torch.ones_like(input_ids)

        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits  # [1, P+R, V]

        # teacher-forcing: predict token t using logits at t-1
        shift_logits = logits[:, :-1, :]
        # response labels are the response tokens; align to the end of the sequence
        R = resp_ids.size(1)
        target_labels = input_ids[:, -R:]  # the response tokens
        source_logits = shift_logits[:, -R:, :]  # positions right before each response token
        token_logprobs = logprobs_from_logits(source_logits, target_labels)
        return token_logprobs.squeeze(0)  # [R]


def forward_resp_only_logprob(model, tokenizer, response: str, device: str = "cpu"):
    """Response-only: teacher-forced on response without prompt context."""
    model.eval()
    with torch.no_grad():
        resp_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        attn = torch.ones_like(resp_ids)
        out = model(input_ids=resp_ids, attention_mask=attn, use_cache=False)
        logits = out.logits  # [1, R, V]
        shift_logits = logits[:, :-1, :]
        labels = resp_ids[:, 1:]  # next tokens within response only
        token_logprobs = logprobs_from_logits(shift_logits, labels)
        # pad one position to match length if desired; here we keep length R-1 (first token has no context)
        return token_logprobs.squeeze(0)  # [R-1]


def no_shift_logprob(model, tokenizer, prompt: str, response: str, device: str = "cpu"):
    """Incorrect variant: gather probabilities at the same positions (no shift)."""
    model.eval()
    with torch.no_grad():
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        resp_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        input_ids = torch.cat([prompt_ids, resp_ids], dim=1)
        attn = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits  # [1, P+R, V]
        R = resp_ids.size(1)
        labels = input_ids[:, -R:]  # the response tokens
        same_pos_logits = logits[:, -R:, :]  # wrong: using logits at the same time-step
        token_logprobs = logprobs_from_logits(same_pos_logits, labels)
        return token_logprobs.squeeze(0)  # [R]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2", help="HF model id, e.g., gpt2 or Qwen/Qwen2.5-1.5B")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--prompt", type=str, default="What is 2+3?")
    ap.add_argument("--response", type=str, default="Let's think step by step. The answer is 5.")
    ap.add_argument("--via_actor", action="store_true", help="Also compute via DataParallelPPOActor.compute_log_prob")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    lp_actor = actor_like_logprob(model, tok, args.prompt, args.response, args.device)
    lp_resp = forward_resp_only_logprob(model, tok, args.response, args.device)
    lp_noshift = no_shift_logprob(model, tok, args.prompt, args.response, args.device)

    # Optional: run through VERL actor.compute_log_prob (single-rank init)
    lp_via_actor = None
    if args.via_actor:
        # Initialize process group (file init to avoid env requirements)
        if not dist.is_initialized():
            try:
                init_file = f"/tmp/pg_{os.getpid()}"
                dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
            except Exception as e:
                print(f"WARN: could not init torch.distributed (via_actor skipped): {e}")
        try:
            cfg = ActorConfig()
            actor = DataParallelPPOActor(config=cfg, actor_module=model, actor_optimizer=None)

            # Build DataProto batch: [prompt || response] with response slice
            prompt_ids = tok(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
            resp_ids = tok(args.response, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
            input_ids = torch.cat([prompt_ids, resp_ids], dim=1)
            attn = torch.ones_like(input_ids)
            pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)

            data = DataProto.from_dict(
                tensors={
                    "input_ids": input_ids,
                    "attention_mask": attn,
                    "position_ids": pos,
                    "responses": resp_ids,
                }
            )
            data.meta_info.update({
                "micro_batch_size": 1,
                "temperature": 1.0,
                "use_dynamic_bsz": False,
            })
            lp, _ = actor.compute_log_prob(data, calculate_entropy=False)
            lp_via_actor = lp.squeeze(0).detach().cpu()
        except Exception as e:
            print(f"WARN: actor.compute_log_prob failed (via_actor skipped): {e}")

    # Print summaries
    def s(v):
        return float(v.sum().item()) if v.numel() > 0 else 0.0

    print("=== Shapes ===")
    print(f"actor_like: per-token {tuple(lp_actor.shape)}, sum={s(lp_actor):.6f}")
    print(f"resp_only:  per-token {tuple(lp_resp.shape)} (R-1), sum={s(lp_resp):.6f}")
    print(f"no_shift:   per-token {tuple(lp_noshift.shape)}, sum={s(lp_noshift):.6f}")
    if lp_via_actor is not None:
        print(f"via_actor:  per-token {tuple(lp_via_actor.shape)}, sum={s(lp_via_actor):.6f}")

    # Align for comparison (trim first actor token to match R-1 length of resp-only)
    Rm1 = min(lp_actor.numel() - 1, lp_resp.numel())
    if Rm1 > 0:
        diff_resp = (lp_actor[1:1+Rm1] - lp_resp[:Rm1]).abs().mean().item()
    else:
        diff_resp = float('nan')
    # no-shift vs actor-like on overlapping length
    R = min(lp_actor.numel(), lp_noshift.numel())
    diff_noshift = (lp_actor[:R] - lp_noshift[:R]).abs().mean().item() if R > 0 else float('nan')

    print("\n=== Mean Absolute Differences ===")
    print(f"actor_like vs resp_only (aligned): {diff_resp:.6e}")
    print(f"actor_like vs no_shift:            {diff_noshift:.6e}")
    if lp_via_actor is not None:
        K = min(lp_actor.numel(), lp_via_actor.numel())
        diff_actor = (lp_actor[:K] - lp_via_actor[:K]).abs().mean().item() if K > 0 else float('nan')
        print(f"actor_like vs via_actor:           {diff_actor:.6e}")

    print("\nNotes:")
    print("- actor_like matches PPO/GRPO compute_log_prob (prompt-conditioned, shift-and-gather).")
    print("- resp_only ignores the prompt context; first token also lacks any preceding context.")
    print("- no_shift gathers at the same positions (common bug), overestimating probabilities.")
    if lp_via_actor is None:
        print("- Tip: pass --via_actor to compare with DataParallelPPOActor.compute_log_prob.")


if __name__ == "__main__":
    main()
