# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import torch
import numpy as np

sys.path.append('/home/aiscuser/jepo/recipe/jepo')

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor
from tqdm import tqdm

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

from jepo_core_algos import compute_jepo_advantages, compute_jepo_from_logits_sparse, jepo_two_pass_step_for_one_question

__all__ = ["JEPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]

from contextlib import nullcontext
import math
import numpy as np
import torch

def _chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

class JEPOActor(DataParallelPPOActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initialized JEPO Actor")
        self._cached_tokenizer = None

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="jepo actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()
        self.actor_optimizer.zero_grad()

        # -------- config --------
        jepo_cfg = data.meta_info.get("jepo_config", {}) or {}
        epochs = int(jepo_cfg.get("epochs", 1))
        mini_bs = int(jepo_cfg.get("mini_batch_size", 1))                # questions per optimizer step
        micro_bs = int(jepo_cfg.get("micro_batch_size_per_gpu", 1))      # questions per backward call
        resp_micro_bs = int(jepo_cfg.get("responses_micro_batch_size", 4))  # responses per backward inside a question
        format_penalty = float(jepo_cfg.get("format_penalty", 1.0))
        beta_supp = float(jepo_cfg.get("beta_supp", 1.0))
        temperature = float(data.meta_info["temperature"])

        # -------- prepare data_dicts (one per question) --------
        model_inputs = {**data.batch, **data.non_tensor_batch}
        ground_truths = model_inputs.get("reward_model", {})
        ground_truths_tokens = np.array(
            [self._cached_tokenizer.encode(gt.get("ground_truth", [])) for gt in ground_truths],
            dtype=object,
        )
        delimiter = jepo_cfg.get("delimiter", "\n\n")
        pad_token = self._cached_tokenizer.pad_token_id

        data_dicts = compute_jepo_advantages(
            response_tokens=data.batch["responses"],
            prompt_tokens=data.batch["prompts"],
            ground_truth_answer_tokens=ground_truths_tokens,
            delimiter_str=delimiter,
            format_penalty=format_penalty,
            model=self.actor_module,
            device=self.actor_module.device,
            pad_token=pad_token,
            index=data.non_tensor_batch["uid"],
            tokenizer=self._cached_tokenizer,
        )

        # -------- meters --------
        meters = dict(
            jepo_loss=0.0,
            supp_loss=0.0,
            total_loss=0.0,
            grad_norm=0.0,
            jepo_advs_mean=0.0,
            jepo_advs_std=0.0,
            cot_log_probs_mean=0.0,
            log_mean_answer_probs_mean=0.0,
        )
        meter_count = 0
        num_delim = 0

        # -------- training loop (mini/micro/accum) --------
        for _ in range(epochs):
            for mini in _chunk_list(data_dicts, mini_bs):
                N_q = len(mini)  # number of questions in this optimizer step
                accum_steps = math.ceil(N_q / micro_bs)
                self.actor_optimizer.zero_grad()

                for k, micro in enumerate(_chunk_list(mini, micro_bs)):
                    # FSDP: delay allreduce until last accum step
                    sync_ctx = (
                        self.actor_module.no_sync()
                        if isinstance(self.actor_module, (FSDP, FSDPModule)) and (k < accum_steps - 1)
                        else nullcontext()
                    )
                    with sync_ctx:
                        # process a few questions; each question internally micro-batches its responses
                        for dd in micro:
                            num_delim += int(np.sum(dd["has_delimiter"]))

                            q_metrics = jepo_two_pass_step_for_one_question(
                                model=self.actor_module,
                                data_dict=dd,
                                temperature=temperature,
                                beta_supp=beta_supp,
                                format_penalty=format_penalty,
                                responses_micro_bs=resp_micro_bs,
                                vocab_chunk=8192,
                                device_name=self.device_name,
                                accum_scale=1.0 / N_q,   # <-- scale for questions in each mini batc
                            )
                            # scale for gradient accumulation so overall grad is mean over the mini-batch
                            # (we already averaged inside each question by B; now average over questions)
                            for k2 in ("total_loss", "jepo_loss", "supp_loss"):
                                q_metrics[k2] = q_metrics[k2] / accum_steps
                            # keep the *last* loss tensor on graph? no—metrics are detached scalars already
                            # just re-create a small scalar for backward scaling:
                            loss_scale = q_metrics["total_loss"]
                            # we already called backward *inside* the function;
                            # here we only accumulate metrics (no extra backward)

                            # accumulate meters
                            for mk in meters:
                                if mk in q_metrics:
                                    meters[mk] += float(q_metrics[mk])
                            meter_count += 1

                # one optimizer step per mini-batch
                grad_norm = self._optimizer_step()
                meters["grad_norm"] += float(grad_norm.detach())

        print("number of responses has delimiter:", num_delim)

        # average meters
        if meter_count > 0:
            for k in meters:
                meters[k] /= meter_count

        return {
            "jepo_actor/jepo_loss": meters["jepo_loss"],
            "jepo_actor/supp_loss": meters["supp_loss"],
            "jepo_actor/total_loss": meters["total_loss"],
            "jepo_actor/grad_norm": meters["grad_norm"],
            "jepo_actor/jepo_advs_mean": meters["jepo_advs_mean"],
            "jepo_actor/jepo_advs_std": meters["jepo_advs_std"],
            "jepo_actor/cot_log_probs_mean": meters["cot_log_probs_mean"],
            "jepo_actor/log_mean_answer_probs_mean": meters["log_mean_answer_probs_mean"],
            "jepo_actor/beta_supp": beta_supp,
            "jepo_actor/format_penalty": format_penalty,
        }