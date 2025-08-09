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

from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from jepo_core_algos import compute_jepo_advantage

__all__ = ["JEPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


class JEPOActor(DataParallelPPOActor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initialized JEPO Actor")
        self.cached_tokenizer = None

    def update_policy(self, data: DataProto):
        self.actor_module.train()
        
        # Compute response mask and JEPO advantages for the whole batch first
        if "response_mask" not in data.batch:
            data.batch["response_mask"] = compute_response_mask(data)
        
        jepo_config = data.meta_info.get("jepo_config", {})
        model_inputs = {**data.batch, **data.non_tensor_batch}
        ground_truths = model_inputs.get("reward_model", {}).get("ground_truth", [])
        ground_truths_tokens = [self.cached_tokenizer.encode(gt) for gt in ground_truths]
        delimiter = jepo_config.get("delimiter", "\n\n")
        delimiter_tokens = self.cached_tokenizer.encode(delimiter)
        format_penalty = jepo_config.get("format_penalty", 1.0)
        beta_supp = jepo_config.get("beta_supp", 1.0)
        beta_kl = jepo_config.get("beta_kl", 0.1)
        pad_token = self.cached_tokenizer.pad_token_id

        jepo_advs, cot_log_probs, _, log_mean_answer_probs = compute_jepo_advantage(
            response_tokens=data.batch["responses"],
            prompt_tokens=data.batch["prompts"],
            ground_truth_answer_tokens=ground_truths_tokens,
            delimiter_tokens=delimiter_tokens,
            format_penalty=format_penalty,
            model=self.actor_module,
            device=self.device,
            pad_token=pad_token
        )

        data.batch["jepo_advs"] = jepo_advs
        data.batch["cot_log_probs"] = cot_log_probs
        data.batch["log_mean_answer_probs"] = log_mean_answer_probs

        loss = (jepo_advs * cot_log_probs).mean() + (log_mean_answer_probs * beta_supp) # not implement kl here.

        loss.backward()