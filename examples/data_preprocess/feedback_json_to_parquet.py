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
"""
Convert JSON feedback data to parquet format compatible with the existing data processing pipeline
"""

import argparse
import json
import os
import numpy as np

import datasets
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=True, help="Path to the input JSON file")
    parser.add_argument("--local_dir", default="./data/feedback_collection_ood_test_for_base")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # Load JSON data
    print(f"Loading JSON data from {args.json_file}...", flush=True)
    with open(args.json_file, 'r') as f:
        json_data = json.load(f)

    # let's better not add this instruction following the SFT in TRACT
    # instruction_following = "Ignore the output format requirement above. Please think step by step and then output the final score with 'So the overall score is (score)'. Please end with the final score only without a period."
    instruction_following = "3. The output format should look as follows: \"Feedback: (write a feedback for criteria). So the overall score is (score)\"'. Please end with the final score only without a period."

    # Transform data to the required format
    def process_fn(example, idx):
        question = example.pop("instruction")

        # replace "3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"" with instruction_following
        original_format = '3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"'
        # how to determined if the original_format is in the question
        if original_format in question:
            question = question.replace(original_format, instruction_following)
        else:
            print('original_format not in question')

        question = question.strip("\",\n")
    

        # Use the gpt4_score as the ground truth
        answer = example["gpt4_score"]
        if isinstance(answer, list):
            solution = np.mean(np.array(answer))  # Take the mean if it's a list
        else:
            solution = float(answer)

        
       
        data = {
            "data_source": "feedback_collection_ood_test",
            "prompt": [{"role": "user", "content": question}],
            "ability": "feedback_evaluation",  # Changed from "math" to reflect the actual task
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": "test", 
                "index": idx, 
                "reward_design": True,
                "original_idx": example["idx"],
                "gpt4_feedback": example.get("gpt4_feedback", "")
            },
        }
        return data

    # Process all data
    processed_data = []
    for idx, example in enumerate(json_data):
        processed_data.append(process_fn(example, idx))

    # Create dataset
    dataset = Dataset.from_list(processed_data)
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet
    dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    print(f"Saved {len(processed_data)} items to {os.path.join(local_dir, 'test.parquet')}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied data to HDFS: {hdfs_dir}")