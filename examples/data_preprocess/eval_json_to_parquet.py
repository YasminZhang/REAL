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
Convert JSON evaluation data (Flask, MT-Bench, Vicuna) to parquet format compatible with the existing data processing pipeline
"""

import argparse
import json
import os
import numpy as np

import datasets
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


def determine_data_source(json_file_path):
    """Determine the data source based on file path"""
    filename = os.path.basename(json_file_path).lower()
    if 'flask' in filename:
        return 'flask_evaluation'
    elif 'mt_bench' in filename:
        return 'mt_bench_evaluation'
    elif 'vicuna' in filename:
        return 'vicuna_evaluation'
    else:
        return 'unknown_evaluation'


def determine_ability(data_source, example):
    """Determine the ability based on data source and example content"""
    if data_source == 'flask_evaluation':
        return example.get('criteria', 'general_evaluation')
    elif data_source == 'mt_bench_evaluation':
        return example.get('category', 'general_evaluation')
    elif data_source == 'vicuna_evaluation':
        return example.get('category', 'general_evaluation')
    else:
        return 'general_evaluation'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=True, help="Path to the input JSON file")
    parser.add_argument("--local_dir", default=None, help="Local directory to save the parquet file")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--use_human_score", action="store_true", help="Use human_score instead of gpt4_score for Flask data")

    args = parser.parse_args()

    # Auto-determine local_dir if not provided
    if args.local_dir is None:
        filename = os.path.basename(args.json_file).replace('.json', '')
        args.local_dir = f"./data/{filename}_for_sft"

    # Load JSON data
    print(f"Loading JSON data from {args.json_file}...", flush=True)
    with open(args.json_file, 'r') as f:
        json_data = json.load(f)

    # Determine data source
    data_source = determine_data_source(args.json_file)
    print(f"Detected data source: {data_source}")

    # let's better not add this instruction following the SFT in TRACT
    instruction_following = "Please think step by step and then output the final score with 'So the overall score is (score)'."

    # Transform data to the required format
    def process_fn(example, idx):
        question = example["instruction"]
        if 'base' in args.local_dir: 
            question = question + " " + instruction_following
        else:
            question = question

        # Determine which score to use
        if data_source == 'flask_evaluation' and args.use_human_score and 'human_score' in example:
            score_list = example["human_score"]
        else:
            score_list = example["gpt4_score"]

        # Handle scores (convert list to mean if it's a list)
        if isinstance(score_list, list):
            solution = float(np.mean(np.array(score_list)))  # Take the mean if it's a list
        else:
            solution = float(score_list)

        # Get feedback if available
        feedback = ""
        if 'gpt4_feedback' in example:
            if isinstance(example['gpt4_feedback'], list):
                feedback = example['gpt4_feedback'][0] if example['gpt4_feedback'] else ""
            else:
                feedback = example['gpt4_feedback']

        # Determine ability
        ability = determine_ability(data_source, example)
        
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": ability,
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": "test", 
                "index": idx, 
                "reward_design": True,
                "original_idx": example["idx"],
                "response_source": example.get("response_source", ""),
                "gpt4_feedback": feedback,
                "original_scores": score_list  # Keep original scores for reference
            },
        }

        # Add source-specific extra info
        if data_source == 'flask_evaluation':
            data["extra_info"]["criteria"] = example.get("criteria", "")
            if 'human_score' in example:
                data["extra_info"]["human_score"] = example["human_score"]
        elif data_source in ['mt_bench_evaluation', 'vicuna_evaluation']:
            data["extra_info"]["category"] = example.get("category", "")

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
    output_file = os.path.join(local_dir, "test.parquet")
    dataset.to_parquet(output_file)
    print(f"Saved {len(processed_data)} items to {output_file}")

    # Print some statistics
    scores = [item['reward_model']['ground_truth'] for item in processed_data]
    print(f"Score statistics:")
    print(f"  Mean: {np.mean(scores):.2f}")
    print(f"  Min: {np.min(scores):.2f}")
    print(f"  Max: {np.max(scores):.2f}")
    print(f"  Std: {np.std(scores):.2f}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied data to HDFS: {hdfs_dir}")