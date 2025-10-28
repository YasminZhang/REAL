# from train.parquet , sample 1/10 of the data and save to train_sample.parquet

import argparse
import datasets
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Sample 1/10 of data from train.parquet")
    parser.add_argument("--input", default="train.parquet", help="Input parquet file")
    parser.add_argument("--output", default="train_sample.parquet", help="Output parquet file")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of data to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    print(f"Original dataset size: {len(df)} rows")
    
    # Sample fraction of the data
    sampled_df = df.sample(frac=args.fraction, random_state=args.seed)
    
    print(f"Sampled dataset size: {len(sampled_df)} rows ({args.fraction*100:.1f}%)")
    
    # Save to output file
    sampled_df.to_parquet(args.output, index=False)
    
    print(f"Saved sampled data to {args.output}")

if __name__ == "__main__":
    main()
    