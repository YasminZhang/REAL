# read in a Parquet file and show an example
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Read a Parquet file and show an example.")
    parser.add_argument("parquet_file", help="Path to the Parquet file")
    args = parser.parse_args()

    try:
        df = pd.read_parquet(args.parquet_file)
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        
        if len(df) > 0:
            print("\nRandom sample:")
            print(df.sample(1).iloc[0])
            # print 'prompt'
            print("\nPrompt content:")
            print(df.sample(1).iloc[0]['prompt'][0]['content'])


    except Exception as e:
        print(f"Error reading Parquet file: {e}")

if __name__ == "__main__":
    main()