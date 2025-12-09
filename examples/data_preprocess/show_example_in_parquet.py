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
            if 'extra_info' in df.columns:
                print("\nExtra info:")
                print(df.sample(1).iloc[0]['extra_info'])
                # print question and answer if exist
                if 'question' in df.sample(1).iloc[0]['extra_info']:
                    print("\nQuestion:")
                    print(df.sample(1).iloc[0]['extra_info']['question'])
                if 'answer' in df.sample(1).iloc[0]['extra_info']:
                    print("\nAnswer:")
                    print(df.sample(1).iloc[0]['extra_info']['answer'])


    except Exception as e:
        print(f"Error reading Parquet file: {e}")

if __name__ == "__main__":
    main()