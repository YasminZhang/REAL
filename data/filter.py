import pandas as pd
import numpy as np

df = pd.read_parquet('./data/train.parquet')
gt = [x.get("ground_truth") for x in df['reward_model'].tolist()]
df_filtered = df[np.array(gt) != ""].reset_index(drop=True)
df_filtered.to_parquet('./data/train_filtered.parquet', index=True)
