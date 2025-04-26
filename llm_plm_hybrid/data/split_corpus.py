#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure the data/ directory exists
data_dir = Path(__file__).resolve().parent
os.makedirs(data_dir, exist_ok=True)


def split(df, seed=42):
    idx = np.arange(len(df))
    np.random.seed(seed)
    np.random.shuffle(idx)
    n = len(df)
    train, val, test = np.split(idx, [int(.7 * n), int(.85 * n)])
    return df.iloc[train], df.iloc[val], df.iloc[test]

if __name__ == "__main__":
    for csv_name in ["classification.csv", "regression.csv"]:
        in_path = Path(__file__).resolve().parent / csv_name
        df = pd.read_csv(in_path)

        tr, va, te = split(df)
        root = csv_name.rsplit(".",1)[0]

        tr.to_csv(data_dir / f"{root}_train.csv", index=False)
        va.to_csv(data_dir / f"{root}_val.csv",   index=False)
        te.to_csv(data_dir / f"{root}_test.csv",  index=False)

        print(f"✔️ {csv_name}: {len(tr)}/{len(va)}/{len(te)} train/val/test rows")
