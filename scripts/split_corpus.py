#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

def split(df, seed=42):
    idx = np.arange(len(df))
    np.random.seed(seed); np.random.shuffle(idx)
    n = len(df)
    train, val, test = np.split(idx, [int(.7*n), int(.85*n)])
    return df.iloc[train], df.iloc[val], df.iloc[test]

for csv_name in ["classification.csv", "regression.csv"]:
    df = pd.read_csv(csv_name)
    tr, va, te = split(df)
    root = os.path.splitext(csv_name)[0]
    tr.to_csv(f"{root}_train.csv", index=False)
    va.to_csv(f"{root}_val.csv",   index=False)
    te.to_csv(f"{root}_test.csv",  index=False)
    print(f"✔️ {csv_name}: {len(tr)}/{len(va)}/{len(te)} train/val/test rows")
