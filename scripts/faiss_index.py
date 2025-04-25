#!/usr/bin/env python3
import numpy as np
import faiss
from pathlib import Path

def build_faiss_index(
    emb_file="embeddings/classification_train.npz",
    idx_file="embeddings/classification_train.index"
):
    print(f"🔄 Loading {emb_file}…")
    data = np.load(emb_file)
    X = data["X"].astype("float32")
    print(f"ℹ️ Loaded {X.shape}")
    print("⚙️ Normalizing & building index…")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    Path(idx_file).parent.mkdir(exist_ok=True)
    faiss.write_index(index, idx_file)
    print(f"✅ Index saved to {idx_file}")

if __name__=="__main__":
    build_faiss_index()
