#!/usr/bin/env python3
import numpy as np
import faiss
from pathlib import Path

def build_faiss_index(
    embeddings_file: str = "embeddings/classification_train.npz",
    index_file:      str = "embeddings/classification_train.index"
):
    print(f"🔄 Loading embeddings from {embeddings_file}…")
    data = np.load(embeddings_file)
    X    = data["X"].astype("float32")
    print(f"ℹ️  Loaded {X.shape[0]} vectors of dim {X.shape[1]}")

    print("⚙️  Normalizing and building FAISS index…")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    Path(index_file).parent.mkdir(exist_ok=True, parents=True)
    faiss.write_index(index, index_file)
    print(f"✅ FAISS index saved to {index_file} ({index.ntotal} vectors)")

if __name__ == "__main__":
    build_faiss_index()
