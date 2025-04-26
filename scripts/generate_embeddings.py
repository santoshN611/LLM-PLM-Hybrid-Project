#!/usr/bin/env python3
import os
import warnings

import torch
import numpy as np
import pandas as pd
import umap          # pip install umap-learn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

print("🔄 Starting embedding generation…")

# ── Device setup ────────────────────────────────────────────────
USE_CUDA = torch.cuda.is_available()
print(f"⚙️ CUDA available? {'Yes' if USE_CUDA else 'No'}")
DEVICE = "cuda" if USE_CUDA else "cpu"

# ── Load two independent ESM-2 instances ─────────────────────────
print("📦 Loading ESM-2 for GPU…")
esm_gpu, alphabet = torch.hub.load(
    "facebookresearch/esm:main", "esm2_t6_8M_UR50D"
)
esm_gpu = esm_gpu.eval().to("cuda")  # separate copy for GPU
print("📦 Loading ESM-2 for CPU…")
esm_cpu, _ = torch.hub.load(
    "facebookresearch/esm:main", "esm2_t6_8M_UR50D"
)
esm_cpu = esm_cpu.eval().to("cpu")   # separate copy for CPU
batch_converter = alphabet.get_batch_converter()
print("✅ ESM-2 ready on both devices")

# ── Chunking parameters ────────────────────────────────────────
# ESM-2’s max_position_embeddings defaults to 1026, so max input ≈1022 
MAX_SEQ_LEN = 2500 # testing large sequence size to handle large lengths, default is 1022
CHUNK_STEP  = 1022
REPORT_EVERY = 1000

def embed_sequence(seq: str) -> np.ndarray:
    """
    Embed an arbitrarily long protein by slicing into ≤1022-aa windows,
    running each window on a single device (GPU first, then CPU on OOM),
    and averaging the window embeddings.
    """
    windows = [seq[i:i + MAX_SEQ_LEN] for i in range(0, len(seq), CHUNK_STEP)]
    embs = []

    for w in windows:
        # tokenize once per window
        _, _, toks = batch_converter([("id", w)])
        # try GPU first, then CPU if OOM
        for dev, model in (("cuda", esm_gpu), ("cpu", esm_cpu)):
            if dev == "cuda" and not USE_CUDA:
                continue
            try:
                toks_dev = toks.to(dev)
                with torch.no_grad():
                    reps = model(toks_dev, repr_layers=[6])["representations"][6]
                emb = reps.mean(1).cpu().numpy()
                embs.append(emb)
                break
            except RuntimeError as e:
                # on GPU OOM, clear cache and retry on CPU
                if dev == "cuda" and "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"⚠️ GPU OOM on window len {len(w)}, retrying on CPU…")
                    continue
                raise

    # average window embeddings into one vector
    return np.mean(embs, axis=0)

def process_split(split_name: str):
    """
    📥 Read <split_name>.csv, embed all sequences, and save:
      embeddings/{split_name}.npz containing arrays X (N×D), y, meta
    """
    print(f"\n📥 Loading {split_name}.csv…")
    df = pd.read_csv(f"{split_name}.csv")
    seqs = df["sequence"].tolist()
    accessions = df["accession"].tolist()
    labels = df.iloc[:, 2].values

    all_embs = []
    for i, seq in enumerate(tqdm(seqs, total=len(seqs)), start=1):
        emb = embed_sequence(seq)
        all_embs.append(emb)
        if i % REPORT_EVERY == 0 or i == len(seqs):
            print(f"   ↳ Completed {i}/{len(seqs)} embeddings")

    X    = np.stack(all_embs)                  # shape: (N, emb_dim)
    y    = labels
    meta = np.array(accessions, dtype=object)

    Path("embeddings").mkdir(exist_ok=True)
    np.savez_compressed(f"embeddings/{split_name}.npz", X=X, y=y, meta=meta)
    print(f"✅ Saved embeddings+meta → embeddings/{split_name}.npz (shape={X.shape})")

if __name__ == "__main__":
    # ── Process all splits ────────────────────────────────────────
    splits = [
        "classification_train", "classification_val", "classification_test",
        "regression_train",     "regression_val",     "regression_test"
    ]
    for sp in splits:
        process_split(sp)

    print("\n🎉 All embeddings generated!")

    # ── Labeled UMAP on train split ──────────────────────────────
    print("🗺️ Computing UMAP on classification_train embeddings…")
    data = np.load("embeddings/classification_train.npz", allow_pickle=True)
    X = np.squeeze(data["X"])
    y = data["y"]

    reducer = umap.UMAP(n_components=2, random_state=42)
    Z = reducer.fit_transform(X)

    plt.figure(figsize=(8,8))
    scatter = plt.scatter(Z[:,0], Z[:,1], c=y, s=5, alpha=0.7)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("UMAP of ESM-2 Embeddings\nColored by Existence Level")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Existence Level")
    out_path = "train_umap_labeled.png"
    plt.savefig(out_path)
    print(f"📊 Saved labeled UMAP plot → {out_path}")
    plt.close()
