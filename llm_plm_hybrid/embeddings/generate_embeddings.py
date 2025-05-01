import os
import warnings

import torch
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

print("🔄 Starting embedding generation…")

USE_CUDA = torch.cuda.is_available()
print(f"⚙️ CUDA available? {'Yes' if USE_CUDA else 'No'}")
DEVICE = "cuda" if USE_CUDA else "cpu"

# in case of GPU out of memory problems
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

# params
MAX_SEQ_LEN = 2500   # large max to handle long seqs
CHUNK_STEP  = 1022
REPORT_EVERY = 1000

# paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMB_DIR  = Path(__file__).resolve().parent
VIS_DIR  = EMB_DIR.parent / "visualizations"

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

def embed_sequence(seq: str) -> np.ndarray:
    windows = [seq[i:i + MAX_SEQ_LEN] for i in range(0, len(seq), CHUNK_STEP)]
    embs = []

    for w in windows:
        _, _, toks = batch_converter([("id", w)])
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
                if dev == "cuda" and "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"⚠️ GPU OOM on window len {len(w)}, retrying CPU…")
                    continue
                raise

    return np.mean(embs, axis=0)

def process_split(split_name: str):
    csv_path = DATA_DIR / f"{split_name}.csv"
    print(f"\n📥 Loading {csv_path}…")
    df = pd.read_csv(csv_path)
    seqs = df["sequence"].tolist()
    accessions = df["accession"].tolist()
    labels = df.iloc[:, 2].values

    all_embs = []
    for i, seq in enumerate(tqdm(seqs, total=len(seqs)), start=1):
        emb = embed_sequence(seq)
        all_embs.append(emb)
        if i % REPORT_EVERY == 0 or i == len(seqs):
            print(f"   ↳ Completed {i}/{len(seqs)} embeddings")

    X    = np.stack(all_embs)
    y    = labels
    meta = np.array(accessions, dtype=object)

    out_path = EMB_DIR / f"{split_name}.npz"
    np.savez_compressed(out_path, X=X, y=y, meta=meta)
    print(f"✅ Saved embeddings+meta → {out_path} (shape={X.shape})")

if __name__ == "__main__":
    splits = [
        "classification_train", "classification_val", "classification_test",
        "regression_train",     "regression_val",     "regression_test"
    ]
    for sp in splits:
        process_split(sp)

    print("\n🎉 All embeddings generated!")

    print("🔗 Combining classification_train + regression_train -> combined_train.npz…")
    class_npz = EMB_DIR / "classification_train.npz"
    reg_npz   = EMB_DIR / "regression_train.npz"

    # load both
    cdata = np.load(class_npz, allow_pickle=True)
    rdata = np.load(reg_npz,   allow_pickle=True)

    # stack embeddings (X), labels (y) and accessions/meta
    X_comb = np.concatenate([cdata["X"], rdata["X"]], axis=0)
    y_comb = np.concatenate([cdata["y"], rdata["y"]], axis=0)
    m_comb = np.concatenate([cdata["meta"], rdata["meta"]], axis=0)

    out_comb = EMB_DIR / "combined_train.npz"
    np.savez_compressed(out_comb, X=X_comb, y=y_comb, meta=m_comb)
    print(f"✅ Saved combined embeddings+meta → {out_comb} (shape={X_comb.shape})")

    # UMAP
    print("🗺️ Computing UMAP on combined training embeddings…")
    train_npz = EMB_DIR / "combined_train.npz"
    data = np.load(train_npz, allow_pickle=True)
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

    umap_path = VIS_DIR / "train_umap_labeled.png"
    plt.savefig(umap_path)
    print(f"📊 Saved labeled UMAP plot → {umap_path}")
    plt.close()
