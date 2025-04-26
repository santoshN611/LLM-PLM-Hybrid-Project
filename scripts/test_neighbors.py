#!/usr/bin/env python3
import numpy as np
import pandas as pd
import faiss
import torch
from pathlib import Path

# -----------------------------------------------------------------------------
# 1) Setup ESM-2 embedder (same as in generate_embeddings.py)
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
esm_model, alphabet = torch.hub.load(
    "facebookresearch/esm:main",
    "esm2_t6_8M_UR50D"
)
esm_model.eval().to(DEVICE)
batch_converter = alphabet.get_batch_converter()

MAX_SEQ_LEN = 1022
CHUNK_SIZE  = 1022

def embed_sequence(seq: str) -> np.ndarray:
    """Embed a protein sequence with chunking + GPU→CPU fallback."""
    if len(seq) <= MAX_SEQ_LEN:
        chunks = [seq]
    else:
        chunks = [seq[i:i+CHUNK_SIZE] for i in range(0, len(seq), CHUNK_SIZE)]
    all_embs = []
    for chunk in chunks:
        for device in (DEVICE, "cpu"):
            try:
                _, _, toks = batch_converter([("id", chunk)])
                toks = toks.to(device)
                model = esm_model if device == DEVICE else esm_model.cpu()
                with torch.no_grad():
                    reps = model(toks, repr_layers=[6])["representations"][6]
                emb = reps.mean(1).cpu().numpy()
                all_embs.append(emb)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device == DEVICE:
                    torch.cuda.empty_cache()
                    print(f"⚠️ OOM on chunk len {len(chunk)}, retrying CPU…")
                    continue
                else:
                    raise
    return np.mean(all_embs, axis=0)

# -----------------------------------------------------------------------------
# 2) Load your saved embeddings + metadata
# -----------------------------------------------------------------------------
data = np.load("embeddings/classification_train.npz", allow_pickle=True)
X = np.squeeze(data["X"])   # shape (N, emb_dim)
meta = data["meta"]         # list of accessions

# Normalize once for cosine sim
Xf = X.astype("float32")
faiss.normalize_L2(Xf)

# -----------------------------------------------------------------------------
# 3) Build FAISS index in memory
# -----------------------------------------------------------------------------
index = faiss.IndexFlatIP(Xf.shape[1])
index.add(Xf)
print(f"✅ Built FAISS index with {Xf.shape[0]} vectors of dim {Xf.shape[1]}")

# -----------------------------------------------------------------------------
# 4) Load sequences from CSV for lookup
# -----------------------------------------------------------------------------
df = pd.read_csv("classification_train.csv")
seq_map = df.set_index("accession")["sequence"].to_dict()

# -----------------------------------------------------------------------------
# 5) Define & run your queries
# -----------------------------------------------------------------------------
queries = [
    "MVHFAELVK",
    "ACDEFGHIKL",
    "MLLTEQFK"      # feel free to add more sequences here!
]

print("\n🔍 Nearest neighbors for each query:\n")
for seq in queries:
    emb = embed_sequence(seq)
    embf = emb.astype("float32").reshape(1, -1)
    faiss.normalize_L2(embf)
    D, I = index.search(embf, 5)
    neighs = [meta[i] for i in I[0]]

    print(f"▶ Query sequence: {seq}")
    for acc in neighs:
        nbr_seq = seq_map.get(acc, "<missing>")
        print(f"   • {acc} → {nbr_seq}")
    print()

# EOF
