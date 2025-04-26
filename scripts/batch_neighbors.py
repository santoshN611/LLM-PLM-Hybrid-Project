#!/usr/bin/env python3
import numpy as np
import pandas as pd
import faiss
import torch
from pathlib import Path

# 1) Load ESM-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
esm_model, alphabet = torch.hub.load("facebookresearch/esm:main","esm2_t6_8M_UR50D")
esm_model.eval().to(DEVICE)
batch_converter = alphabet.get_batch_converter()
MAX_LEN, CHUNK = 1022, 1022

def embed_sequence(seq):
    chunks = [seq[i:i+CHUNK] for i in range(0, len(seq), CHUNK)]
    embs=[]
    for c in chunks:
        for dev in (DEVICE,"cpu"):
            try:
                _,_,toks = batch_converter([("id",c)])
                toks = toks.to(dev)
                model = esm_model if dev==DEVICE else esm_model.cpu()
                with torch.no_grad():
                    r = model(toks,repr_layers=[6])["representations"][6]
                emb = r.mean(1).cpu().numpy()
                embs.append(emb); break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and dev==DEVICE:
                    torch.cuda.empty_cache(); continue
                else: raise
    return np.mean(embs,axis=0)

# 2) Load embeddings + FAISS
data = np.load("embeddings/classification_train.npz",allow_pickle=True)
X = np.squeeze(data["X"]).astype("float32")
faiss.normalize_L2(X)
index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
meta = data["meta"]

# 3) Load accession→sequence map
df = pd.read_csv("classification_train.csv")
seq_map = df.set_index("accession")["sequence"].to_dict()

# 4) Define your batch of queries
queries = [
    "MVHFAELVK", 
    "ACDEFGHIKL", 
    "GGLVPRGSH", 
    # … add as many as you like …
]

# 5) Retrieve & display
results = []
for q in queries:
    emb = embed_sequence(q).astype("float32").reshape(1,-1)
    faiss.normalize_L2(emb)
    _,I = index.search(emb, 5)
    neighs = meta[I[0]]
    for acc in neighs:
        results.append({
            "query":       q,
            "neighbor_acc":acc,
            "neighbor_seq":seq_map.get(acc,"<not found>")
        })

# 6) Output to screen or file
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv("batch_neighbors.csv", index=False)
