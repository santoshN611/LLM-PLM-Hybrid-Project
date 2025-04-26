# retrieval_utils.py

import numpy as np
import faiss
from pathlib import Path
import requests

INDEX_PATH = 'embeddings/classification_train.index'
META_PATH  = 'embeddings/classification_train.meta.npy'

def build_index(
    emb_file='embeddings/classification_train.npz',
    idx_file=INDEX_PATH,
    meta_file=META_PATH
):
    data = np.load(emb_file, allow_pickle=True)
    X = data['X']

    # — FORCE to 2D —
    X = np.squeeze(X)
    if X.ndim > 2:
        n = X.shape[0]
        X = X.reshape(n, -1)

    print(f"🔄 [build_index] Final embedding shape: {X.shape}")

    X = X.astype('float32')
    faiss.normalize_L2(X)

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    Path(idx_file).parent.mkdir(exist_ok=True)
    faiss.write_index(index, idx_file)
    print(f"✅ [build_index] Saved FAISS index to {idx_file}")

    if 'meta' in data:
        np.save(meta_file, data['meta'])
        print(f"📦 [build_index] Saved metadata to {meta_file}")
    else:
        print(f"⚠️ [build_index] Warning: no 'meta' array found in {emb_file}")

def load_index(idx_file=INDEX_PATH, meta_file=META_PATH):
    print(f"🔍 [load_index] Loading index from {idx_file}")
    index = faiss.read_index(idx_file)
    meta  = np.load(meta_file, allow_pickle=True)
    print(f"✅ [load_index] Loaded {len(meta)} accessions from {meta_file}")
    return index, meta

def search_neighbors(query_emb: np.ndarray, k=5):
    index, meta = load_index()
    q = query_emb.astype('float32').reshape(1, -1)
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    neighbors = [meta[i] for i in I[0]]
    print(f"🎯 [search_neighbors] Top {k} neighbors: {neighbors}")
    return neighbors

def build_context_block(accessions):
    print(f"📝 [build_context_block] Fetching names for accessions: {accessions}")
    lines = []
    for acc in accessions:
        try:
            r = requests.get(f'https://rest.uniprot.org/uniprotkb/{acc}.json', timeout=5)
            r.raise_for_status()
            d = r.json()
            name = (d.get('proteinDescription', {})
                     .get('recommendedName', {})
                     .get('fullName', {})
                     .get('value', acc))
        except Exception:
            name = acc
        lines.append(f"- **{acc}**: {name}")
    context = "Nearby proteins (by embedding similarity):\n" + "\n".join(lines)
    print(f"✅ [build_context_block] Context built")
    return context

if __name__ == "__main__":
    # Default entry point
    print("🚀 Running build_index via retrieval_utils.py")
    build_index(
        emb_file='embeddings/classification_train.npz',
        idx_file='embeddings/classification_train.index',
        meta_file='embeddings/classification_train.meta.npy'
    )
