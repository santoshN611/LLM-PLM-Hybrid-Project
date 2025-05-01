import numpy as np
import faiss
from pathlib import Path

def build_faiss_index():
    base     = Path(__file__).resolve().parent.parent
    emb_dir  = base / "embeddings"
    emb_file = emb_dir / "combined_train.npz"
    idx_file = emb_dir / "combined_train.index"
    print(f"🔄 Loading {emb_file}…")
    data = np.load(emb_file, allow_pickle=True)
    X = data["X"]

    # reshape
    if X.ndim == 3 and X.shape[1] == 1:
        print(f"ℹ️ Squeezing singleton dimension: {X.shape} →", end=" ")
        X = X.squeeze(1)
        print(f"{X.shape}")
    elif X.ndim > 2:
        n = X.shape[0]
        X = X.reshape(n, -1)
        print(f"ℹ️ Reshaped embeddings to 2D: now {X.shape}")

    print(f"ℹ️ Loaded {X.shape} (n_vectors, dim)")

    print("⚙️ Normalizing & building index…")
    X = X.astype("float32")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    Path(idx_file).parent.mkdir(exist_ok=True)
    faiss.write_index(index, str(idx_file))
    print(f"✅ Index saved to {idx_file}")

if __name__=="__main__":
    build_faiss_index()
