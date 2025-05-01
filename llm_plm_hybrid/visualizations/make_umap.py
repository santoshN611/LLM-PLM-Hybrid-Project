import os, warnings
from pathlib import Path
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# from cuml.manifold import UMAP
# import cupy as cp
import numpy as np

def main():

    # UMAP
    VIS_DIR  = Path(__file__).resolve().parent
    print("🗺️ Computing UMAP on combined training embeddings…")
    EMB_DIR  = Path(__file__).resolve().parent.parent / "embeddings"
    train_npz = EMB_DIR / "combined_train.npz"
    data = np.load(train_npz, allow_pickle=True)
    X = np.squeeze(data["X"])
    y = data["y"]

    # X_gpu = cp.asarray(X)


    N_EPOCHS = 200

    pbar = tqdm(total=N_EPOCHS, desc="UMAP epochs")

    def umap_callback(current_epoch, total_epochs):
        pbar.update(1)


    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        verbose=True,
        n_epochs=N_EPOCHS,
        n_jobs=-1
    )
    # reducer = UMAP(
    #     n_neighbors=15,
    #     min_dist=0.1,
    #     n_components=2,
    #     random_state=42
    # )

    Z = reducer.fit_transform(X, callback=umap_callback)
    # Z_gpu = reducer.fit_transform(X_gpu, callback=umap_callback)
    # Z = cp.asnumpy(Z_gpu)

    pbar.close()

    plt.figure(figsize=(8,8))
    scatter = plt.scatter(Z[:,0], Z[:,1], c=y, s=5, alpha=0.7)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("UMAP of ESM-2 Embeddings\nColored by Existence Level")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Existence Level")

    umap_path = VIS_DIR / "train_umap_labeled.png"
    plt.savefig(umap_path)
    print(f"📊 Saved labeled UMAP plot -> {umap_path}")
    plt.close()

if __name__ == "__main__":
    main()
