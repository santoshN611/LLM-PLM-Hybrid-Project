# embedding_utils.py

import numpy as np
import matplotlib.pyplot as plt
import umap  # pip install umap-learn
from sklearn.manifold import TSNE
import torch
import torch.nn as nn

def plot_embedding(X: np.ndarray, method='umap', save_path='embedding.png'):
    """
    🗺️ Reduce `X` (n_samples × emb_dim) to 2D via UMAP or t-SNE and save a scatter plot.
    Automatically squeezes any singleton dimensions so UMAP/TSNE always sees a 2D array.
    """
    # Remove any singleton dimensions: e.g. (N,1,D) → (N,D)
    X2 = np.squeeze(X)
    if X2.ndim != 2:
        raise ValueError(f"❌ plot_embedding: expected 2D array after squeeze, got shape {X2.shape}")

    print(f"🔍 [plot_embedding] Generating {method.upper()} projection for shape {X2.shape}...")
    if method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42)

    Z = reducer.fit_transform(X2)
    plt.figure(figsize=(8, 8))
    plt.scatter(Z[:, 0], Z[:, 1], s=5, alpha=0.6)
    plt.title(f'Embeddings visualized via {method.upper()}')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ [plot_embedding] Saved plot to {save_path}")

class SparseAutoencoder(nn.Module):
    """
    🏗️ A simple sparse autoencoder for interpretability:
    - encoder: emb_dim -> bottleneck_dim
    - decoder: bottleneck_dim -> emb_dim
    - sparsity enforced via L1 penalty on bottleneck activations
    """
    def __init__(self, emb_dim, bottleneck_dim=64, sparsity_coef=1e-3):
        super().__init__()
        self.encoder = nn.Linear(emb_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, emb_dim)
        self.sparsity_coef = sparsity_coef

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_rec = self.decoder(z)
        return x_rec, z

    def loss(self, x, x_rec, z):
        rec_loss = nn.MSELoss()(x_rec, x)
        sparse_loss = self.sparsity_coef * torch.norm(z, 1)
        return rec_loss + sparse_loss

def train_autoencoder(X: np.ndarray, epochs=20, lr=1e-3):
    """
    🚀 Train the sparse autoencoder on `X` and return the trained model.
    """
    model = SparseAutoencoder(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.from_numpy(X).float()
    print(f"🛠️ [AE] Starting training for {epochs} epochs on shape {X.shape}...")
    for ep in range(1, epochs+1):
        model.train()
        x_rec, z = model(data)
        loss = model.loss(data, x_rec, z)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0 or ep == epochs:
            print(f"🏷️ [AE] Epoch {ep}/{epochs}, Loss = {loss.item():.4f}")
    print("🎉 [AE] Autoencoder training complete!")
    return model

def linear_probe(Z: np.ndarray, labels: np.ndarray):
    """
    🔎 Fit a logistic regression on each embedding dimension Z[:, i] vs `labels`
    to gauge interpretability of each latent dimension.
    Returns an array of accuracy scores per dimension.
    """
    from sklearn.linear_model import LogisticRegression
    scores = []
    print(f"🧪 [probe] Running linear probes on shape {Z.shape}...")
    for i in range(Z.shape[1]):
        clf = LogisticRegression(max_iter=200)
        s = clf.fit(Z[:, [i]], labels).score(Z[:, [i]], labels)
        scores.append(s)
    print("✅ [probe] Linear probing complete!")
    return np.array(scores)
