#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

print("🔄 Starting tiny-heads training…")

class PEHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.layers(x)


class PTMHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


def load_data(prefix):
    """
    📥 Loads embeddings/{prefix}.npz → (X, y), squeezing any singleton dims.
    """
    print(f"📥 Loading embeddings/{prefix}.npz…")
    data = np.load(f"embeddings/{prefix}.npz", allow_pickle=True)
    X = data["X"]
    X = np.squeeze(X)
    if X.ndim != 2:
        raise ValueError(f"❌ load_data: expected 2D array, got shape {X.shape}")
    y = data["y"]
    print(f"ℹ️ Loaded X shape {X.shape}, y length {len(y)}")
    return X, y


def train(model, X_tr, y_tr, X_va, y_va, criterion, optimizer, epochs=5000):
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        inp = torch.from_numpy(X_tr).float()
        if isinstance(criterion, nn.CrossEntropyLoss):
            tgt = torch.from_numpy(y_tr).long()
        else:
            tgt = torch.from_numpy(y_tr).float().unsqueeze(1)

        out = model(inp)
        loss = criterion(out, tgt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vin = torch.from_numpy(X_va).float()
            if isinstance(criterion, nn.CrossEntropyLoss):
                vt = torch.from_numpy(y_va).long()
            else:
                vt = torch.from_numpy(y_va).float().unsqueeze(1)
            vout = model(vin)
            vloss = criterion(vout, vt)

        print(f"🏷️ Epoch {epoch:02d}: train={loss:.4f}, val={vloss:.4f}")
        if vloss < best:
            best = vloss
            Path("tiny_heads").mkdir(exist_ok=True)
            name = "pe" if out.shape[-1] == 5 else "ptm"
            torch.save(model.state_dict(), f"tiny_heads/{name}.pt")
            print(f"✅ Saved best {name} head (loss={best:.4f})")

    print("🎉 Head training complete!")


if __name__ == "__main__":
    # ── Protein-Existence head ────────────────────────────────────
    X_tr, y_tr = load_data("classification_train")
    X_va, y_va = load_data("classification_val")
    y_tr, y_va = y_tr - 1, y_va - 1  # shift to 0-based

    emb_dim = X_tr.shape[1]
    print(f"ℹ️ Embedding dimension: {emb_dim}")

    pe = PEHead(emb_dim)
    train(
        pe, X_tr, y_tr,
        X_va, y_va,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(pe.parameters()),
        epochs=5000
    )

    # ── PTM-Count head ───────────────────────────────────────────
    X_tr, y_tr = load_data("regression_train")
    X_va, y_va = load_data("regression_val")
    y_tr, y_va = np.log1p(y_tr), np.log1p(y_va)

    ptm = PTMHead(emb_dim)
    train(
        ptm, X_tr, y_tr,
        X_va, y_va,
        nn.SmoothL1Loss(),
        torch.optim.Adam(ptm.parameters(), lr=1e-3, weight_decay=1e-5),
        epochs=5000
    )
