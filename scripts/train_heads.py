#!/usr/bin/env python3
import torch, torch.nn as nn, numpy as np
from pathlib import Path

print("🔄 Starting tiny-heads training…")

# ─── Model definitions ──────────────────────────────────────────────────
class PEHead(nn.Module):
<<<<<<< HEAD
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(),
=======
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(),
>>>>>>> 0a2edc19e048b73bb4e4255270613a728ee264b6
            nn.Linear(128, 5)
        )
    def forward(self, x):
        return self.layers(x)

class PTMHead(nn.Module):
<<<<<<< HEAD
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(),
=======
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(),
>>>>>>> 0a2edc19e048b73bb4e4255270613a728ee264b6
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.layers(x)

# ─── Data loader ────────────────────────────────────────────────────────
def load_data(prefix):
    print(f"📥 Loading embeddings/{prefix}.npz…")
    data = np.load(f'embeddings/{prefix}.npz')
    return data['X'], data['y']

# ─── Training loop ──────────────────────────────────────────────────────
def train(model, X_train, y_train, X_val, y_val, criterion, optimizer, epochs=10):
    best_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
<<<<<<< HEAD

        inputs = torch.from_numpy(X_train).float()

        # classification vs regression target workup
        if isinstance(criterion, nn.CrossEntropyLoss):
            # classification: LongTensor [N]
            targets = torch.from_numpy(y_train).long()
        else:
            # regression: FloatTensor [N,1]
            targets = torch.from_numpy(y_train).float().unsqueeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
=======
        inputs = torch.from_numpy(X_train).float()
        targets = torch.from_numpy(y_train)
        if targets.dtype != torch.long:
            targets = targets.float()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
>>>>>>> 0a2edc19e048b73bb4e4255270613a728ee264b6
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
<<<<<<< HEAD
            val_inputs = torch.from_numpy(X_val).float()
            if isinstance(criterion, nn.CrossEntropyLoss):
                val_targets = torch.from_numpy(y_val).long()
            else:
                val_targets = torch.from_numpy(y_val).float().unsqueeze(1)

            val_out  = model(val_inputs)
            val_loss = criterion(val_out, val_targets)

        print(f"🏷️ Epoch {epoch:02d}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")
=======
            val_out = model(torch.from_numpy(X_val).float())
            val_loss = criterion(val_out.squeeze(), torch.from_numpy(y_val).float())

        print(f"🏷️  Epoch {epoch:02d}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")
>>>>>>> 0a2edc19e048b73bb4e4255270613a728ee264b6

        if val_loss < best_loss:
            best_loss = val_loss
            Path('tiny_heads').mkdir(exist_ok=True)
<<<<<<< HEAD
            name = 'pe' if outputs.shape[-1] == 5 else 'ptm'
=======
            name = 'pe' if outputs.shape[-1]==5 else 'ptm'
>>>>>>> 0a2edc19e048b73bb4e4255270613a728ee264b6
            torch.save(model.state_dict(), f'tiny_heads/{name}.pt')
            print(f"✅ Saved best {name} head (val_loss={best_loss:.4f})")

    print("🎉 Head training complete!")

<<<<<<< HEAD
# ─── Main script ────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Protein-Existence (classification) head
    X_tr, y_tr = load_data('classification_train')
    X_va, y_va = load_data('classification_val')
    # shift labels from 1–5 → 0–4
    y_tr = y_tr - 1
    y_va = y_va - 1

    emb_dim = X_tr.shape[1]
    print(f"ℹ️ Detected embedding dimension: {emb_dim}")

    pe_model = PEHead(emb_dim)
    print("🔄 Training Protein-Existence head…")
    train(
        pe_model,
        X_tr, y_tr,
        X_va, y_va,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(pe_model.parameters())
    )

    # PTM-Count (regression) head
    X_tr, y_tr = load_data('regression_train')
    X_va, y_va = load_data('regression_val')
    # ⚙️ log-transform targets to tame heavy tail
    y_tr = np.log1p(y_tr)
    y_va = np.log1p(y_va)

    ptm_model = PTMHead(emb_dim)
    print("🔄 Training PTM-Count head…")
    train(
        ptm_model,
        X_tr, y_tr,
        X_va, y_va,
        nn.SmoothL1Loss(),
        torch.optim.Adam(ptm_model.parameters(), lr=1e-3, weight_decay=1e-5)
    )
=======
if __name__ == '__main__':
    # Protein-existence head
    X_tr, y_tr = load_data('classification_train')
    X_va, y_va = load_data('classification_val')
    pe_model = PEHead()
    print("🔄 Training Protein-Existence head…")
    train(pe_model, X_tr, y_tr, X_va, y_va, nn.CrossEntropyLoss(), torch.optim.Adam(pe_model.parameters()))

    # PTM-count head
    X_tr, y_tr = load_data('regression_train')
    X_va, y_va = load_data('regression_val')
    ptm_model = PTMHead()
    print("🔄 Training PTM-Count head…")
    train(ptm_model, X_tr, y_tr, X_va, y_va, nn.MSELoss(), torch.optim.Adam(ptm_model.parameters()))
>>>>>>> 0a2edc19e048b73bb4e4255270613a728ee264b6
