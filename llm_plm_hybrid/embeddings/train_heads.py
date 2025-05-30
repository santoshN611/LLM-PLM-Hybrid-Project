import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

print("🔄 Starting tiny-heads training…")

# paths
ROOT      = Path(__file__).resolve().parent.parent.parent
EMB_DIR   = ROOT / "llm_plm_hybrid" / "embeddings"
HEADS_DIR = ROOT / "tiny_heads"
HEADS_DIR.mkdir(exist_ok=True)

# data loader
def load_data(prefix):
    npz_path = EMB_DIR / f"{prefix}.npz"
    print(f"📥 Loading {npz_path}…")
    data = np.load(npz_path, allow_pickle=True)
    X = np.squeeze(data["X"])
    y = data["y"]
    print(f"ℹ️ Loaded X shape {X.shape}, y length {len(y)}")
    return X, y

# models
class PEHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )
    def forward(self, x):
        return self.layers(x)

class PTMHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.layers(x)

# training with history recording
def train(model, X_tr, y_tr, X_va, y_va, criterion, optimizer,
          epochs=10, head_name="head"):
    history = {
        "train_loss": [], "val_loss": [],
        # only classification needs acc
        "train_acc": [] if isinstance(criterion, nn.CrossEntropyLoss) else None,
        "val_acc":   [] if isinstance(criterion, nn.CrossEntropyLoss) else None,
    }

    best_loss = float("inf")
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        optimizer.zero_grad()

        inp = torch.from_numpy(X_tr).float()
        tgt = (torch.from_numpy(y_tr).long()
               if isinstance(criterion, nn.CrossEntropyLoss)
               else torch.from_numpy(y_tr).float().unsqueeze(1))

        out = model(inp)
        loss = criterion(out, tgt)
        loss.backward()
        optimizer.step()

        # compute train metrics
        history["train_loss"].append(loss.item())
        if isinstance(criterion, nn.CrossEntropyLoss):
            preds = out.argmax(dim=1).cpu().numpy()
            acc   = (preds == y_tr).mean()
            history["train_acc"].append(acc)

        # validation
        model.eval()
        with torch.no_grad():
            vin = torch.from_numpy(X_va).float()
            vt  = (torch.from_numpy(y_va).long()
                   if isinstance(criterion, nn.CrossEntropyLoss)
                   else torch.from_numpy(y_va).float().unsqueeze(1))
            vout  = model(vin)
            vloss = criterion(vout, vt)

            history["val_loss"].append(vloss.item())
            if isinstance(criterion, nn.CrossEntropyLoss):
                vpreds = vout.argmax(dim=1).cpu().numpy()
                vacc   = (vpreds == y_va).mean()
                history["val_acc"].append(vacc)

        print(f"🏷️ {head_name} Epoch {epoch:03d} | "
              f"train_loss={loss:.4f} "
              + (f"train_acc={acc:.4f} " if history["train_acc"] is not None else "")
              + f"val_loss={vloss:.4f} "
              + (f"val_acc={vacc:.4f}" if history["val_acc"] is not None else ""))

        # checkpoint best
        if vloss < best_loss:
            best_loss = vloss
            ckpt = HEADS_DIR / f"{head_name}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"✅ Saved best {head_name} head (val_loss={best_loss:.4f})")

    # plot
    epochs_range = np.arange(1, epochs+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs_range, history["train_loss"], label="train loss")
    plt.plot(epochs_range, history["val_loss"],   label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{head_name.upper()} Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HEADS_DIR / f"{head_name}_loss.png")
    plt.close()

    if history["train_acc"] is not None:
        plt.figure(figsize=(8,5))
        plt.plot(epochs_range, history["train_acc"], label="train acc")
        plt.plot(epochs_range, history["val_acc"],   label="val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{head_name.upper()} Accuracy Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(HEADS_DIR / f"{head_name}_acc.png")
        plt.close()

    print(f"🎉 {head_name.upper()} training complete!  Plots saved to {HEADS_DIR}")

if __name__ == "__main__":
    X_tr, y_tr = load_data("classification_train")
    X_va, y_va = load_data("classification_val")
    y_tr, y_va = y_tr - 1, y_va - 1  # shift 1–5 to 0–4 due to outputs

    emb_dim = X_tr.shape[1]
    pe_head = PEHead(emb_dim)
    train(
        model=pe_head,
        X_tr=X_tr, y_tr=y_tr,
        X_va=X_va, y_va=y_va,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(pe_head.parameters()),
        epochs=2500,
        head_name="pe"
    )

    X_tr, y_tr = load_data("regression_train")
    X_va, y_va = load_data("regression_val")
    y_tr, y_va = np.log1p(y_tr), np.log1p(y_va)

    ptm_head = PTMHead(emb_dim)
    train(
        model=ptm_head,
        X_tr=X_tr, y_tr=y_tr,
        X_va=X_va, y_va=y_va,
        criterion=nn.SmoothL1Loss(),
        optimizer=torch.optim.Adam(ptm_head.parameters(), lr=1e-3, weight_decay=1e-5),
        epochs=2500,
        head_name="ptm"
    )
