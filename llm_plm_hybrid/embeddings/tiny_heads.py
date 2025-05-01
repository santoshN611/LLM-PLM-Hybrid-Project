import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from llm_plm_hybrid.embeddings.generate_embeddings import embed_sequence



HEADS_DIR = Path.cwd() / "tiny_heads"

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
    
    def predict(self, seq: str, device="cpu") -> int:
        # 1) get raw embedding
        emb = embed_sequence(seq)
        if emb.ndim == 2:
            emb = emb.mean(0)
        x = torch.from_numpy(emb).float().unsqueeze(0).to(device)
        # 2) forward
        self.eval()
        with torch.no_grad():
            logits = self(x)                 # shape (1,5)
            pred = logits.argmax(dim=-1).item()
        # 3) convert 0–4 back to 1–5
        return pred + 1


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
    
    def predict(self, seq: str, device="cpu") -> float:
        emb = embed_sequence(seq)
        if emb.ndim == 2:
            emb = emb.mean(0)
        x = torch.from_numpy(emb).float().unsqueeze(0).to(device)
        self.eval()
        with torch.no_grad():
            out = self(x).squeeze().item()   # this is log1p(ptm_count)
        return float(np.expm1(out))          # invert the log1p


def load_heads(device="cpu"):
    # protein existence level classification head
    pe_state = torch.load(HEADS_DIR / "pe.pt", map_location=device)
    emb_dim = pe_state["layers.0.weight"].shape[1]
    pe = PEHead(emb_dim)
    pe.load_state_dict(pe_state)
    pe.to(device).eval()

    # ptm regression head
    ptm_state = torch.load(HEADS_DIR / "ptm.pt", map_location=device)
    emb_dim   = ptm_state["layers.0.weight"].shape[1]
    ptm = PTMHead(emb_dim)
    ptm.load_state_dict(ptm_state)
    ptm.to(device).eval()

    return pe, ptm
