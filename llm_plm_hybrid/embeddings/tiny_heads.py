#!/usr/bin/env python3
import torch
import torch.nn as nn
from pathlib import Path

# ── Directory where the pretrained head weights live ────────────────
# Assumes you're running from the project root, so that
#   <project_root>/tiny_heads/pe.pt  and  <project_root>/tiny_heads/ptm.pt
# exist.
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


def load_heads(device="cpu"):
    """
    Load saved tiny-head weights from HEADS_DIR.
    Returns:
      - pe:  PEHead, set to eval() on `device`
      - ptm: PTMHead, set to eval() on `device`
    """
    # ── Load Protein-Existence head ────────────────────────────────────
    pe_state = torch.load(HEADS_DIR / "pe.pt", map_location=device)
    emb_dim = pe_state["layers.0.weight"].shape[1]
    pe = PEHead(emb_dim)
    pe.load_state_dict(pe_state)
    pe.to(device).eval()

    # ── Load PTM-Count head ────────────────────────────────────────────
    ptm_state = torch.load(HEADS_DIR / "ptm.pt", map_location=device)
    emb_dim   = ptm_state["layers.0.weight"].shape[1]
    ptm = PTMHead(emb_dim)
    ptm.load_state_dict(ptm_state)
    ptm.to(device).eval()

    return pe, ptm
