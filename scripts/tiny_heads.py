#!/usr/bin/env python3
import torch, torch.nn as nn
from pathlib import Path

class PEHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(),
            nn.Linear(128, 5)
        )
    def forward(self, x):
        return self.layers(x)

class PTMHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.layers(x)

def load_heads(device="cpu"):
    heads_dir = Path("tiny_heads")
    pe_sd = torch.load(heads_dir/"pe.pt", map_location=device)
    emb_dim = pe_sd["layers.0.weight"].shape[1]
    pe = PEHead(emb_dim); pe.load_state_dict(pe_sd)
    pe.to(device).eval()
    ptm_sd = torch.load(heads_dir/"ptm.pt", map_location=device)
    emb_dim = ptm_sd["layers.0.weight"].shape[1]
    ptm = PTMHead(emb_dim); ptm.load_state_dict(ptm_sd)
    ptm.to(device).eval()
    return pe, ptm
