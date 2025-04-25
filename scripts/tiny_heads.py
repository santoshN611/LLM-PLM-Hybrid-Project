# tiny_heads.py
"""
Defines the PEHead and PTMHead architectures and a load_heads helper
that instantiates and loads saved head weights for inference.
"""
import torch
import torch.nn as nn
from pathlib import Path

class PEHead(nn.Module):
    """Protein existence classification head."""
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.layers(x)

class PTMHead(nn.Module):
    """Log-scaled PTM count regression head."""
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


def load_heads(device: str):
    """
    Load the trained PEHead and PTMHead weight files from tiny_heads/.
    Returns:
        pe_head (nn.Module), ptm_head (nn.Module)
    """
    heads_dir = Path("tiny_heads")
    # Load PE head
    pe_path = heads_dir / "pe.pt"
    pe_sd = torch.load(pe_path, map_location=device)
    emb_dim = pe_sd['layers.0.weight'].shape[1]
    pe_head = PEHead(emb_dim)
    pe_head.load_state_dict(pe_sd)
    pe_head.to(device).eval()

    # Load PTM head
    ptm_path = heads_dir / "ptm.pt"
    ptm_sd = torch.load(ptm_path, map_location=device)
    emb_dim = ptm_sd['layers.0.weight'].shape[1]
    ptm_head = PTMHead(emb_dim)
    ptm_head.load_state_dict(ptm_sd)
    ptm_head.to(device).eval()

    return pe_head, ptm_head
