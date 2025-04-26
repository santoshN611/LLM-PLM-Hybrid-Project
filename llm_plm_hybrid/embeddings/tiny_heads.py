import torch
import torch.nn as nn
from pathlib import Path


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
