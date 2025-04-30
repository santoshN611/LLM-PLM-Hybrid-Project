# llm_plm_hybrid/qa/stats_utils.py

import numpy as np
from collections import Counter

def neighbor_stats(neighbors: list[dict]) -> tuple[float, int]:
    """
    Compute:
      - mean PTM count over the 5 neighbors
      - majority protein‐existence class among them
    """
    ptm_counts = [n.get("ptm_count", 0) for n in neighbors]
    pe_classes = [n.get("pe_class") for n in neighbors]

    mean_ptm = float(np.mean(ptm_counts)) if ptm_counts else 0.0
    majority_pe = Counter(pe_classes).most_common(1)[0][0] if pe_classes else None

    return mean_ptm, majority_pe
