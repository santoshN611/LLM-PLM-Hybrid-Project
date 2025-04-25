#!/usr/bin/env python3
import torch, esm, numpy as np
import pandas as pd
from pathlib import Path
import warnings
from tqdm import tqdm

print("🔄 Starting embedding generation…")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load ESM-2 ────────────────────────────────────────────────────────────
print("📦 Loading ESM-2 model…")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model.eval().to(DEVICE)
batch_converter = alphabet.get_batch_converter()
print(f"✅ ESM-2 ready on {DEVICE}")

# Maximum sequence length ESM-2 can reasonably handle in one chunk:
MAX_SEQ_LEN = 50000

def embed_sequence(seq: str) -> np.ndarray:
    """🎯 Embed a single protein sequence, truncating if too long."""
    original_len = len(seq)
    if original_len > MAX_SEQ_LEN:
        warnings.warn(
            f"⚠️ Sequence length {original_len} > {MAX_SEQ_LEN}, truncating."
        )
        seq = seq[:MAX_SEQ_LEN]

    data = [('id', seq)]
    toks = batch_converter(data)[2].to(DEVICE)

    try:
        with torch.no_grad():
            out = esm_model(toks, repr_layers=[6])['representations'][6]
        emb = out.mean(1).squeeze(0).cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        # Retry on CPU
        warnings.warn("🔥 OOM on GPU, retrying on CPU (no_grad)…")
        torch.cuda.empty_cache()
        esm_model.cpu()
        toks = toks.cpu()
        with torch.no_grad():
            out = esm_model(toks, repr_layers=[6])['representations'][6]
        emb = out.mean(1).squeeze(0).numpy()
        esm_model.to(DEVICE)

    # Free up any reserved GPU memory
    if DEVICE.startswith('cuda'):
        torch.cuda.empty_cache()

    return emb

def process_split(split_name: str):
    """🔄 Load → embed → save for one split."""
    print(f"📥 Loading {split_name}.csv…")
    df = pd.read_csv(f'{split_name}.csv')
    seqs = df['sequence'].tolist()
    labels = df.iloc[:, 2].values

    # Embed one by one (to limit peak GPU usage)
    embeddings = []
    for i, seq in tqdm(enumerate(seqs, 1), total=len(seqs)):
        emb = embed_sequence(seq)
        embeddings.append(emb)
        if i % 1000 == 0:
            print(f"   ↳ Embedded {i}/{len(seqs)} sequences")

    X = np.stack(embeddings)
    y = labels

    Path('embeddings').mkdir(exist_ok=True)
    np.savez_compressed(f'embeddings/{split_name}', X=X, y=y)
    print(f"✅ Saved embeddings for {split_name}")

if __name__ == '__main__':
    splits = [
        'classification_train', 'classification_val', 'classification_test',
        'regression_train', 'regression_val', 'regression_test'
    ]
    for split in splits:
        process_split(split)
    print("🎉 All embeddings generated!")
