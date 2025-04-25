#!/usr/bin/env python3
import torch, esm, numpy as np, pandas as pd, warnings
from pathlib import Path
from tqdm import tqdm

print("🔄 Starting embedding generation…")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("📦 Loading ESM-2 model…")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model.eval().to(DEVICE)
batch_converter = alphabet.get_batch_converter()
print(f"✅ ESM-2 ready on {DEVICE}")

MAX_SEQ_LEN = 50000
REPORT_EVERY = 1000  # change this number to report more or less frequently

def embed_sequence(seq: str) -> np.ndarray:
    """Embed a single sequence, with GPU OOM fallback."""
    if len(seq) > MAX_SEQ_LEN:
        warnings.warn(f"⚠️ Truncating seq len {len(seq)} > {MAX_SEQ_LEN}")
        seq = seq[:MAX_SEQ_LEN]

    _, _, toks = batch_converter([("id", seq)])
    toks = toks.to(DEVICE)

    try:
        with torch.no_grad():
            out = esm_model(toks, repr_layers=[6])["representations"][6]
            reps = out.mean(1)
        emb = reps.cpu().numpy()
        del out, reps
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
        return emb

    except torch.cuda.OutOfMemoryError:
        warnings.warn("🔥 OOM on GPU, retrying on CPU")
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
            esm_model.cpu()
        with torch.no_grad():
            out = esm_model(toks.cpu(), repr_layers=[6])["representations"][6]
            reps = out.mean(1)
        emb = reps.numpy()
        if torch.cuda.is_available():
            esm_model.to(DEVICE)
            torch.cuda.empty_cache()
        del out, reps
        return emb

def process_split(split_name: str):
    print(f"📥 Loading {split_name}.csv…")
    df = pd.read_csv(f"{split_name}.csv")
    seqs = df["sequence"].tolist()
    embeddings = []
    total = len(seqs)

    for i, seq in enumerate(tqdm(seqs, total=total), start=1):
        emb = embed_sequence(seq)
        embeddings.append(emb)

        # dropdown‐style progress report:
        if i % REPORT_EVERY == 0 or i == total:
            print(f"   ↳ Completed {i}/{total} embeddings")

    X = np.stack(embeddings)
    y = df.iloc[:, 2].values
    Path("embeddings").mkdir(exist_ok=True)
    np.savez_compressed(f"embeddings/{split_name}.npz", X=X, y=y)
    print(f"✅ Saved embeddings for {split_name}")

if __name__ == "__main__":
    for split in [
        "classification_train", "classification_val", "classification_test",
        "regression_train",     "regression_val",     "regression_test"
    ]:
        process_split(split)
    print("🎉 All embeddings generated!")
