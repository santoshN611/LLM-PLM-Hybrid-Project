#!/usr/bin/env python3
import json
import csv
import numpy as np
from evaluate import load as load_metric
from rag_pipeline import answer
from retrieval_utils import load_index, search_neighbors
from tqdm import tqdm

# ── 1) Load test corpus ───────────────────────────────────────────
TEST_JSONL = "scripts/test_protein_qa.jsonl"
entries     = [json.loads(line) for line in open(TEST_JSONL)]
questions   = [e["question"] for e in entries]
gold_answers= [e["answer"]   for e in entries]
ids         = [e["id"]       for e in entries]

# ── 2) Run the pipeline ────────────────────────────────────────────
pred_answers    = []
retrieved_ranks = []

# Load FAISS index + metadata (index_file, meta_file)
index, meta = load_index(
    "embeddings/classification_train.index",
    "embeddings/classification_train.meta.npy"
)

for q, gold in zip(questions, gold_answers):
    # 2a) QA output
    pred_answers.append(answer(q))

    # 2b) If there's a UniProt accession in the question, compute its retrieval rank
    import re
    m = re.search(r'\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])\b', q)
    if m:
        acc = m.group(0)
        # Build a map accession→row in the embeddings
        data = np.load("embeddings/classification_train.npz", allow_pickle=True)
        accs = data["meta"]
        idx_map = {a:i for i,a in enumerate(accs)}
        if acc in idx_map:
            vec = data["X"].squeeze(1)[ idx_map[acc] ].astype("float32").reshape(1, -1)
            # Search top-10
            D, I = index.search(vec, 10)
            neighbors = I[0].tolist()
            if idx_map[acc] in neighbors:
                retrieved_ranks.append(neighbors.index(idx_map[acc]) + 1)
            else:
                retrieved_ranks.append(None)
        else:
            retrieved_ranks.append(None)

# ── 3) Generation metrics ──────────────────────────────────────────
bleu   = load_metric("bleu").compute(predictions=pred_answers, references=[[g] for g in gold_answers])["bleu"]
rouge  = load_metric("rouge").compute(predictions=pred_answers, references=[[g] for g in gold_answers])
meteor = load_metric("meteor").compute(predictions=pred_answers, references=[[g] for g in gold_answers])["meteor"]
bertsc = load_metric("bertscore").compute(predictions=pred_answers, references=[[g] for g in gold_answers])["f1"]

print(f"BLEU: {bleu:.3f}")
print(f"ROUGE: {rouge}")
print(f"METEOR: {meteor:.3f}")
print(f"BERTScore (mean F1): {np.mean(bertsc):.3f}")

# ── 4) Retrieval metrics ───────────────────────────────────────────
valid = [r for r in retrieved_ranks if r is not None]
prec1 = sum(1 for r in valid if r == 1) / len(valid) if valid else 0.0
mrr   = np.mean([1.0/r for r in valid])           if valid else 0.0

print(f"Precision@1: {prec1:.3f}")
print(f"MRR: {mrr:.3f}")

# ── 5) Save detailed results ───────────────────────────────────────
with open("eval_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id","question","gold","pred","retrieval_rank"])
    for i, q, g, p, rr in zip(ids, questions, gold_answers, pred_answers, retrieved_ranks):
        writer.writerow([i, q, g, p, rr])

print("✅ Evaluation complete! Results saved to eval_results.csv")
