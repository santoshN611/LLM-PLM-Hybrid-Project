#!/usr/bin/env python3
import os
# suppress oneDNN/TensorFlow noise if any
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import torch
import esm
from transformers import BertTokenizerFast, BertForSequenceClassification

from tiny_heads import load_heads
from retrieval import search_uniprot_name, fetch_uniprot, TAXON_MAP

print("🚀 Starting RAG pipeline…")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────── Intent Detection ─────────────────────────────────────────────
print("📦 Loading intent model (BioBERT)…")
tok_intent = BertTokenizerFast.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
tok_intent.model_max_length = 128
mdl_intent = BertForSequenceClassification.from_pretrained(
    'dmis-lab/biobert-base-cased-v1.1',
    num_labels=2
).eval().to(DEVICE)
print("✅ BioBERT ready")

def parse_question(q: str):
    inputs = tok_intent(q, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    logits = mdl_intent(**inputs).logits
    task = 'protein_existence' if logits.argmax(-1).item() == 0 else 'ptm_count'

    q_low = q.lower()
    if 'existence' in q_low: task = 'protein_existence'
    if 'ptm' in q_low:       task = 'ptm_count'

    # Accession pattern
    acc = next(
        (t.upper() for t in re.split(r'\W+', q)
         if re.fullmatch(r'[OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d[A-Z0-9]{3}\d', t)),
        None
    )

    # Raw AA sequence
    raw = re.search(r'([ACDEFGHIKLMNPQRSTVWY]{4,})', q.replace(' ', ''))
    raw_seq = raw.group(0).upper() if raw else None

    # Common name fallback
    name = None
    if not acc and not raw_seq:
        if ' of ' in q_low:
            name = q_low.split(' of ')[-1].strip().rstrip('?')
        elif ' for ' in q_low:
            name = q_low.split(' for ')[-1].strip().rstrip('?')
        else:
            m = re.search(r'\b[A-Za-z0-9\-]+\b', q_low)
            if m:
                name = m.group(0)

    # Organism ID detection (normalize e. coli → ecoli)
    q_norm = re.sub(r'[\.\s]+','', q_low)
    org = None
    for k, tid in TAXON_MAP.items():
        if k in q_norm:
            org = tid
            break

    print(f"ℹ️ Parsed: task={task}, accession={acc}, raw_seq={'yes' if raw_seq else 'no'}, "
          f"name={name}, organism={org}")
    return {
        'task': task,
        'accession': acc,
        'raw_seq_str': raw_seq,
        'name': name,
        'organism': org
    }

# ─────── ESM-2 Embedder ───────────────────────────────────────────────
print("📦 Loading ESM-2 embedder…")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model.eval().to(DEVICE)
batch_converter = alphabet.get_batch_converter()
print(f"✅ ESM-2 ready on {DEVICE}")

def embed_sequence(seq: str) -> torch.Tensor:
    toks = batch_converter([('Q', seq)])[2].to(DEVICE)
    with torch.no_grad():
        out = esm_model(toks, repr_layers=[6])['representations'][6]
    return out.mean(1).squeeze(0)

# ─────── tiny-heads ───────────────────────────────────────────────────
print("📦 Loading tiny_heads…")
pe_head, ptm_head = load_heads(DEVICE)
print("✅ tiny_heads loaded")

# ─────── Main answer fn ───────────────────────────────────────────────
def answer(q: str) -> str:
    info = parse_question(q)

    # Name → accession (reviewed first)
    if info['name'] and not info['accession']:
        info['accession'] = search_uniprot_name(info['name'], organism=info['organism'])

    # Fetch entry if accession available
    up = fetch_uniprot(info['accession']) if info['accession'] else None
    seq = up['seq'] if up and up.get('seq') else info['raw_seq_str']
    if not seq:
        return "⚠️ Need a UniProt accession or a raw AA sequence."

    emb = embed_sequence(seq)

    if info['task'] == 'protein_existence':
        if up:
            return (f"📖 UniProt reports: **{up['pe']}** "
                    f"(accession {info['accession']}, taxon {up.get('organism_id')}).")
        pred   = pe_head(emb).softmax(-1).argmax(-1).item()
        levels = ['protein','transcript','homology','predicted','uncertain']
        return f"🤖 I predict level {pred+1} ({levels[pred]})."

    # PTM count
    if up:
        return (f"📖 UniProt lists **{up['ptm']}** modified residues "
                f"(accession {info['accession']}).")
    logp = ptm_head(emb).item()
    est  = float(torch.expm1(torch.tensor(logp)))
    return f"🤖 I predict ~{est:.1f} PTM sites."

# ─────── Demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    for q in [
        "What is the protein existence level of lactase?",
        "How many predicted PTM sites are there for P09812?",
        "PTM count for MVHFAELVK?",
        "Level of protein existence for ACDEFGHIKL?",
        "What is the existence level of mouse hemoglobin?",
        "PTM count for E. coli DnaK?"
    ]:
        print("▶", q)
        print("→", answer(q), "\n")
