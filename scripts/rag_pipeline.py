#!/usr/bin/env python3
import os
import re
import math
import warnings

import torch
import spacy
from transformers import BertTokenizerFast, BertForSequenceClassification

from tiny_heads import load_heads
from retrieval import search_uniprot_name, fetch_uniprot, TAXON_MAP

# Quiet unwanted TensorFlow/oneDNN logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("🚀 Starting RAG pipeline…")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Intent Detection (BioBERT) ─────────────────────────────────────
print("📦 Loading intent model (BioBERT)…")
tok_intent = BertTokenizerFast.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
tok_intent.model_max_length = 128
mdl_intent = BertForSequenceClassification.from_pretrained(
    'dmis-lab/biobert-base-cased-v1.1', num_labels=2
).eval().to(DEVICE)
print("✅ BioBERT ready")

# ─── Protein NER (SciSpaCy) ──────────────────────────────────────────
print("📦 Loading SciSpaCy NER model…")
nlp = spacy.load("en_ner_bc5cdr_md")
print("✅ SciSpaCy ready")

# ─── tiny‐heads ──────────────────────────────────────────────────────
print("📦 Loading tiny_heads…")
pe_head, ptm_head = load_heads(DEVICE)
print("✅ tiny_heads loaded")

# ─── ESM-2 embedder via torch.hub ───────────────────────────────────
print("📦 Loading ESM-2 embedder…")
esm_model, alphabet = torch.hub.load(
    "facebookresearch/esm:main", "esm2_t6_8M_UR50D"
)
esm_model.to(DEVICE).eval()
batch_converter = alphabet.get_batch_converter()
print(f"✅ ESM-2 ready on {DEVICE}")

# Reverse taxon map for display
REV_TAXON = {v:k for k,v in TAXON_MAP.items()}

def parse_question(q: str):
    q_low = q.lower()
    # 1) Task by keywords or BioBERT fallback
    if 'existence' in q_low:
        task = 'protein_existence'
    elif 'ptm' in q_low:
        task = 'ptm_count'
    else:
        inp = tok_intent(q, return_tensors='pt', truncation=True).to(DEVICE)
        logits = mdl_intent(**inp).logits
        task = 'protein_existence' if logits.argmax(-1).item()==0 else 'ptm_count'
    # 2) Accession
    acc = next((t.upper() for t in re.split(r'\W+', q)
                if re.fullmatch(r'(?:[OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d[A-Z0-9]{3}\d)', t)),
               None)
    # 3) Raw sequence
    m = re.search(r'([ACDEFGHIKLMNPQRSTVWY]{4,})', q.replace(' ',''))
    raw_seq = m.group(1).upper() if m else None
    # 4) Name by NER
    name = None
    if not acc and not raw_seq:
        for ent in nlp(q).ents:
            if ent.label_ in ("GENE_OR_GENE_PRODUCT", "CHEMICAL"):
                name = ent.text
                break
    # 5) Organism
    org = next((tid for k,tid in TAXON_MAP.items() if k in q_low), None)
    print(f"ℹ️ Parsed → task:{task}, acc:{acc}, raw_seq:{bool(raw_seq)}, name:{name}, org:{org}")
    return dict(task=task, accession=acc, raw_seq=raw_seq, name=name, organism=org)

def embed_sequence(seq: str):
    _, _, toks = batch_converter([("Q", seq)])
    toks = toks.to(DEVICE)
    with torch.no_grad():
        reps = esm_model(toks, repr_layers=[6])['representations'][6]
    return reps.mean(1).squeeze(0)

def answer(q: str) -> str:
    info = parse_question(q)

    # 0) Raw‐sequence‐only: model prediction
    if info['raw_seq']:
        emb = embed_sequence(info['raw_seq'])
        if info['task']=='protein_existence':
            lvl = pe_head(emb).softmax(-1).argmax().item() + 1
            return f"🤖 I predict existence level **{lvl}** (ESM-2 + tiny-head)."
        logp = ptm_head(emb).item()
        est  = max(0, int(round(math.expm1(logp))))
        return f"🤖 I predict **{est}** PTM sites (ESM-2 + tiny-head)."

    # 1) Name → Accession lookup
    if info['name'] and not info['accession']:
        info['accession'] = search_uniprot_name(
            info['name'], organism=info['organism']
        )

    # 2) Fetch UniProt entry
    up = fetch_uniprot(info['accession']) if info['accession'] else None
    if not up:
        return ("⚠️ I couldn’t map that to a UniProt entry. "
                "Please provide an accession or raw AA sequence.")

    prot = info['name'] or up['accession']
    organ = REV_TAXON.get(up['organism_id'], str(up['organism_id']))

    # 3) Exact-data response
    if info['task']=='protein_existence':
        return (f"📖 **{prot}** (accession **{up['accession']}**, "
                f"organism **{organ}**) has existence level **{up['pe']}**.")
    else:
        return (f"📖 **{prot}** (accession **{up['accession']}**, "
                f"organism **{organ}**) has **{up['ptm']}** PTM sites.")

if __name__=="__main__":
    for q in [
        "What is the protein existence level of lactase?",
        "How many predicted PTM sites are there for P09812?",
        "PTM count for MVHFAELVK?",
        "Level of protein existence for ACDEFGHIKL?",
        "What is the existence level of mouse hemoglobin?",
        "PTM count for E. coli DnaK?",
    ]:
        print("▶", q)
        print("→", answer(q), "\n")
