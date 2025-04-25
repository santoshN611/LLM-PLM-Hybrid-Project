#!/usr/bin/env python3
"""
RAG‐style protein Q&A using:
  • BioBERT intent parsing
  • SciSpaCy NER + regex cleanup
  • UniProt full‐JSON fetch
  • ESM-2 embeddings via Torch Hub
  • tiny‐heads classifiers on top of ESM-2
"""
import os
import re
import math
import warnings

import torch
import spacy
from transformers import BertTokenizerFast, BertForSequenceClassification

from tiny_heads import load_heads
from retrieval import search_uniprot_name, fetch_uniprot, TAXON_MAP

# Suppress TensorFlow oneDNN chatter (if TF is also on your PATH)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("🚀 Starting RAG pipeline…")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Intent Detection (BioBERT) ─────────────────────────────────────
print("📦 Loading intent model (BioBERT)…")
tok_intent = BertTokenizerFast.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
tok_intent.model_max_length = 128
mdl_intent = (
    BertForSequenceClassification
    .from_pretrained('dmis-lab/biobert-base-cased-v1.1', num_labels=2)
    .to(DEVICE)
    .eval()
)
print("✅ BioBERT ready")

# ─── Protein NER (SciSpaCy) ───────────────────────────────────────
print("📦 Loading SciSpaCy NER model…")
nlp_bio = spacy.load("en_ner_bc5cdr_md")
print("✅ SciSpaCy ready")

# ─── tiny-heads (on top of ESM-2) ────────────────────────────────
print("📦 Loading tiny_heads…")
pe_head, ptm_head = load_heads(DEVICE)
print("✅ tiny_heads loaded")

# ─── ESM-2 Embedder via Torch Hub ───────────────────────────────
print("📦 Loading ESM-2 embedder…")
esm_model, alphabet = torch.hub.load(
    "facebookresearch/esm:main",
    "esm2_t6_8M_UR50D"
)
esm_model = esm_model.to(DEVICE).eval()
batch_converter = alphabet.get_batch_converter()
print(f"✅ ESM-2 ready on {DEVICE}")

# Reverse mapping from taxon ID → common name
REV_TAXON = {v: k for k, v in TAXON_MAP.items()}
REV_TAXON[9606] = "human"


def parse_question(q: str):
    """
    Returns a dict with:
      - task: 'protein_existence' or 'ptm_count'
      - accession: str or None
      - raw_seq_str: raw AA sequence or None
      - name: protein common name or None
      - organism: taxon ID or None
    """
    # 1) Intent by keyword → fallback to BioBERT
    q_low = q.lower()
    if any(kw in q_low for kw in ("existence", "level of")):
        task = "protein_existence"
    elif "ptm" in q_low or "modified residue" in q_low:
        task = "ptm_count"
    else:
        inputs = tok_intent(q, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        logits = mdl_intent(**inputs).logits
        task = 'protein_existence' if logits.argmax(-1).item() == 0 else 'ptm_count'

    # 2) Accession by regex
    acc = next((
        tok.upper() for tok in re.split(r'\W+', q)
        if re.fullmatch(r'(?:[OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d[A-Z0-9]{3}\d)', tok)
    ), None)

    # 3) Raw‐AA‐sequence detection
    raw_match = re.search(r'([ACDEFGHIKLMNPQRSTVWY]{4,})', q.replace(' ', ''))
    raw_seq = raw_match.group(1).upper() if raw_match else None

    # 4) NER‐extracted name
    name = None
    if not acc and not raw_seq:
        for ent in nlp_bio(q).ents:
            if ent.label_ in ("GENE_OR_GENE_PRODUCT", "CHEMICAL"):
                name = ent.text
                break

    # 5) Fallback name extraction by simple regex
    if not name and not acc and not raw_seq:
        m = re.search(r'(?:of|for)\s+([A-Za-z0-9\-\s]+)\??', q_low)
        if m:
            name = m.group(1).strip()

    # 6) Organism keyword → taxon
    org = None
    for key, tid in TAXON_MAP.items():
        if key in q_low:
            org = tid
            break

    print(f"ℹ️ Parsed → task:{task}, acc:{acc}, raw_seq:{bool(raw_seq)}, name:{name}, org:{org}")
    return dict(
        task=task,
        accession=acc,
        raw_seq_str=raw_seq,
        name=name,
        organism=org
    )


def embed_sequence(seq: str) -> torch.Tensor:
    """
    Returns the averaged ESM-2 representation for a raw AA sequence.
    """
    _, _, toks = batch_converter([("Q", seq)])
    toks = toks.to(DEVICE)
    with torch.no_grad():
        reps = esm_model(toks, repr_layers=[6])["representations"][6]
    return reps.mean(1).squeeze(0)


def answer(q: str) -> str:
    info = parse_question(q)

    # ─── Raw‐sequence branch (highest priority) ──────────────────────
    if info['raw_seq_str']:
        emb = embed_sequence(info['raw_seq_str'])
        if info['task'] == 'protein_existence':
            pred = pe_head(emb).softmax(-1).argmax(-1).item()
            levels = ['protein', 'transcript', 'homology', 'predicted', 'uncertain']
            return (
                f"🤖 I predict existence level **{pred+1}** ({levels[pred]}) "
                "for your provided sequence (via ESM-2 embeddings + tiny-head)."
            )
        # PTM‐count
        logp = ptm_head(emb).item()
        est = max(0, int(round(math.expm1(logp))))
        return (
            f"🤖 I predict ~**{est}** PTM sites for your provided sequence "
            "(via ESM-2 embeddings + tiny-head)."
        )

    # ─── Name→UniProt lookup (if no accession) ───────────────────────
    if info['name'] and not info['accession']:
        # Try with organism→fallback to all
        acc = search_uniprot_name(info['name'], organism=info['organism'])
        info['accession'] = acc

    # ─── Fetch full UniProt JSON ────────────────────────────────────
    up = fetch_uniprot(info['accession']) if info['accession'] else None
    if not up:
        return (
            "⚠️ I couldn’t map that to a UniProt entry. "
            "Could you specify the organism or give an accession?"
        )

    # ─── Compose a friendly, LLM‐style answer ──────────────────────
    acc   = up['accession']
    prot  = info['name'] or acc
    org_id = up.get('organism')
    organ = REV_TAXON.get(org_id, str(org_id))

    if info['task'] == 'protein_existence':
        return (
            f"📖 **{prot}** (UniProt **{acc}**, organism **{organ}**) "
            f"has existence level **{up['pe']}** according to UniProt. "
            "If you’d like a model‐based prediction instead, supply a raw sequence."
        )

    # PTM count branch
    return (
        f"📖 **{prot}** (UniProt **{acc}**, organism **{organ}**) "
        f"has **{up['ptm']}** annotated PTM sites in UniProt. "
        "Model‐based estimates are also available with a sequence."
    )


if __name__ == "__main__":
    examples = [
        "What is the protein existence level of lactase?",
        "How many predicted PTM sites are there for P09812?",
        "PTM count for MVHFAELVK?",
        "Level of protein existence for ACDEFGHIKL?",
        "What is the existence level of mouse hemoglobin?",
        "PTM count for E. coli DnaK?",
    ]
    for q in examples:
        print("▶", q)
        print("→", answer(q), "\n")
