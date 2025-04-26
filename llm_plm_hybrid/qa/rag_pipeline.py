import os
import re
import math
import warnings
import torch
import spacy
from transformers import BertTokenizerFast, BertForSequenceClassification
from pathlib import Path

# ── tiny‐heads (PE & PTM predictors) ─────────────────────────────
from llm_plm_hybrid.embeddings.tiny_heads import load_heads

# ── Retrieval functions & TAXON_MAP ──────────────────────────────
from llm_plm_hybrid.retrieval.retrieval import search_uniprot_name, fetch_uniprot, TAXON_MAP
import llm_plm_hybrid.retrieval.retrieval_utils as retrieval_utils

# ── Adapter block & augmented BioBERT ────────────────────────────
from llm_plm_hybrid.qa.adapters import BioBERTWithAdapters

# Suppress TensorFlow oneDNN noise
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("🚀 Starting RAG pipeline…")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Intent model (BioBERT) ────────────────────────────────────────
print("📦 Loading BioBERT intent model…")
tok_intent = BertTokenizerFast.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
tok_intent.model_max_length = 128
mdl_intent = BertForSequenceClassification.from_pretrained(
    'dmis-lab/biobert-base-cased-v1.1',
    num_labels=2
).to(DEVICE).eval()
print("✅ Intent model ready")

# ── NER (SciSpaCy) ───────────────────────────────────────────────
print("📦 Loading SciSpaCy NER model…")
nlp = spacy.load("en_ner_bc5cdr_md")
print("✅ NER ready")

# ── tiny‐heads (PE & PTM predictors) ─────────────────────────────
print("📦 Loading tiny-heads…")
pe_head, ptm_head = load_heads(DEVICE)
print("✅ tiny-heads loaded")

# ── Adapter-augmented BioBERT for QA ─────────────────────────────
print("📦 Loading BioBERT+Adapters for QA…")
tok_qa = BertTokenizerFast.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
model_qa = BioBERTWithAdapters().to(DEVICE).train()
print("✅ QA model ready")

# ── ESM-2 embedder (via torch.hub) ────────────────────────────────
print("📦 Loading ESM-2 via torch.hub…")
esm_model, alphabet = torch.hub.load(
    "facebookresearch/esm:main", "esm2_t6_8M_UR50D"
)
esm_model.eval().to(DEVICE)
batch_converter = alphabet.get_batch_converter()
print(f"✅ ESM-2 ready on {DEVICE}")

# ── FAISS index + metadata ───────────────────────────────────────
base    = Path(__file__).resolve().parent.parent
emb_dir = base / "embeddings"
index, meta = retrieval_utils.load_index(
    emb_dir / "classification_train.index",
    emb_dir / "classification_train.meta.npy"
)


def parse_question(q: str):
    """
    Extracts:
      - task: 'protein_existence' or 'ptm_count'
      - accession (if present)
      - raw_seq (if present)
      - name (gene/protein name via NER or heuristics)
      - organism taxon_id (if mentioned)
    """
    low = q.lower()
    # 1) keyword‐based intent shortcuts
    exist_kw = ['existence', 'evidence', 'classification']
    ptm_kw   = ['ptm', 'post-translational', 'modified residue', 'modification', 'site count']

    if any(k in low for k in exist_kw):
        task = 'protein_existence'
    elif any(k in low for k in ptm_kw):
        task = 'ptm_count'
    else:
        # fallback to BioBERT classifier
        inp = tok_intent(q, return_tensors='pt', truncation=True).to(DEVICE)
        logits = mdl_intent(**inp).logits
        task = 'protein_existence' if logits.argmax(-1).item() == 0 else 'ptm_count'

    # 2) accession?
    acc = next((
        t.upper() for t in re.split(r'\W+', q)
        if re.fullmatch(r'(?:[OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d[A-Z0-9]{3}\d)', t)
    ), None)

    # 3) raw sequence?
    mseq = re.search(r'([ACDEFGHIKLMNPQRSTVWY]{4,})', q.replace(' ', ''))
    raw = mseq.group(1).upper() if mseq else None

    # 4) organism?
    org = next((tid for name, tid in TAXON_MAP.items() if name in low), None)

    # 5) name via NER or heuristics
    name = None
    if not acc and not raw:
        # E. coli special case
        if org == TAXON_MAP.get('ecoli'):
            m = re.search(r'e\.?\s*coli\s+(\w+)', low)
            if m:
                name = m.group(1)
        # SciSpaCy NER
        if not name:
            for ent in nlp(q).ents:
                if ent.label_ in ("GENE_OR_GENE_PRODUCT", "CHEMICAL"):
                    name = ent.text
                    break
        # “for X” or “of X” fallback
        if not name:
            m2 = re.search(r'(?:for|of)\s+([A-Za-z0-9\-\s]+)', low)
            if m2:
                name = m2.group(1).strip()

    if not acc and not raw and not name:
        name = low

    return {
        'task': task,
        'accession': acc,
        'raw_seq': raw,
        'name': name,
        'organism': org
    }


def embed_sequence(seq: str) -> torch.Tensor:
    """Mean-pooled ESM-2 embedding for a given protein sequence."""
    _, _, toks = batch_converter([("Q", seq)])
    toks = toks.to(DEVICE)
    with torch.no_grad():
        reps = esm_model(toks, repr_layers=[6])["representations"][6]
    return reps.mean(1).squeeze(0)


def answer(q: str) -> str:
    info = parse_question(q)

    # ── (A) Raw-sequence queries → adapter QA w/ FAISS context ─────────
    if info['raw_seq']:
        emb = embed_sequence(info['raw_seq']).cpu().numpy()
        nbrs = retrieval_utils.search_neighbors(emb, k=5)
        context = retrieval_utils.build_context_block(nbrs)
        prompt = f"{context}\n\n{q}"
        toks = tok_qa(prompt, return_tensors='pt', truncation=True, padding=True).to(DEVICE)
        logits, cls_emb = model_qa(**toks)
        pred = logits.argmax(-1).item()
        if info['task'] == 'protein_existence':
            return f"🤖 Predicted existence level **{pred+1}**."
        else:
            return f"🤖 Predicted PTM count class **{pred}**."

    # ── (B) Name lookup → accession → factual UniProt JSON fetch ───────
    if info['name'] and not info['accession']:
        info['accession'] = search_uniprot_name(info['name'], info['organism'])

    up = fetch_uniprot(info['accession']) if info['accession'] else None
    if not up:
        return "⚠️ Couldn’t map to UniProt. Please provide an accession or raw sequence."

    prot = info['name'] or up['accession']
    acc  = up['accession']
    org  = up.get('organism_name', '')

    if info['task'] == 'protein_existence':
        return (
            f"📖 **{prot}** (UniProt **{acc}**, {org}) has existence level **{up['pe']}**."
        )
    else:
        return (
            f"📖 **{prot}** (UniProt **{acc}**, {org}) has **{up['ptm']}** annotated PTM "
            f"site{'s' if up['ptm']!=1 else ''}."
        )


if __name__ == "__main__":
    for q in [
        "What is the protein existence level of lactase?",
        "How many PTM sites for P09812?",
        "PTM count for MVHFAELVK?",
        "Predict existence level for MLLTEQFK right now."
    ]:
        print("▶", q)
        print("→", answer(q), "\n")
