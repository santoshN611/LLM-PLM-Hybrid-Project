# rag_pipeline.py
#!/usr/bin/env python3
import os, re, warnings, math
import torch, spacy
from transformers import BertTokenizerFast, BertForSequenceClassification

# Suppress TensorFlow noise
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tiny_heads import load_heads
from retrieval import search_uniprot_name, fetch_uniprot, TAXON_MAP

print("🚀 Starting RAG pipeline…")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Intent (BioBERT) ────────────────────────────────────────
print("📦 Loading BioBERT intent model…")
tok_intent = BertTokenizerFast.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
tok_intent.model_max_length = 128
mdl_intent = BertForSequenceClassification.from_pretrained(
    'dmis-lab/biobert-base-cased-v1.1',
    num_labels=2
).to(DEVICE).eval()
print("✅ BioBERT ready")

# ── Protein NER ─────────────────────────────────────────────
print("📦 Loading SciSpaCy NER model…")
nlp = spacy.load("en_ner_bc5cdr_md")
print("✅ SciSpaCy ready")

# ── tiny-heads ───────────────────────────────────────────────
print("📦 Loading tiny-heads…")
pe_head, ptm_head = load_heads(DEVICE)
print("✅ tiny-heads loaded")

# ── ESM-2 embedder via torch.hub ─────────────────────────────
print("📦 Loading ESM-2 (via torch.hub)…")
esm_model, alphabet = torch.hub.load(
    "facebookresearch/esm:main", "esm2_t6_8M_UR50D"
)
esm_model.to(DEVICE).eval()
batch_converter = alphabet.get_batch_converter()
print(f"✅ ESM-2 ready on {DEVICE}")

def parse_question(q: str):
    """Extract task, accession, raw_seq, name, organism_id."""
    # 1) task by keyword → fallback to BioBERT
    low = q.lower()
    if 'existence' in low:    task = 'protein_existence'
    elif 'ptm' in low:        task = 'ptm_count'
    else:
        inp = tok_intent(q, return_tensors='pt', truncation=True).to(DEVICE)
        logits = mdl_intent(**inp).logits
        task = 'protein_existence' if logits.argmax(-1).item()==0 else 'ptm_count'

    # 2) accession?
    acc = next((t.upper() for t in re.split(r'\W+', q)
                if re.fullmatch(r'(?:[OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d[A-Z0-9]{3}\d)', t)), None)

    # 3) raw sequence?
    mseq = re.search(r'([ACDEFGHIKLMNPQRSTVWY]{4,})', q.replace(' ', ''))
    raw = mseq.group(1).upper() if mseq else None

    # 4) organism?
    org = next((tid for name, tid in TAXON_MAP.items() if name in low), None)

    # 5) name: only if no acc & no raw
    name = None
    if not acc and not raw:
        # 5a) special E. coli DnaK
        if org == TAXON_MAP['ecoli']:
            m = re.search(r'e\.?\s*coli\s+(\w+)', low)
            if m:
                name = m.group(1)
        # 5b) SciSpaCy NER
        if not name:
            for ent in nlp(q).ents:
                if ent.label_ in ("GENE_OR_GENE_PRODUCT","CHEMICAL"):
                    name = ent.text
                    break
        # 5c) “for X” or “of X”
        if not name:
            m2 = re.search(r'(?:for|of)\s+([A-Za-z0-9\-\s]+)', low)
            if m2:
                name = m2.group(1).strip()
    # 6) fallback: if still nothing, use the question body literally
    if not acc and not raw and not name:
        name = low

    print(f"ℹ️ Parsed → task:{task}, acc:{acc}, raw:{bool(raw)}, name:{name!r}, org:{org}")
    return dict(task=task, accession=acc, raw_seq=raw, name=name, organism=org)

def embed_sequence(seq: str) -> torch.Tensor:
    _, _, toks = batch_converter([("Q", seq)])
    toks = toks.to(DEVICE)
    with torch.no_grad():
        reps = esm_model(toks, repr_layers=[6])["representations"][6]
    return reps.mean(1).squeeze(0)

def answer(q: str) -> str:
    info = parse_question(q)

    # 0) raw-seq answers override
    if info['raw_seq']:
        emb = embed_sequence(info['raw_seq'])
        if info['task']=='protein_existence':
            lvl = pe_head(emb).softmax(-1).argmax(-1).item() + 1
            names = ['protein','transcript','homology','predicted','uncertain']
            return f"🤖 I predict level **{lvl} ({names[lvl-1]})** for your sequence."
        est = max(0, int(round(math.expm1(ptm_head(emb).item()))))
        return f"🤖 I predict **~{est}** PTM site{'s' if est!=1 else ''} for your sequence."

    # 1) name→accession lookup
    if info['name'] and not info['accession']:
        a = search_uniprot_name(info['name'], organism=info['organism'])
        info['accession'] = a

    # 2) fetch UniProt JSON
    up = fetch_uniprot(info['accession']) if info['accession'] else None
    if not up:
        return "⚠️ Couldn’t map to UniProt. Please provide an accession or raw sequence."

    prot = info['name'] or up['accession']
    org  = up.get('organism_name','')
    acc  = up['accession']

    if info['task']=='protein_existence':
        return (
            f"📖 **{prot}** (UniProt **{acc}**, {org}) has existence level "
            f"**{up['pe']}** according to UniProt."
        )

    # ptm_count
    return (
        f"📖 **{prot}** (UniProt **{acc}**, {org}) has **{up['ptm']}** annotated PTM "
        f"site{'s' if up['ptm']!=1 else ''} in UniProt. "
        "You can supply a raw sequence for model‐based estimates."
    )

if __name__=="__main__":
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
