#!/usr/bin/env python3
<<<<<<< HEAD
"""
RAG‐style protein Q&A with embedding‐based retrieval

• heavily ESM-2 embedding–driven
• text lookup only for seed accession
• FAISS k-NN fallback for unknown names
• UniProt JSON lookup
• ESM-2 + tiny-heads for raw AA sequences
"""

import os, re, torch, esm, faiss, requests, pandas as pd
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    logging as hflog
)
from tiny_heads import load_heads
from retrieval   import search_uniprot_name, TAXON_MAP

# ───────────────────────── Silence Noisy Logs ───────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
hflog.set_verbosity_error()

# ─────────────────── Normalization aliases ──────────────────────────────────
ALIASES = {
    r"\be[.\s]?coli\b": "ecoli",
    r"\bdna[ \-]?k\b":   "dnak",
    # add more as needed
}

def normalize(q: str) -> str:
    nq = q.lower()
    for pat, repl in ALIASES.items():
        nq = re.sub(pat, repl, nq, flags=re.IGNORECASE)
    nq = re.sub(r"[^a-z0-9\s]", " ", nq)
    return re.sub(r"\s+", " ", nq).strip()

# ─────────────────── species+protein regex ─────────────────────────────────
species_keys    = sorted(map(re.escape, TAXON_MAP), key=len, reverse=True)
species_pattern = "|".join(species_keys)
ORG_PROT_RE     = re.compile(rf"\b({species_pattern})\b\s+([\w\-]+)",
                             re.IGNORECASE)

# ─────────────────── Load BioBERT intent model ──────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
tok_intent  = BertTokenizerFast.from_pretrained(
                 "dmis-lab/biobert-base-cased-v1.1")
tok_intent.model_max_length = 512
mdl_intent  = BertForSequenceClassification.from_pretrained(
                 "dmis-lab/biobert-base-cased-v1.1",
                 num_labels=2
              ).eval().to(DEVICE)

# ───────────────── ESM-2 & tiny-heads ──────────────────────────────────────
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model.eval().to(DEVICE)
batch_converter   = alphabet.get_batch_converter()
pe_head, ptm_head = load_heads(DEVICE)

PE_TXT = [
    'evidence at protein level',
    'evidence at transcript level',
    'inferred from homology',
    'predicted',
    'uncertain'
]

# ──────────── Load FAISS index & corpus accessions ──────────────────────────
print("📦 Loading FAISS index…")
index = faiss.read_index("embeddings/classification_train.index")
print(f"✅ FAISS index with {index.ntotal} vectors loaded")

print("📦 Loading corpus accessions…")
df_acc = pd.read_csv("classification_train.csv")
corpus_accessions = df_acc["accession"].tolist()
print(f"✅ Loaded {len(corpus_accessions)} accessions")

# ───────────────────────── Helpers ──────────────────────────────────────────
def embed_sequence(seq: str) -> torch.Tensor:
    print(f"🔗 Embedding sequence (len={len(seq)})…")
    _, _, toks = batch_converter([("Q", seq)])
    toks = toks.to(DEVICE)
    with torch.no_grad():
        reps = esm_model(toks, repr_layers=[6])["representations"][6]
    return reps.mean(1)

def fetch_uniprot(acc: str) -> dict:
    print(f"🔄 Fetching UniProt entry for {acc}…")
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json",
                     timeout=15)
    r.raise_for_status()
    js  = r.json()
    seq = js["sequence"]["value"]
    pe  = js["proteinExistence"]
    ptm = sum(
        1 for f in js.get("features", [])
        if f.get("type","").lower() == "modified residue"
    )
    print(f"✅ UniProt data: len={len(seq)}, pe={pe}, ptm={ptm}")
    return {"seq": seq, "pe": pe, "ptm": ptm}

def retrieve_neighbors(seq: str, k: int = 5):
    emb = embed_sequence(seq).cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)
    D, I = index.search(emb, k)
    return [corpus_accessions[i] for i in I[0]], D[0]

# ─────────────────────── parse_question() ─────────────────────────────────
def parse_question(raw_q: str) -> dict:
    print(f"ℹ️ raw : {raw_q}")
    q = normalize(raw_q)
    print(f"ℹ️ norm: {q}")

    # 1) species+protein
    m = ORG_PROT_RE.search(q)
    organism, common_name = None, None
    if m:
        organism    = TAXON_MAP[m.group(1).lower()]
        common_name = m.group(2)
        q = q.replace(m.group(0), "")

    # 2) intent
    inputs = tok_intent(raw_q,
                       return_tensors="pt",
                       truncation=True,
                       max_length=tok_intent.model_max_length).to(DEVICE)
    logits = mdl_intent(**inputs).logits
    task   = "protein_existence" if logits.argmax(-1).item()==0 else "ptm_count"
    if "existence" in q:
        task = "protein_existence"
    elif "ptm" in q or "post translational" in q:
        task = "ptm_count"

    # 3) explicit accession
    accession = None
    for t in re.split(r"[^A-Za-z0-9]", raw_q):
        if re.fullmatch(
            r"[OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d[A-Z0-9]{3}\d", t
        ):
            accession = t.upper()
            break

    # 4) raw AA sequence
    aa_m    = re.search(
        r"([ACDEFGHIKLMNPQRSTVWY]{4,})", raw_q.replace(" ", "")
    )
    raw_seq = aa_m.group(1).upper() if aa_m else None

    # 5) fallback after “of”/“for”
    if not common_name and not accession and not raw_seq:
        frag = None
        if " of " in q:
            frag = q.split(" of ")[-1]
        elif " for " in q:
            frag = q.split(" for ")[-1]
        if frag:
            fn = normalize(frag)
            m2 = ORG_PROT_RE.match(fn)
            if m2:
                organism    = TAXON_MAP[m2.group(1).lower()]
                common_name = m2.group(2)
            else:
                common_name = fn

    print(
        f"   → task={task}, acc={accession}, raw="
        f"{'yes' if raw_seq else 'no'}, name={common_name}, org={organism}"
    )
    return {
        "task":        task,
        "accession":   accession,
        "raw_seq":     raw_seq,
        "common_name": common_name,
        "organism":    organism
    }

# ───────────────────────────── answer() ─────────────────────────────────────
def answer(q: str) -> str:
    info = parse_question(q)

    # A) Seed text lookup
    if info["common_name"] and not info["accession"]:
        acc0 = search_uniprot_name(info["common_name"], info["organism"])
        if acc0:
            info["accession"] = acc0
            print(f"ℹ️ seed accession → {acc0}")

    # B) Retrieval fallback only if STILL no accession
    if info["common_name"] and not info["accession"]:
        print(f"🔍 embedding‐based retrieval for '{info['common_name']}'…")
        # fetch seed sequence if any
        seq0 = fetch_uniprot(acc0)["seq"]
        cands, sims = retrieve_neighbors(seq0, k=5)
        info["accession"] = cands[0]
        print(f"ℹ️ retrieval → {cands[0]} (sim={sims[0]:.3f})")

    # C) UniProt JSON if we have accession
    uni_data = None
    if info["accession"]:
        try:
            uni_data = fetch_uniprot(info["accession"])
        except Exception as e:
            return f"⚠️ UniProt fetch failed: {e}"

    # D) Return authoritative UniProt data
    if uni_data:
        if info["task"] == "protein_existence":
            return f"📖 UniProt reports: {uni_data['pe'].lower()}."
        else:
            return f"📖 UniProt lists {uni_data['ptm']} PTM sites."

    # E) Raw‐sequence fallback
    if info["raw_seq"]:
        emb = embed_sequence(info["raw_seq"])
        if info["task"] == "protein_existence":
            lvl = pe_head(emb).softmax(-1).argmax(-1).item()
            return f"🤖 I predict level {lvl+1} ({PE_TXT[lvl]})."
        est = float(torch.expm1(torch.tensor(ptm_head(emb).item())))
        return f"🤖 I predict ~{est:.1f} PTM sites."

    return "⚠️ Need a UniProt accession or raw AA sequence."

# ─────────────────────────── Demo ───────────────────────────────────────────
if __name__ == "__main__":
    for q in [
        "What is the protein existence level of lactase?",
        "How many predicted PTM sites are there for P09812?",
        "PTM count for MVHFAELVK?",
        "Level of protein existence for ACDEFGHIKL?",
        "What is the existence level of mouse hemoglobin?",
        "PTM count for E. coli DnaK?"
    ]:
=======
import re, requests, torch, esm, warnings
from tiny_heads import load_heads
from retrieval import search_uniprot_name
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

print("🚀 Starting RAG pipeline…")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PE_TXT = ['evidence at protein level', 'evidence at transcript level',
          'inferred from homology', 'predicted', 'uncertain']

# ─── Intent detection ─────────────────────────────────────────────────────
print("📦 Loading intent model…")
tok_intent = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
mdl_intent = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2).eval().to(DEVICE)
print("✅ Intent model ready")

def parse_question(q):
    print(f"ℹ️ Parsing question: {q}")
    inputs = tok_intent(q, return_tensors='pt', truncation=True).to(DEVICE)
    logits = mdl_intent(**inputs).logits
    task = 'protein_existence' if logits.argmax(-1).item()==0 else 'ptm_count'
    acc = None
    for t in re.split(r'[^A-Za-z0-9]', q):
        if re.fullmatch(r'[OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d[A-Z0-9]{3}\d', t):
            acc = t.upper(); break
    raw_match = re.search(r'([ACDEFGHIKLMNPQRSTVWY]{4,})', q.replace(' ', ''))
    raw = raw_match.group(0).upper() if raw_match else None
    name = None
    if not acc and not raw:
        m = re.search(r'([A-Za-z0-9\-]+?ase)\b', q, re.I)
        if m: name = m.group(1).lower()
    print(f"   → task={task}, accession={acc}, raw_seq={bool(raw)}, name={name}")
    return {'task':task,'accession':acc,'raw_seq':raw,'common_name':name}

# ─── UniProt fetcher ──────────────────────────────────────────────────────
def fetch_uniprot(acc):
    print(f"🔄 Fetching UniProt entry for {acc}…")
    try:
        r = requests.get(
            f'https://rest.uniprot.org/uniprotkb/{acc}.json'
            '?fields=sequence,proteinExistence,feature(MODIFIED_RESIDUE)',
            timeout=15
        )
        r.raise_for_status()
        js = r.json()
        seq = js['sequence']['value']
        pe_txt = js['proteinExistence']
        ptm = sum(1 for f in js.get('features', [])
                  if f.get('type','').lower()=='modified residue')
        print(f"✅ UniProt data: len={len(seq)}, pe={pe_txt}, ptm={ptm}")
        return {'seq':seq,'pe':pe_txt,'ptm':ptm}
    except Exception as e:
        warnings.warn(f"⚠️ UniProt fetch failed for {acc}: {e}")
        return None

# ─── ESM-2 embedder ───────────────────────────────────────────────────────
print("📦 Loading ESM-2 embedder…")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model.eval().to(DEVICE)
batch_converter = alphabet.get_batch_converter()
print("✅ ESM-2 ready")

def embed(seq):
    print(f"🔗 Embedding sequence (len={len(seq)})…")
    _, _, toks = batch_converter([('Q', seq)])
    toks = toks.to(DEVICE)
    with torch.no_grad():
        reps = esm_model(toks, repr_layers=[6])['representations'][6]
    print("✅ Embedding complete")
    return reps.mean(1)

# ─── Load tiny heads ───────────────────────────────────────────────────────
pe_head, ptm_head = load_heads(DEVICE)
print("✅ Loaded tiny_heads models")

# ─── Main answer function ─────────────────────────────────────────────────
def answer(q):
    info = parse_question(q)
    if info['common_name'] and not info['accession']:
        info['accession'] = search_uniprot_name(info['common_name'])

    up = fetch_uniprot(info['accession']) if info['accession'] else None
    seq = up['seq'] if up else info['raw_seq']
    if not seq:
        return '⚠️ Need accession or sequence to proceed.'

    emb = embed(seq)
    if info['task']=='protein_existence':
        if up:
            return f"📖 UniProt reports: {up['pe'].lower()}."
        pred = pe_head(emb).softmax(-1).argmax(-1).item()
        return f"🤖 I predict level {pred+1} ({PE_TXT[pred]})."
    else:
        if up:
            return f"📖 UniProt lists {up['ptm']} PTM sites."
        est = torch.expm1(ptm_head(emb)).item()
        return f"🤖 I predict ~{est:.1f} PTM sites."

# ─── Demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":   
    qs = [
        "What is the protein existence level of lactase?",
        "How many predicted PTM sites are there for P09812?",
        "PTM count for MVHFAELVK?",
        "Level of protein existence for ACDEFGHIKL?"
    ]
    for q in qs:
>>>>>>> 0a2edc19e048b73bb4e4255270613a728ee264b6
        print("▶", q)
        print("→", answer(q), "\n")
