#!/usr/bin/env python3
import os, re, torch, esm, faiss, requests, pandas as pd
from transformers import (
    BertTokenizerFast, BertForSequenceClassification, logging as hf_log
)
from tiny_heads import load_heads
from retrieval import search_uniprot_name, TAXON_MAP

# ─── Silence logs ─────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
hf_log.set_verbosity_error()

# ─── Normalization ────────────────────────────────────────────────
ALIASES = {
    r"\be[.\s]?coli\b":"ecoli",
    r"\bdna[ \-]?k\b":"dnak",
}
def normalize(q):
    nq = q.lower()
    for pat,rep in ALIASES.items():
        nq = re.sub(pat,rep,nq,flags=re.IGNORECASE)
    nq = re.sub(r"[^a-z0-9\s]"," ",nq)
    return re.sub(r"\s+"," ",nq).strip()

species_keys = sorted(map(re.escape,TAXON_MAP),key=len,reverse=True)
ORG_RE = re.compile(rf"\b({'|'.join(species_keys)})\b\s+([\w\-]+)",re.I)

# ─── Load models ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok  = BertTokenizerFast.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
tok.model_max_length=512
mdl  = BertForSequenceClassification.from_pretrained(
           "dmis-lab/biobert-base-cased-v1.1", num_labels=2
       ).eval().to(DEVICE)
esm_m, alpha = esm.pretrained.esm2_t6_8M_UR50D()
esm_m.eval().to(DEVICE)
batch_converter = alpha.get_batch_converter()
pe_h, ptm_h = load_heads(DEVICE)
PE_TXT = [
    'evidence at protein level',
    'evidence at transcript level',
    'inferred from homology',
    'predicted','uncertain'
]

# ─── Load FAISS index & accessions ────────────────────────────────
print("📦 Loading FAISS index…")
idx = faiss.read_index("embeddings/classification_train.index")
df_acc = pd.read_csv("classification_train.csv")
corpus = df_acc["accession"].tolist()

def embed_seq(seq):
    _,_,toks = batch_converter([("Q",seq)])
    toks = toks.to(DEVICE)
    with torch.no_grad():
        reps = esm_m(toks,repr_layers=[6])["representations"][6]
    return reps.mean(1)

def fetch_uniprot(acc):
    print(f"🔄 Fetching UniProt {acc}…")
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json",timeout=15)
    r.raise_for_status()
    js = r.json()
    seq = js["sequence"]["value"]
    pe  = js["proteinExistence"]
    ptm = sum(1 for f in js.get("features",[])
              if f.get("type","").lower()=="modified residue")
    print(f"✅ UniProt len={len(seq)}, pe={pe}, ptm={ptm}")
    return {"seq":seq,"pe":pe,"ptm":ptm}

def retrieve_neighbors(seq,k=5):
    emb = embed_seq(seq).cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)
    D,I = idx.search(emb,k)
    return [corpus[i] for i in I[0]], D[0]

def parse_question(raw_q):
    print(f"ℹ️ raw : {raw_q}")
    q = normalize(raw_q)
    print(f"ℹ️ norm: {q}")
    m = ORG_RE.search(q)
    organism=name=None
    if m:
        organism=TAXON_MAP[m.group(1).lower()]
        name=m.group(2)
        q=q.replace(m.group(0),"")
    # intent
    inp = tok(raw_q,return_tensors="pt",truncation=True,
              max_length=tok.model_max_length).to(DEVICE)
    logits=mdl(**inp).logits
    task = "protein_existence" if logits.argmax(-1).item()==0 else "ptm_count"
    if "existence" in q: task="protein_existence"
    if "ptm" in q:       task="ptm_count"
    acc=None
    for t in re.split(r"[^A-Za-z0-9]", raw_q):
        if re.fullmatch(r"[OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d[A-Z0-9]{3}\d",t):
            acc=t.upper(); break
    aa_m = re.search(r"([ACDEFGHIKLMNPQRSTVWY]{4,})", raw_q.replace(" ",""))
    raw_seq = aa_m.group(1).upper() if aa_m else None
    if not name and not acc and not raw_seq:
        frag=None
        if " of " in q:  frag=q.split(" of ")[-1]
        if " for " in q: frag=q.split(" for ")[-1]
        if frag:
            fn=normalize(frag)
            m2=ORG_RE.match(fn)
            if m2:
                organism=TAXON_MAP[m2.group(1).lower()]
                name=m2.group(2)
            else:
                name=fn
    print(f"   → task={task}, acc={acc}, raw={'yes' if raw_seq else 'no'}, name={name}, org={organism}")
    return {"task":task,"accession":acc,"raw_seq":raw_seq,"common_name":name,"organism":organism}

def answer(q):
    info=parse_question(q)
    # seed text lookup
    if info["common_name"] and not info["accession"]:
        acc0 = search_uniprot_name(info["common_name"], info["organism"])
        if acc0:
            info["accession"]=acc0
            print(f"ℹ️ seed accession → {acc0}")
    # fallback retrieval only if still no accession
    if info["common_name"] and not info["accession"]:
        print(f"🔍 retrieving neighbors for '{info['common_name']}'…")
        seq0=fetch_uniprot(acc0)["seq"]
        cands,sims=retrieve_neighbors(seq0,5)
        info["accession"]=cands[0]
        print(f"ℹ️ retrieval → {cands[0]} (sim={sims[0]:.3f})")
    uni=None
    if info["accession"]:
        try: uni=fetch_uniprot(info["accession"])
        except Exception as e: return f"⚠️ UniProt fetch failed: {e}"
    if uni:
        if info["task"]=="protein_existence":
            return f"📖 UniProt reports: {uni['pe'].lower()}."
        else:
            return f"📖 UniProt lists {uni['ptm']} PTM sites."
    if info["raw_seq"]:
        emb=embed_seq(info["raw_seq"])
        if info["task"]=="protein_existence":
            lvl=pe_h(emb).softmax(-1).argmax(-1).item()
            return f"🤖 I predict level {lvl+1} ({PE_TXT[lvl]})."
        est=float(torch.expm1(torch.tensor(ptm_h(emb).item())))
        return f"🤖 I predict ~{est:.1f} PTM sites."
    return "⚠️ Need a UniProt accession or raw AA sequence."

if __name__=="__main__":
    for q in [
        "What is the protein existence level of lactase?",
        "How many predicted PTM sites are there for P09812?",
        "PTM count for MVHFAELVK?",
        "Level of protein existence for ACDEFGHIKL?",
        "What is the existence level of mouse hemoglobin?",
        "PTM count for E. coli DnaK?"
    ]:
        print("▶",q)
        print("→",answer(q),"\n")
