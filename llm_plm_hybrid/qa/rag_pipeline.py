import re
import torch
import requests
import spacy

from transformers import (
    BertTokenizerFast,
    BloomForCausalLM,
    BloomTokenizerFast,
)
from llm_plm_hybrid.embeddings.generate_embeddings import embed_sequence
from llm_plm_hybrid.retrieval.retrieval_utils import (
    search_neighbors,
    build_context_block,
)
from llm_plm_hybrid.qa.adapters import BioBERTWithAdapters

# regex identifiers, no longer used
ACC_RE = re.compile(r'\b[A-Z][0-9][A-Z0-9]{3}[0-9]\b', re.IGNORECASE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# bioBERT adapters
print(" Loading BioBERT+Adapters for QA…")
tok_qa   = BertTokenizerFast.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
model_qa = BioBERTWithAdapters().to(DEVICE).eval()
print("✅ BioBERT+Adapters ready")

# bloom
print(" Loading BLOOM tokenizer & model…")
bloom_tok   = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
bloom_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m") \
                              .to(DEVICE).eval()
print("✅ BLOOM ready")

# named entity recognition for parsing
nlp = spacy.load("en_ner_bc5cdr_md")

# truncate tokens to maximum length if needed
def _gen_bloom(toks, new_tokens: int = 128):
    cfg = bloom_model.config
    max_pos = getattr(cfg, "max_position_embeddings", None) \
              or getattr(cfg, "n_positions", None) or 2048
    cur_len = toks["input_ids"].shape[-1]
    room    = max_pos - new_tokens
    if cur_len > room:
        toks["input_ids"]      = toks["input_ids"][:, -room:]
        toks["attention_mask"] = toks["attention_mask"][:, -room:]
    return bloom_model.generate(**toks, max_new_tokens=new_tokens)

def answer(q: str) -> str:
    # answering function
    q_low = q.lower()

    m_seq_any = re.search(r'([ACDEFGHIKLMNPQRSTVWY]{6,})', q.strip())
    if m_seq_any:
        seq  = m_seq_any.group(1)
        emb  = embed_sequence(seq)
        nbrs = search_neighbors(emb, k=5)
        ctx  = build_context_block(nbrs)

        # look at closest neighbors from embedding space from FAISS index
        ptm_counts = []
        pe_levels  = []
        for acc in nbrs:
            try:
                js = requests.get(
                    f'https://rest.uniprot.org/uniprotkb/{acc}.json',
                    timeout=10
                ).json()
                ptm_counts.append(
                    sum(1 for f in js.get("features", [])
                        if f.get("type") == "MOD_RES")
                )
                pe_levels.append(
                    int(str(js.get("proteinExistence", "0"))[0])
                )
            except Exception:
                pass

        mean_ptm = round(sum(ptm_counts)/len(ptm_counts)) if ptm_counts else 0
        maj_pe   = max(set(pe_levels), key=pe_levels.count) if pe_levels else 5

        # slight prompt engineering
        if any(k in q_low for k in ("existence", "level", "evidence")):
            pred_num      = str(maj_pe)
            base_sentence = f"The predicted UniProt existence level is {pred_num}."
        else:
            pred_num      = str(mean_ptm)
            plural        = "" if pred_num == "1" else "s"
            base_sentence = f"The predicted number of PTM sites is {pred_num}{plural}."

        # construct the rest of the prompt
        prompt = base_sentence + "\n\n" + ctx + "\n\nExplanation:"
        toks   = bloom_tok(prompt, return_tensors="pt",
                           truncation=True, padding=True).to(DEVICE)
        generated = bloom_tok.decode(_gen_bloom(toks)[0],
                                     skip_special_tokens=True)

        return base_sentence + generated[len(base_sentence):]


    m   = ACC_RE.search(q)
    accession = m.group(0).upper() if m else None

    entry = None
    if accession:
        try:
            resp = requests.get(
                f'https://rest.uniprot.org/uniprotkb/{accession}.json',
                timeout=10
            ); resp.raise_for_status()
            entry = resp.json()
        except requests.HTTPError:
            accession = None

    if entry is None:
        doc  = nlp(q); ents = [e.text for e in doc.ents]
        if ents:
            name = ents[0]
            params = {
                "query": f'protein_name:"{name}"',
                "format":"json", "size":1,
                "fields":"accession,proteinExistence,features"
            }
            try:
                r = requests.get("https://rest.uniprot.org/uniprotkb/search",
                                 params=params, timeout=10)
                r.raise_for_status()
                results = r.json().get("results", [])
                if results:
                    entry     = results[0]
                    accession = entry.get("accession", name)
            except requests.RequestException:
                entry = None

    if entry:
        pe_field = entry.get("proteinExistence", "Unknown")
        pe = pe_field.get("evidenceType", pe_field) if isinstance(pe_field, dict) else pe_field
        feats = entry.get("features", [])
        ptm  = sum(1 for f in feats if f.get("type") == "MOD_RES")

        if any(k in q_low for k in ("existence","level","evidence")):
            base_sentence = f"The predicted UniProt existence level is {pe}."
        else:
            plural = "" if ptm == 1 else "s"
            base_sentence = f"The predicted number of PTM sites is {ptm}{plural}."

        toks = bloom_tok(base_sentence, return_tensors="pt",
                         truncation=True, padding=True).to(DEVICE)
        return bloom_tok.decode(_gen_bloom(toks)[0], skip_special_tokens=True)

    return "Sorry, I couldn't understand that query."
