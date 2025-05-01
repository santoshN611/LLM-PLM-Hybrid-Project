import json, csv, random, re, requests, numpy as np
from collections import Counter
from pathlib     import Path
from tqdm        import tqdm
import time
import torch

from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats     import spearmanr
from evaluate        import load as load_metric
import nltk; nltk.download("punkt", quiet=True)

from llm_plm_hybrid.qa.rag_pipeline               import answer, ACC_RE
from llm_plm_hybrid.embeddings.generate_embeddings import embed_sequence
from llm_plm_hybrid.retrieval.retrieval_utils      import load_index, search_neighbors

from llm_plm_hybrid.embeddings.tiny_heads import load_heads

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pe_model, ptm_model = load_heads(device=DEVICE)

def is_ptm_seq(e): return e.get("label") == "ptm_seq"
def is_pe_seq(e):  return e.get("label") == "pe_seq"

SEQ_EXTRACT = re.compile(r"([ACDEFGHIKLMNPQRSTVWY]{6,})")
INT_RE      = re.compile(r"\d+")

# failsafe for some NaN problems
def safe_int(x):
    try: return int(x)
    except Exception: return np.nan

def parse_pe_str(txt):
    if txt is None: return np.nan
    m = INT_RE.match(str(txt).strip())
    return int(m.group()) if m else np.nan

SPLIT = "test"
# SPLIT = "val"
JSONL = Path(__file__).parent / f"{SPLIT}_protein_qa.jsonl"

all_entries = [json.loads(l) for l in JSONL.open()]
for e in all_entries:
    if "gold_num"  not in e: e["gold_num"]  = e["answer"]
    if "gold_text" not in e: e["gold_text"] = e["answer"]

random.seed(42)
entries = random.sample(all_entries, 750)

ids, questions, gold_texts, gold_nums = zip(*[
    (e["id"], e["question"], e["gold_text"], e["gold_num"]) for e in entries
])

pred_answers, retrieved_ranks = [], []
mean_ptm_5nn,   maj_pe_5nn    = [], []
ptm_gold, ptm_pred            = {}, {}
pe_gold,  pe_pred             = {}, {}

tiny_ptm_preds = {}
tiny_pe_preds  = {}

# load faiss
emb_dir  = Path(__file__).parent.parent / "embeddings"
index, _ = load_index(emb_dir/"combined_train.index",
                      emb_dir/"combined_train.meta.npy")
train_npz = np.load(emb_dir/"combined_train.npz", allow_pickle=True)
accs  = train_npz["meta"];   emb_X = train_npz["X"].squeeze(1).astype("float32")
idx_map = {a:i for i,a in enumerate(accs)}

# main function
total_response_time = 0.0
for e in tqdm(entries, desc=f"Evaluating {SPLIT} Q&A"):
    start = time.time()
    q, gold_num = e["question"], e["gold_num"]
    pred   = answer(q)
    pred_answers.append(pred)

    if is_ptm_seq(e):
        ptm_gold[e["id"]] = safe_int(gold_num)
        

        m = INT_RE.search(pred); ptm_pred[e["id"]] = safe_int(m.group()) if m else np.nan
        seq = SEQ_EXTRACT.search(e["question"]).group(1)
        tiny_ptm_preds[e["id"]] = ptm_model.predict(seq, device=DEVICE)
    elif is_pe_seq(e):
        pe_gold[e["id"]]  = safe_int(gold_num)
        

        m = INT_RE.search(pred); pe_pred[e["id"]]  = safe_int(m.group()) if m else np.nan
        seq = SEQ_EXTRACT.search(e["question"]).group(1)
        tiny_pe_preds[e["id"]] = pe_model.predict(seq, device=DEVICE)

    # look at neighbors for metrics
    if (is_ptm_seq(e) or is_pe_seq(e)) and (m := SEQ_EXTRACT.search(q)):
        emb = embed_sequence(m.group(1))
        nbrs = search_neighbors(emb, k=5)
        ptm_counts, pe_levels = [], []
        for acc in nbrs:
            try:
                d = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json",
                                 timeout=10).json()
                ptm_counts.append(sum(1 for f in d.get("features", [])
                                      if f.get("type") == "MOD_RES"))
                pe_levels.append(parse_pe_str(d.get("proteinExistence")))
            except Exception:
                continue
        mean_ptm_5nn.append(np.mean(ptm_counts) if ptm_counts else np.nan)
        maj_pe_5nn.append(Counter(pe_levels).most_common(1)[0][0]
                          if pe_levels else np.nan)
    else:
        mean_ptm_5nn.append(np.nan); maj_pe_5nn.append(np.nan)

    # retrieval rank not working, but not useful for us anyway
    # gonna leave it in here for fear of breaking any code
    if (m := ACC_RE.search(q)):
        acc = m.group(0)
        if acc in idx_map:
            _, I = index.search(emb_X[idx_map[acc]].reshape(1,-1), 10)
            retrieved_ranks.append(I[0].tolist().index(idx_map[acc])+1
                                    if idx_map[acc] in I[0] else None)
        else:
            retrieved_ranks.append(None)
    else:
        retrieved_ranks.append(None)

    end = time.time()
    total_response_time += (end-start)

print(f"⏰ Average response time: {total_response_time / len(entries)}")

#metrics
bleu   = load_metric("bleu").compute(predictions=pred_answers,
                                     references=[[t] for t in gold_texts])["bleu"]
rouge1 = load_metric("rouge").compute(predictions=pred_answers,
                                      references=[[t] for t in gold_texts])["rouge1"]
meteor = load_metric("meteor").compute(predictions=pred_answers,
                                       references=[[t] for t in gold_texts])["meteor"]
berts  = load_metric("bertscore").compute(predictions=pred_answers,
                                          references=[[t] for t in gold_texts],
                                          lang="en")["f1"]

print(f"\nBLEU      : {bleu:.3f}")
print(f"ROUGE-1   : {rouge1:.3f}")
print(f"METEOR    : {meteor:.3f}")
print(f"BERTScore : {np.mean(berts):.3f}")

valid = [r for r in retrieved_ranks if r]
if valid:
    prec1 = sum(1 for r in valid if r==1)/len(valid)
    mrr   = np.mean([1/r for r in valid])
    print(f"\nRetrieval  Precision@1={prec1:.3f} | MRR={mrr:.3f}")
else:
    print("\nRetrieval  (skipped - no accessions in this split)")

# more metrics
if ptm_gold:
    g = np.array(list(ptm_gold.values()), dtype=float)
    p = np.array([ptm_pred[i] for i in ptm_gold], dtype=float)
    k = np.array([mean_ptm_5nn[ids.index(i)] for i in ptm_gold], dtype=float)
    mask_m = ~np.isnan(g)&~np.isnan(p); mask_k = ~np.isnan(g)&~np.isnan(k)
    tp = np.array([tiny_ptm_preds[i] for i in ptm_gold], dtype=float)
    mask_t = ~np.isnan(g)&~np.isnan(tp)
    print(f"PTM tiny-head (N={mask_t.sum()}) MAE={mean_absolute_error(g[mask_t], tp[mask_t]):.2f}")
    if mask_m.any():
        print(f"\nPTM Model  (N={mask_m.sum()})  MAE={mean_absolute_error(g[mask_m],p[mask_m]):.2f} | ρ=n/a")
    else: print("\nPTM Model  (no usable examples)")
    if mask_k.any():
        print(f"PTM 5-NN   (N={mask_k.sum()})  MAE={mean_absolute_error(g[mask_k],k[mask_k]):.2f} | ρ=n/a")
    else: print("PTM 5-NN   (no usable examples)")

if pe_gold:
    g = np.array(list(pe_gold.values()), dtype=float)
    p = np.array([pe_pred[i] for i in pe_gold], dtype=float)
    k = np.array([maj_pe_5nn[ids.index(i)] for i in pe_gold], dtype=float)
    mask_m = ~np.isnan(g)&~np.isnan(p); mask_k = ~np.isnan(g)&~np.isnan(k)
    tp = np.array([tiny_pe_preds[i] for i in pe_gold], dtype=float)
    mask_t = ~np.isnan(g)&~np.isnan(tp)
    print(f"PTM tiny-head (N={mask_t.sum()}) MAE={mean_absolute_error(g[mask_t], tp[mask_t]):.2f}")
    if mask_m.any():
        print(f"\nPE  Model  (N={mask_m.sum()})  Acc={accuracy_score(g[mask_m],p[mask_m]):.3f}")
    else: print("\nPE  Model  (no usable examples)")
    if mask_k.any():
        print(f"PE 5-NN    (N={mask_k.sum()})  Acc={accuracy_score(g[mask_k],k[mask_k]):.3f}")
    else: print("PE 5-NN    (no usable examples)")

# save answers
with Path(f"eval_results_{SPLIT}.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id","question","gold_num","gold_text","pred",
                     "retrieval_rank","mean_ptm_5NN","maj_pe_5NN", "tiny_ptm_pred", "tiny_pe_pred"])
    for rid, q, gn, gt, pa, rr, m5, p5 in zip(
            ids, questions, gold_nums, gold_texts,
            pred_answers, retrieved_ranks, mean_ptm_5nn, maj_pe_5nn):
        writer.writerow([
            rid, q, gn, gt, pa, rr, m5, p5,
            tiny_ptm_preds.get(rid, ""),
            tiny_pe_preds.get(rid, ""),
        ])

print(f"\n✅ {SPLIT.capitalize()} evaluation complete - results saved.")
