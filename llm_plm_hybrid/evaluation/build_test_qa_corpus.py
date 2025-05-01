

import csv, json, requests
from pathlib import Path
from typing  import List, Set
from tqdm    import tqdm

ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"

SEQ_EXISTENCE_TEMPLATES = [
    "Predict the existence level for this protein: {seq}",
    "What UniProt evidence level would you assign to the sequence {seq}?",
    "Estimate the protein-existence class of {seq}.",
    "Give the probable existence level for {seq}.",
    "Which UniProt PE tier best fits {seq}?"
]

SEQ_PTM_TEMPLATES = [
    "Predict how many PTM sites the sequence {seq} might have.",
    "Give an estimated PTM site count for {seq}.",
    "How many 'Modified residue' features do you expect in {seq}?",
    "Roughly how many PTM sites are present in {seq}?",
    "Estimate the total PTM annotations for {seq}."
]

def load_accessions(*csv_paths: str) -> List[str]:
    seen: Set[str] = set()
    for p in csv_paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                seen.add(r["accession"])
    return sorted(seen)


def fetch_uniprot_json(acc: str) -> dict:
    r = requests.get(ENTRY_URL.format(acc=acc), timeout=10)
    r.raise_for_status()
    return r.json()


def parse_entry(js: dict) -> dict:
    pe = js.get("proteinExistence", "")
    if isinstance(pe, dict):
        pe = pe.get("value", pe.get("text", ""))
    feats = js.get("features", [])
    ptm   = sum(1 for f in feats if f.get("type", "").lower() == "modified residue")
    return {"existence": pe.strip() or "Unknown", "ptm_count": ptm}


def build_corpus(accessions: List[str], out_path: Path):
    out_path.parent.mkdir(exist_ok=True)
    idx = 1
    with out_path.open("w", encoding="utf-8") as fout:
        csv_map = {}
        data_dir = Path(__file__).resolve().parent.parent / "data"
        for fname in ("classification_test.csv", "regression_test.csv"):
            with (data_dir / fname).open() as f:
                for row in csv.DictReader(f):
                    acc = row["accession"]
                    csv_map.setdefault(acc, {})["seq"] = row["sequence"]
                    if fname.startswith("classification"):
                        csv_map[acc]["pe"]  = int(row["existence_level"])
                    else:
                        csv_map[acc]["ptm"] = int(row["ptm_site_count"])

        for acc in tqdm(accessions):
            seq      = csv_map[acc]["seq"]
            true_pe  = csv_map[acc]["pe"]
            true_ptm = csv_map[acc]["ptm"]

            for tmpl in SEQ_EXISTENCE_TEMPLATES:
                q = tmpl.format(seq=seq)
                num  = str(true_pe)
                text = f"The predicted UniProt existence level is {num}."
                fout.write(json.dumps({
                    "id":        idx,
                    "question":  q,
                    "gold_num":  num,
                    "gold_text": text,
                    "label":     "pe_seq"
                }) + "\n")
                idx += 1

            for tmpl in SEQ_PTM_TEMPLATES:
                q    = tmpl.format(seq=seq)
                num  = str(true_ptm)
                text = f"This sequence is predicted to have {num} PTM site" \
                       f"{'s' if true_ptm!=1 else ''}."
                fout.write(json.dumps({
                    "id":        idx,
                    "question":  q,
                    "gold_num":  num,
                    "gold_text": text,
                    "label":     "ptm_seq"
                }) + "\n")
                idx += 1

    print(f"✅ Wrote {idx-1} entries ➜ {out_path}")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    accs = load_accessions(data_dir/"classification_test.csv",
                           data_dir/"regression_test.csv")
    build_corpus(accs, Path(__file__).resolve().parent / "test_protein_qa.jsonl")
