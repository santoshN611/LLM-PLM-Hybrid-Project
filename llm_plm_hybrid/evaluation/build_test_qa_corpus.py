import csv
import json
import requests
from pathlib import Path
from typing import List, Set
from tqdm import tqdm

# UniProt REST endpoint
ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"

# variant templates
EXISTENCE_TEMPLATES = [
    "What is the existence level of {acc}?",
    "What evidence supports the existence of {acc}?",
    "Give me the protein existence classification for {acc}.",
    "According to UniProt, at which evidence level does {acc} stand?",
    "Which protein-existence category does {acc} fall into?",
    "How is {acc}'s existence annotated in UniProt?",
    "UniProt reports what level of existence for {acc}?",
    "On what basis is the existence of {acc} classified?",
    "State the UniProt protein‐existence level for {acc}.",
    "By what evidence level is {acc} marked in UniProt?"
]

PTM_TEMPLATES = [
    "How many PTM sites for {acc}?",
    "How many post-translational modification sites does {acc} have?",
    "Number of modified residues reported for {acc}?",
    "What is the total count of PTM annotations in {acc}?",
    "UniProt lists how many modified residues for {acc}?",
    "How many ‘Modified residue’ features appear in {acc}?",
    "Tell me the PTM site count for {acc}.",
    "What’s the number of PTM sites annotated on {acc}?",
    "Provide the count of post-translational modifications in {acc}.",
    "List how many PTM annotations are present for {acc}."
]

def load_accessions(*csv_paths: str) -> List[str]:
    """Read test‐split CSVs and return sorted unique accessions."""
    acs: Set[str] = set()
    for path in csv_paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                acs.add(row["accession"])
    return sorted(acs)

def fetch_uniprot_json(acc: str) -> dict:
    """Fetch full UniProt JSON for accession `acc`."""
    url = ENTRY_URL.format(acc=acc)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def parse_entry(jsond: dict) -> dict:
    """
    Extract from JSON:
      - existence: full 'proteinExistence' string
      - ptm_count: count of features type 'Modified residue'
    """
    pe = jsond.get("proteinExistence", "")
    if isinstance(pe, dict):
        pe = pe.get("value", pe.get("text", ""))
    existence = pe.strip() or "Unknown"
    feats = jsond.get("features", [])
    ptm_count = sum(
        1 for f in feats
        if f.get("type", "").lower() == "modified residue"
    )
    return {"existence": existence, "ptm_count": ptm_count}

def build_corpus(accessions: List[str], out_path: str):
    """
    For each accession, generate paraphrased questions from the templates,
    fetch & parse UniProt JSON, and write JSONL entries with IDs, questions, and answers.
    """
    Path(out_path).parent.mkdir(exist_ok=True)
    idx = 1
    with open(out_path, "w", encoding="utf-8") as fout:
        for acc in tqdm(accessions[:1000]): # only 1000 due to time limitations
            data = fetch_uniprot_json(acc)
            info = parse_entry(data)

            # existence‐level questions
            for tmpl in EXISTENCE_TEMPLATES:
                q = tmpl.format(acc=acc)
                a = info["existence"]
                fout.write(json.dumps({"id": idx, "question": q, "answer": a}) + "\n")
                idx += 1

            # PTM‐count questions
            for tmpl in PTM_TEMPLATES:
                q = tmpl.format(acc=acc)
                a = str(info["ptm_count"])
                fout.write(json.dumps({"id": idx, "question": q, "answer": a}) + "\n")
                idx += 1

    print(f"✅ Wrote {idx-1} Q&A entries to {out_path}")

if __name__ == "__main__":
    # 1) Load all test accessions
    data_dir = Path(__file__).resolve().parent.parent / "data"
    test_accs = load_accessions(
        data_dir / "classification_test.csv",
        data_dir / "regression_test.csv"
    )
    print(f"📂 Loaded {len(test_accs)} unique accessions from test splits")

    # 2) Build and save the expanded JSONL corpus
    out_path = Path(__file__).resolve().parent / "test_protein_qa.jsonl"
    build_corpus(test_accs, out_path=out_path)
    print(f"🎉 Created {out_path} successfully!")
