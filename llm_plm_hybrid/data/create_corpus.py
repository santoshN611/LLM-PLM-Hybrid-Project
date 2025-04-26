"""
create_corpus.py

Produces into data/:
  - classification.csv: accession,sequence,existence_level (1–5)
  - regression.csv:   accession,sequence,ptm_site_count

First tries the TSV “stream” endpoint; falls back to JSON search with pagination.
"""

import os
import requests, csv, time
from requests.utils import parse_header_links
from pathlib import Path

# existence level map to human language
PE_MAP = {
    "Evidence at protein level":    1,
    "Evidence at transcript level": 2,
    "Inferred from homology":       3,
    "Predicted":                    4,
    "Uncertain":                    5,
}

# STREAM mode first (more efficient than paging, but tends to fail)
STREAM_URL     = "https://rest.uniprot.org/uniprotkb/stream"
STREAM_FIELDS  = ["accession","sequence","protein_existence","ft_mod_res"]
STREAM_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept":      "text/tab-separated-values",
}

def stream_all_uniprot():
    """Yield all records via the TSV 'stream' endpoint."""
    params = {
        "query":     "*",
        "format":    "tsv",
        "fields":    ",".join(STREAM_FIELDS),
        "compressed":"true",
    }
    with requests.get(STREAM_URL, params=params,
                      headers=STREAM_HEADERS, stream=True, timeout=60) as r:
        r.raise_for_status()
        lines = r.iter_lines(decode_unicode=True)
        header = next(lines).split("\t")
        idx = {col:i for i,col in enumerate(header)}
        for line in lines:
            if not line:
                continue
            cols = line.split("\t")
            yield {
                "accession":         cols[idx["accession"]],
                "sequence":          cols[idx["sequence"]],
                "protein_existence": cols[idx["protein_existence"]],
                "ft_mod_res":        cols[idx["ft_mod_res"]],
            }

# SEARCH pagination
SEARCH_URL     = "https://rest.uniprot.org/uniprotkb/search"
SEARCH_HEADERS = {"User-Agent": "Mozilla/5.0"}

def search_paginated_uniprot(max_pages=None):
    url = SEARCH_URL
    params = {
        "query": "*",
        "format":"json",
        "size":  500,
        "fields": ",".join(STREAM_FIELDS),
    }
    page = 0
    while url and (max_pages is None or page < max_pages):
        r = requests.get(url, params=params,
                         headers=SEARCH_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        print(f"📄 Page {page+1}: {len(results)} entries")
        for e in results:
            yield {
                "accession":         e["primaryAccession"],
                "sequence":          e["sequence"]["value"],
                "protein_existence": e.get("proteinExistence",""),
                "ft_mod_res": ";".join(
                    f.get("description","")
                    for f in e.get("features", [])
                    if f.get("type","").lower()=="modified residue"
                )
            }
        page += 1
        link_hdr = r.headers.get("Link","")
        links    = parse_header_links(link_hdr.rstrip(","))
        next_link = next((L["url"] for L in links if L.get("rel")=="next"), None)
        url = next_link
        params = {}
        time.sleep(1)
    print(f"✅ Finished JSON search — pages fetched: {page}")

def parse_existence_level(pe_txt):
    if not pe_txt:
        return 0
    parts = pe_txt.split(":",1)
    if parts[0].isdigit():
        return int(parts[0])
    return PE_MAP.get(pe_txt, 0)

def main():
    data_dir = Path(__file__).resolve().parent
    os.makedirs(data_dir, exist_ok=True)

    clf_path = data_dir / "classification.csv"
    reg_path = data_dir / "regression.csv"

    w_clf = csv.writer(open(clf_path, "w", newline=""))
    w_reg = csv.writer(open(reg_path, "w", newline=""))
    w_clf.writerow(["accession","sequence","existence_level"])
    w_reg.writerow(["accession","sequence","ptm_site_count"])

    try:
        print("🔄 Attempting STREAM endpoint…")
        seen = set(); count = 0
        for rec in stream_all_uniprot():
            acc = rec["accession"]
            if acc in seen: continue
            seen.add(acc)
            pe = parse_existence_level(rec["protein_existence"])
            ft = rec["ft_mod_res"].strip()
            ptm = len([x for x in ft.split(";") if x.strip()]) if ft else 0
            w_clf.writerow([acc, rec["sequence"], pe])
            w_reg.writerow([acc, rec["sequence"], ptm])
            count += 1
        print(f"✅ STREAM succeeded: wrote {count} records to {clf_path} & {reg_path}.")
        return
    except Exception as e:
        print(f"❌ STREAM failed ({e}), falling back to JSON…")

    seen = set(); count = 0
    for rec in search_paginated_uniprot(max_pages=5):
        acc = rec["accession"]
        if acc in seen: continue
        seen.add(acc)
        pe = parse_existence_level(rec["protein_existence"])
        ft = rec["ft_mod_res"].strip()
        ptm = len([x for x in ft.split(";") if x.strip()]) if ft else 0
        w_clf.writerow([acc, rec["sequence"], pe])
        w_reg.writerow([acc, rec["sequence"], ptm])
        count += 1
    print(f"✅ JSON fallback succeeded: wrote {count} records to {clf_path} & {reg_path}.")

if __name__=="__main__":
    main()
