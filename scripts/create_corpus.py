#!/usr/bin/env python3
"""
create_corpus.py

Produces:
  - classification.csv: accession,sequence,existence_level (1–5)
  - regression.csv:   accession,sequence,ptm_site_count

1) Try UniProt’s TSV “stream” endpoint (all records at once).
2) If that fails (403/400), fall back to JSON “search” endpoint,
   following the 'next' URL in the Link header for true pagination.
"""

import requests, csv, time
from requests.utils import parse_header_links

# ── Map UniProt text to numeric existence levels ─────────────────────────────
PE_MAP = {
    "Evidence at protein level":    1,
    "Evidence at transcript level": 2,
    "Inferred from homology":       3,
    "Predicted":                    4,
    "Uncertain":                    5,
}

# ── STREAM mode: TSV dump ───────────────────────────────────────────────────
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
        missing = [f for f in STREAM_FIELDS if f not in idx]
        if missing:
            raise RuntimeError(f"Stream TSV missing columns: {missing}")
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

# ── FALLBACK: JSON search w/ Link-header pagination ────────────────────────
SEARCH_URL     = "https://rest.uniprot.org/uniprotkb/search"
SEARCH_HEADERS = {"User-Agent": "Mozilla/5.0"}

def search_paginated_uniprot(max_pages=None):
    """
    Yield records via the JSON search endpoint, following 'next' in Link.
    max_pages: int or None.  None => no page limit (fetch until no more 'next').
    """
    url = SEARCH_URL
    params = {
        "query": "*",
        "format":"json",
        "size":  500,
        "fields": ",".join(STREAM_FIELDS),
    }
    page = 0

    # Loop until we run out of pages, or hit max_pages (if specified)
    while url and (max_pages is None or page < max_pages):
        r = requests.get(url, params=params, headers=SEARCH_HEADERS, timeout=30)
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
                "protein_existence": e.get("proteinExistence", ""),
                "ft_mod_res": ";".join(
                    f.get("description","")
                    for f in e.get("features", [])
                    if f.get("type","").lower() == "modified residue"
                )
            }
        page += 1

        # parse Link header for next page
        link_hdr = r.headers.get("Link","")
        links    = parse_header_links(link_hdr.rstrip(","))
        next_link = next((L["url"] for L in links if L.get("rel")=="next"), None)
        if not next_link:
            break

        url    = next_link
        params = {}  # baked into next_link
        time.sleep(1)

    if max_pages is None:
        print(f"✅ Finished JSON search — fetched all {page} pages.")
    else:
        print(f"✅ Finished JSON search — up to {page} pages.")

# ── Helper: parse existence level from either plain text or "N: text" ────────
def parse_existence_level(pe_txt):
    """
    Convert a protein existence string into an integer level.
    Handles both:
      - "Evidence at protein level"
      - "1: Evidence at protein level"
    """
    if not pe_txt:
        return 0
    parts = pe_txt.split(":", 1)
    if parts[0].isdigit():
        return int(parts[0])
    return PE_MAP.get(pe_txt, 0)

# ── MAIN: write CSVs, dedupe by accession ──────────────────────────────────
def main():
    w_clf = csv.writer(open("classification.csv","w",newline=""))
    w_reg = csv.writer(open("regression.csv",  "w",newline=""))
    w_clf.writerow(["accession","sequence","existence_level"])
    w_reg.writerow(["accession","sequence","ptm_site_count"])

    # 1) Try TSV stream first
    try:
        print("🔄 Attempting STREAM endpoint…")
        seen, count = set(), 0
        for rec in stream_all_uniprot():
            acc = rec["accession"]
            if acc in seen:
                continue
            seen.add(acc)

            pe_level  = parse_existence_level(rec["protein_existence"])
            ft         = rec["ft_mod_res"].strip()
            ptm_count  = len([x for x in ft.split(";") if x.strip()]) if ft else 0

            w_clf.writerow([acc, rec["sequence"], pe_level])
            w_reg.writerow([acc, rec["sequence"], ptm_count])
            count += 1

        print(f"✅ STREAM succeeded: wrote {count} unique records.")
        return

    except requests.exceptions.HTTPError as e:
        print(f"❌ STREAM failed ({e}), falling back to JSON search…")

    # 2) Fall back to JSON search + Link-header pagination
    seen, count = set(), 0
    # Pass max_pages=None to fetch all pages
    for rec in search_paginated_uniprot(max_pages=1100):
        acc = rec["accession"]
        if acc in seen:
            continue
        seen.add(acc)

        pe_level  = parse_existence_level(rec["protein_existence"])
        ft         = rec["ft_mod_res"].strip()
        ptm_count  = len([x for x in ft.split(";") if x.strip()]) if ft else 0

        w_clf.writerow([acc, rec["sequence"], pe_level])
        w_reg.writerow([acc, rec["sequence"], ptm_count])
        count += 1

    print(f"✅ JSON search fallback: wrote {count} unique records.")

    import pandas as pd
    import matplotlib.pyplot as plt

    # 1) Load your files
    df_clf = pd.read_csv("classification.csv")   # accession, sequence, existence_level
    df_reg = pd.read_csv("regression.csv")       # accession, sequence, ptm_site_count

    # 2) Classification.csv: sequence‐length spread + duplicates
    print("=== classification.csv ===")
    print("Total rows:", len(df_clf))
    print("Duplicate accessions:", df_clf['accession'].duplicated().sum())
    print("Duplicate full rows:", df_clf.duplicated().sum())

    # Sequence‐length distribution:
    seq_lens = df_clf['sequence'].str.len()
    print("\nSequence length (aa) summary:")
    print(seq_lens.describe())

    # # Optional histogram:
    # plt.hist(seq_lens, bins=50)
    # plt.title("Histogram of protein sequence lengths")
    # plt.xlabel("Length (aa)")
    # plt.ylabel("Count")
    # plt.show()


    # 3) Regression.csv: PTM‐count spread + duplicates
    print("\n=== regression.csv ===")
    print("Total rows:", len(df_reg))
    print("Duplicate accessions:", df_reg['accession'].duplicated().sum())
    print("Duplicate full rows:", df_reg.duplicated().sum())

    # PTM‐site‐count distribution:
    ptm_counts = df_reg['ptm_site_count']
    print("\nPTM site count summary:")
    print(ptm_counts.describe())

if __name__=="__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"🕐 Total time taken: {((end-start)/60):.2f} minutes")
