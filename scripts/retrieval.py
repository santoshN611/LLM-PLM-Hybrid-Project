#!/usr/bin/env python3
import requests, warnings
from functools import lru_cache

TAXON_MAP = {
    'human':       9606,
    'mouse':       10090,
    'yeast':       559292,
    'ecoli':       562,
    'fly':         7227,
    'arabidopsis': 3702,
}

@lru_cache(maxsize=128)
def search_uniprot_name(name, organism=None):
    base = "https://rest.uniprot.org/uniprotkb/search"
    clauses = [f"protein_name:{name}", "reviewed:true"]
    if organism:
        if isinstance(organism,int):
            clauses.append(f"organism_id:{organism}")
        else:
            clauses.append(f'organism:""{organism}""')
    query = " AND ".join(clauses)
    def do_q(qstr):
        r = requests.get(base, params={
            "query": qstr, "format":"json", "size":1,
            "fields":"accession,organism_name"
        }, timeout=10)
        r.raise_for_status()
        return r.json().get("results",[])
    try:
        print(f"🔍 Strict search '{name}'…")
        res = do_q(query)
        if not res:
            warnings.warn(f"⚠️ Strict failed for '{name}', retrying…")
            res = do_q(f'"{name}" AND reviewed:true'
                       + (f" AND organism_id:{organism}" if organism else ""))
        if not res:
            print(f"⚠️ No UniProt entry for '{name}'")
            return None
        hit = res[0]
        acc = hit.get("accession")
        print(f"✅ Found {acc} for '{name}'")
        return acc
    except Exception as e:
        warnings.warn(f"⚠️ UniProt search error: {e}")
        return None
