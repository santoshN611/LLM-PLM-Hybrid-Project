#!/usr/bin/env python3
import requests
<<<<<<< HEAD
import warnings
from functools import lru_cache

# ── Map common organism names to taxonomy IDs ───────────────────────────────
TAXON_MAP = {
    'human':        9606,
    'mouse':        10090,
    'yeast':        559292,
    'ecoli':        562,
    'fly':          7227,
    'arabidopsis':  3702,
}

@lru_cache(maxsize=128)
def search_uniprot_name(name, organism=None):
    """
    🔍 Query UniProt for a reviewed accession by protein common name,
    optionally restricted to a specific organism.
    Falls back from strict protein_name:name → generic "name" if needed.
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    # Build clauses
    clauses = [f'protein_name:{name}', 'reviewed:true']
    if organism:
        if isinstance(organism, int):
            clauses.append(f"organism_id:{organism}")
        else:
            clauses.append(f'organism:"{organism}"')
    strict_query = " AND ".join(clauses)

    def run_query(query_str):
        params = {
            "query": query_str,
            "format": "json",
            "size":   1,
            "fields": "accession,organism_name"
        }
        r = requests.get(base_url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])

    try:
        print(f"🔍 Strict UniProt search for '{name}'"
              + (f" in organism {organism}" if organism else "") + "…")
        results = run_query(strict_query)

        if not results:
            # fallback to generic text search
            fallback = f'"{name}" AND reviewed:true'
            if organism:
                fallback += f" AND organism_id:{organism}"
            warnings.warn(f"⚠️ Strict search failed for '{name}', retrying generic search…")
            results = run_query(fallback)

        if not results:
            print(f"⚠️  No reviewed UniProt entry for '{name}'"
                  + (f" in {organism}" if organism else ""))
            return None

        hit = results[0]
        acc = hit.get("accession") or hit.get("primaryAccession")
        org = hit.get("organism_name", "")
        print(f"✅ Found {acc} ({org}) for '{name}'")
        return acc

    except requests.HTTPError as e:
        warnings.warn(f"⚠️ HTTP error during UniProt search: {e}")
        return None
    except Exception as e:
        warnings.warn(f"⚠️ Unexpected error during UniProt search: {e}")
=======

def search_uniprot_name(name):
    """
    🔍 Query UniProt for a reviewed accession by protein common name.
    """
    print(f"🔍 Searching UniProt for name '{name}'…")
    url = 'https://rest.uniprot.org/uniprotkb/search'
    params = {
        'query': f'(protein_name:{name}) AND reviewed:true',
        'format': 'json',
        'size': 1,
        'fields': 'primaryAccession'
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    results = r.json().get('results', [])
    if results:
        acc = results[0]['primaryAccession']
        print(f"✅ Found accession {acc} for '{name}'")
        return acc
    else:
        print(f"⚠️  No accession found for '{name}'")
>>>>>>> 0a2edc19e048b73bb4e4255270613a728ee264b6
        return None
