#!/usr/bin/env python3
import requests
import warnings

# Map common organism names to taxonomy IDs
TAXON_MAP = {
    'human':        9606,
    'mouse':        10090,
    'yeast':        559292,
    'ecoli':        562,
    'fly':          7227,
    'arabidopsis':  3702,
}

def search_uniprot_name(name: str, organism=None):
    """
    1) Strict search (reviewed entries only)
    2) Fallback search (all entries)
    Returns the top accession or None.
    """
    base = 'https://rest.uniprot.org/uniprotkb/search'
    core_filters = [f'protein_name:"{name}"']
    if organism:
        core_filters.append(f'taxonomy_id:{organism}')

    for reviewed in (True, False):
        filters = list(core_filters)
        if reviewed:
            filters.append('reviewed:true')
        query = '(' + ' AND '.join(filters) + ')'

        params = {
            'query':  query,
            'format': 'json',
            'size':   1,
            'fields': 'accession,organism_id'
        }
        try:
            print(f"🔍 Searching UniProt ({'reviewed' if reviewed else 'all'}) for '{name}'…")
            resp = requests.get(base, params=params, timeout=10)
            resp.raise_for_status()
            results = resp.json().get('results', [])
            if results:
                hit = results[0]
                # accession alias maps to primaryAccession in JSON
                acc = hit.get('primaryAccession') or hit.get('accession')
                org = hit.get('organism_id')
                print(f"✅ Found {acc} (taxon {org})")
                return acc
        except requests.exceptions.HTTPError as e:
            warnings.warn(f"⚠️ UniProt search failed ({'reviewed' if reviewed else 'all'}): {e}")

    print(f"⚠️ No UniProt accession found for '{name}'.")
    return None


def fetch_uniprot(acc: str):
    """
    Fetch full UniProt entry for accession (no field filtering on server).
    Returns dict with 'seq', 'pe', and 'ptm'.
    """
    url = f'https://rest.uniprot.org/uniprotkb/{acc}.json'
    params = {'format': 'json'}
    try:
        print(f"🔄 Fetching UniProt entry for {acc}…")
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        js = resp.json()

        seq = js['sequence']['value']
        pe  = js['proteinExistence']
        ptm_count = sum(
            1 for f in js.get('features', [])
            if f.get('type','').lower() == 'modified residue'
        )

        print(f"✅ UniProt: len={len(seq)}, pe={pe}, ptm={ptm_count}")
        return {'seq': seq, 'pe': pe, 'ptm': ptm_count}

    except Exception as e:
        warnings.warn(f"⚠️ UniProt fetch failed for {acc}: {e}")
        return None
