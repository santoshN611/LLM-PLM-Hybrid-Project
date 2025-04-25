#!/usr/bin/env python3
import requests
import warnings

# Map common organism names to taxonomy IDs
TAXON_MAP = {
    'human':       9606,
    'mouse':       10090,
    'yeast':       559292,
    'ecoli':       562,
    'fly':         7227,
    'arabidopsis': 3702,
}

def search_uniprot_name(name: str, organism=None):
    """
    Look up a UniProtKB accession by protein name.

    If `organism` is provided, we do two passes:
      1) reviewed entries in that taxon
      2) all entries     in that taxon

    If `organism` is None, we fall back as:
      1) reviewed, human
      2) reviewed, any
      3) all,     human
      4) all,     any
    """
    base = 'https://rest.uniprot.org/uniprotkb/search'
    core = [f'protein_name:"{name}"']

    # Build lookup passes
    if organism is not None:
        passes = [(True, organism), (False, organism)]
    else:
        passes = [
            (True,  TAXON_MAP['human']),
            (True,  None),
            (False, TAXON_MAP['human']),
            (False, None),
        ]

    for reviewed, tax in passes:
        filters = core.copy()
        if reviewed:
            filters.append('reviewed:true')
        if tax is not None:
            filters.append(f'taxonomy_id:{tax}')
        query = '(' + ' AND '.join(filters) + ')'

        # **Do not specify fields** — pull full JSON
        params = {
            'query':  query,
            'format': 'json',
            'size':   1,
        }

        try:
            mode = 'reviewed' if reviewed else 'all'
            scope = f'{mode}, taxon={tax}'
            print(f"🔍 Searching UniProt ({scope}) for '{name}'…")
            resp = requests.get(base, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get('results', [])
            if results:
                hit = results[0]
                # primaryAccession is guaranteed in the full JSON
                acc = hit.get('primaryAccession') or hit.get('uniProtkbId')
                org_id = hit.get('organism', {}).get('taxonId')
                print(f"✅ Found {acc} (taxon {org_id})")
                return acc
        except requests.exceptions.HTTPError as e:
            warnings.warn(f"⚠️ UniProt search failed ({scope}): {e}")

    print(f"⚠️ No UniProt accession found for '{name}'.")
    return None


def fetch_uniprot(acc: str):
    """
    Retrieve the full UniProt JSON entry for `acc`.
    Extract locally: sequence, existence, PTM count, organism.
    """
    url = f'https://rest.uniprot.org/uniprotkb/{acc}.json'
    try:
        print(f"🔄 Fetching UniProt entry for {acc}…")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        js = resp.json()

        seq     = js['sequence']['value']
        pe      = js['proteinExistence']
        ptm_cnt = sum(
            1 for f in js.get('features', [])
            if f.get('type','').lower() == 'modified residue'
        )
        org_id  = js.get('organism', {}).get('taxonId')

        print(f"✅ UniProt: len={len(seq)}, pe={pe}, ptm={ptm_cnt}, org={org_id}")
        return {
            'accession': acc,
            'seq':       seq,
            'pe':        pe,
            'ptm':       ptm_cnt,
            'organism':  org_id
        }

    except Exception as e:
        warnings.warn(f"⚠️ UniProt fetch failed for {acc}: {e}")
        return None
