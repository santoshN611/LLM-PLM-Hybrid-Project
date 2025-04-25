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
    Query UniProt for a UniProtKB accession by protein common name.
    If `organism` is provided, search that taxon first, then all organisms.
    Otherwise default to human then all organisms.
    """
    base = 'https://rest.uniprot.org/uniprotkb/search'
    q_base = f'protein_name:"{name}"'

    passes = [organism] if organism else [TAXON_MAP['human'], None]
    for org in passes:
        scope = f"taxon={org}" if org else "all organisms"
        q = q_base + (f' AND organism_id:{org}' if org else '')
        params = {
            'query':  q,
            'format': 'json',
            'size':   1,
            'fields': 'accession,organism_id'
        }
        try:
            print(f"🔍 Searching UniProt ({scope}) for '{name}'…")
            r = requests.get(base, params=params, timeout=10)
            r.raise_for_status()
            results = r.json().get('results', [])
            if results:
                hit = results[0]
                acc = hit.get('accession')
                org_id = hit.get('organism_id')
                print(f"✅ Found {acc} (taxon {org_id})")
                return acc
        except requests.exceptions.HTTPError as e:
            warnings.warn(f"⚠️ UniProt search failed ({scope}): {e}")

    print(f"⚠️ No UniProt accession found for '{name}'.")
    return None


def fetch_uniprot(acc: str):
    """
    Fetch entire UniProt JSON entry and extract:
      - accession
      - seq: sequence string
      - pe: proteinExistence (string)
      - ptm: count of modified residues
      - organism_id: taxonomy ID
    """
    url = f'https://rest.uniprot.org/uniprotkb/{acc}.json'
    try:
        print(f"🔄 Fetching UniProt entry for {acc}…")
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        js = r.json()

        seq = js.get('sequence', {}).get('value')
        pe  = js.get('proteinExistence', "")
        ptm = sum(
            1 for f in js.get('features', [])
            if f.get('type', '').lower() == 'modified residue'
        )
        org = js.get('organism', {}).get('taxonId')

        print(f"✅ UniProt: len={len(seq) if seq else 'NA'}, pe={pe}, ptm={ptm}, taxon={org}")
        return {
            'accession':    acc,
            'seq':          seq,
            'pe':           pe,
            'ptm':          ptm,
            'organism_id':  org
        }
    except Exception as e:
        warnings.warn(f"⚠️ UniProt fetch failed for {acc}: {e}")
        return None
