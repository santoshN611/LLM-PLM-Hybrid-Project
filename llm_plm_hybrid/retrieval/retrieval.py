import requests, warnings

TAXON_MAP = {
    'human': 9606, 'mouse':10090, 'yeast':559292,
    'ecoli':562,   'fly':7227,     'arabidopsis':3702,
}

def search_uniprot_name(name: str, organism=None):
    base = 'https://rest.uniprot.org/uniprotkb/search'
    core = f'protein_name:"{name}"'
    passes = []
    if organism:
        passes = [(True,organism),(False,organism)]
    else:
        passes = [(True,9606),(True,None),(False,9606),(False,None)]

    for reviewed,tax in passes:
        flt = [core]
        if reviewed: flt.append('reviewed:true')
        if tax:      flt.append(f'organism_id:{tax}')
        q = "(" + " AND ".join(flt) + ")"
        params = {'query':q,'format':'json','size':1,'fields':'accession,organism_id'}
        try:
            resp = requests.get(base,params=params,timeout=10); resp.raise_for_status()
            res = resp.json().get('results',[])
            if res:
                return res[0].get('primaryAccession') or res[0]['accession']
        except Exception as e:
            warnings.warn(f"⚠️ UniProt search failed: {e}")
    return None

def fetch_uniprot(acc: str):
    url = f'https://rest.uniprot.org/uniprotkb/{acc}.json'
    try:
        r = requests.get(url,timeout=15); r.raise_for_status()
        d = r.json()
        seq  = d['sequence']['value']
        pe   = d['proteinExistence']  # e.g. "1: Evidence at protein level"
        ptm  = sum(1 for f in d.get('features',[])
                   if f.get('type','').lower()=='modified residue')
        orgn = d.get('organism',{}).get('scientificName') \
            or d.get('organism',{}).get('commonName','')
        return {'seq':seq,'pe':pe,'ptm':ptm,'organism_name':orgn,'accession':acc}
    except Exception as e:
        warnings.warn(f"⚠️ UniProt fetch failed for {acc}: {e}")
        return None
