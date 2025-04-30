# llm_plm_hybrid/data/taxonomy.py

"""
Mapping from simple organism names to NCBI taxonomy IDs.
Used by `rag_pipeline.py` to resolve name‐based queries.
"""

TAXON_MAP = {
    'human':      9606,
    'mouse':     10090,
    'yeast':    559292,
    'ecoli':      562,
    'fly':        7227,
    'arabidopsis':3702,
}
