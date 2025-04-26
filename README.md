# LLM-PLM-Hybrid-Project

A hybrid pipeline combining large language models (LLMs) and pretrained language models (PLMs) for protein existence classification and post-translational modification (PTM) site counting. This repository automates data collection, embedding generation, model training, retrieval indexing, and end-to-end evaluation.

This project was done for CSE 7850 at Georgia Institute of Technology in Spring 2025.

Authors:
* Santosh Nachimuthu
* Saiyash Vishnubhatt

## 📁 Directory Structure

```bash
LLM-PLM-Hybrid-Project/
├── Makefile                   # Orchestrates the full pipeline
├── requirements.txt           # Python dependencies
├── tiny_heads/                # Pretrained tiny-head weights
│   ├── pe.pt
│   └── ptm.pt
└── llm_plm_hybrid/            # Main package
    ├── __init__.py
    ├── data/                  # Data ingestion & splitting
    │   ├── __init__.py
    │   ├── create_corpus.py   # Download & build raw UniProt corpus
    │   └── split_corpus.py    # Split into train/val/test
    ├── embeddings/            # Embedding generation & head training
    │   ├── __init__.py
    │   ├── generate_embeddings.py  # ESM-2 embedding pipeline
    │   ├── train_heads.py          # Train tiny-head PE & PTM models
    │   └── tiny_heads.py           # Load trained tiny-head weights
    ├── retrieval/             # FAISS index and retrieval utilities
    │   ├── __init__.py
    │   ├── faiss_index.py     # Build FAISS index from embeddings
    │   ├── retrieval.py       # (Optional) retrieval scripts
    │   └── retrieval_utils.py # Context-building & neighbor search
    ├── qa/                    # Retrieval-augmented generation (RAG)
    │   ├── __init__.py
    │   ├── adapters.py        # BioBERT adapter modules
    │   └── rag_pipeline.py    # End-to-end RAG QA pipeline
    ├── evaluation/            # Evaluation scripts and tests
    │   ├── __init__.py
    │   ├── build_test_qa_corpus.py  # Build JSONL test QA set
    │   ├── evaluate_pipeline.py    # Compute metrics + save results
    │   ├── test_intent.py           # Unit test for intent parsing
    │   └── test_neighbors.py        # QA neighbor lookup examples
    └── utils/                # Shared utilities
        ├── __init__.py
        └── embedding_utils.py    # Helper functions for embeddings
```

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone <repo-url> LLM-PLM-Hybrid-Project
   cd LLM-PLM-Hybrid-Project
   ```

2. **Create a Python environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **System prerequisites**:

   - Python 3.8+
   - CUDA toolkit (if you plan to use GPU for ESM-2 and BioBERT)
   - FAISS (if not installed via pip, see [FAISS installation docs](https://github.com/facebookresearch/faiss))

## 🎯 Pipeline Usage

The `Makefile` wraps all steps; to run end-to-end:

```bash
make all
```

Or execute each stage individually:

1. **Preprocess** (download & build raw corpus)
   ```bash
   make preprocess
   ```

2. **Split** (train/validation/test)
   ```bash
   make split
   ```

3. **Embed** (generate ESM-2 embeddings)
   ```bash
   make embed
   ```

4. **Train** (train tiny-head PE & PTM classifiers)
   ```bash
   make train
   ```

5. **Index** (build FAISS index on training embeddings)
   ```bash
   make index
   ```

6. **Build Test Corpus** (create QA JSONL)
   ```bash
   make build-test
   ```

7. **Evaluate** (run end-to-end QA & retrieval evaluation)
   ```bash
   make evaluate
   ```

## 📝 Outputs

- **Data files** (`llm_plm_hybrid/data/`): `classification.csv`, `regression.csv`, splits (`_train`, `_val`, `_test`).
- **Embeddings** (`llm_plm_hybrid/embeddings/`): `.npz` files (`X`, `y`, `meta`), `.index` and `.meta.npy`.
- **Models** (`tiny_heads/`): `pe.pt` & `ptm.pt` PyTorch weights.
- **Test corpus** (`llm_plm_hybrid/evaluation/test_protein_qa.jsonl`): JSONL of QA pairs.
- **Evaluation results** (`eval_results.csv`): per-question metrics & retrieval ranks.

## 🧰 Tips & Troubleshooting

- Run commands from the project root (where the `Makefile` is located).
- If GPU OOM occurs during embedding, the script will automatically retry on CPU.
- To regenerate only the FAISS index after new embeddings:
  ```bash
  make index
  ```
- To manually inspect neighbor retrieval:
  ```bash
  python -m llm_plm_hybrid.evaluation.test_neighbors
  ```

---

