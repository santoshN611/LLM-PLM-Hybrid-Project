# LLM-PLM-Hybrid-Project

Run the following commands in order:
* `create_corpus.py`: Requests UniProt REST API for data. Keep in mind the number of pages you are paging for (default=50)
* `split_corpus.py`: Creates train/val/test splits
* `generate_embeddings.py`: Generate and save ESM2 embeddings
* `faiss_index.py`: create FAISS index
* `train_heads.py`: train Classification and Regression heads
* `rag_pipeline.py`: the pipeline, including model loading and question answering

<!-- # DO NOT ADD *.CSV, *.NPZ, OR *.INDEX TO GIT COMMIT -->
