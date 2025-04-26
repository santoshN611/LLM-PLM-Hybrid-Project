# Makefile for LLM-PLM-Hybrid-Project

.PHONY: all preprocess split embed train index build-test evaluate

all: preprocess split embed train index build-test evaluate

preprocess:
	@echo "🔄 [1/7] Creating raw UniProt corpus…"
	python -m llm_plm_hybrid.data.create_corpus

split:
	@echo "🔄 [2/7] Splitting into train/val/test…"
	python -m llm_plm_hybrid.data.split_corpus

embed:
	@echo "🔄 [3/7] Generating ESM-2 embeddings…"
	python -m llm_plm_hybrid.embeddings.generate_embeddings

train:
	@echo "🔄 [4/7] Training tiny-heads…"
	python -m llm_plm_hybrid.embeddings.train_heads

index:
	@echo "🔄 [5/7] Building FAISS index…"
	python -m llm_plm_hybrid.retrieval.faiss_index

build-test:
	@echo "🔄 [6/7] Building test QA corpus…"
	python -m llm_plm_hybrid.evaluation.build_test_qa_corpus

evaluate:
	@echo "🔄 [7/7] Evaluating end-to-end pipeline…"
	python -m llm_plm_hybrid.evaluation.evaluate_pipeline
