"""Build SciFact abstract retrievers by config name."""
from __future__ import annotations

import os
from pathlib import Path

from .bm25 import BM25
from .hybrid import HybridRetriever

EMB_CACHE = Path(os.environ.get("OBLIG_EMB_CACHE") or Path(__file__).resolve().parents[3] / "results" / "cache" / "embeddings")
DENSE_MODELS = {"minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
                "bge_base": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ")}
BM25_PARAMS = {"bm25": (0.9, 0.4), "bm25_k1.2_b.75": (1.2, 0.75)}


def abstract_text(a) -> str:
    return a.title + " " + " ".join(a.sentences)


def build_retriever(name: str, corpus: dict):
    ids = list(corpus)
    texts = [abstract_text(corpus[i]) for i in ids]
    if name in BM25_PARAMS:
        return BM25(*BM25_PARAMS[name]).fit(texts, ids)
    if name.startswith("dense_"):
        from .dense import DenseRetriever
        model, prefix = DENSE_MODELS[name.removeprefix("dense_")]
        return DenseRetriever(model, EMB_CACHE, query_prefix=prefix, max_seq_length=512,
                              device=os.environ.get("OBLIG_DENSE_DEVICE", "cpu")).fit(texts, ids)
    if name.startswith("hybrid_bm25_"):
        return HybridRetriever([build_retriever("bm25", corpus),
                                build_retriever("dense_" + name.removeprefix("hybrid_bm25_"), corpus)])
    raise ValueError(f"unknown retriever {name!r}")
