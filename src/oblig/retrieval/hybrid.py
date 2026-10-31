"""Reciprocal-rank fusion of several retrievers."""
from __future__ import annotations


def rrf(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists; ties broken by id for determinism."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


class HybridRetriever:
    def __init__(self, retrievers: list, depth: int = 100, rrf_k: int = 60):
        self.retrievers, self.depth, self.rrf_k = retrievers, depth, rrf_k

    def search(self, query: str, k: int = 10, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        runs = [r.search(query, k=self.depth, exclude=exclude) for r in self.retrievers]
        return rrf(runs, self.rrf_k)[:k]
