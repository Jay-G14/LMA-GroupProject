"""Dense retrieval with sentence-transformers and an on-disk embedding cache."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np


class DenseRetriever:
    def __init__(self, model_name: str, cache_dir: Path, query_prefix: str = "", batch_size: int = 64,
                 max_seq_length: int | None = None, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model_name, self.query_prefix, self.batch_size = model_name, query_prefix, batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.lock = threading.Lock()               # one encoder shared by the runner's worker threads
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.cache_dir = cache_dir

    def _encode(self, texts: list[str]) -> np.ndarray:
        with self.lock:
            return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def fit(self, texts: list[str], ids: list[str]) -> "DenseRetriever":
        self.ids = np.asarray(ids)
        h = hashlib.sha256()
        for i, t in zip(ids, texts):
            h.update(i.encode()); h.update(t.encode())
        key = f"{self.model_name.replace('/', '__')}_{self.model.max_seq_length}_{h.hexdigest()[:16]}.npy"
        path = self.cache_dir / key
        if path.exists():
            self.E = np.load(path)
        else:
            self.E = self._encode(texts)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.E)
        return self

    def scores(self, query: str) -> np.ndarray:
        return self.E @ self._encode([self.query_prefix + query])[0]

    def search(self, query: str, k: int = 10, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        s = self.scores(query)
        out = []
        for i in np.argsort(-s, kind="stable"):
            if exclude and self.ids[i] in exclude:
                continue
            out.append((str(self.ids[i]), float(s[i])))
            if len(out) == k:
                break
        return out
