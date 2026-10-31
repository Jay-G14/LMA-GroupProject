"""Okapi BM25 over a sparse term matrix (no external BM25 dependency)."""
from __future__ import annotations

import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class BM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b

    def fit(self, texts: list[str], ids: list[str]) -> "BM25":
        self.ids = np.asarray(ids)
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vec.fit_transform(texts).tocsc().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.diff(tf.indptr)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        # Precomputed per-(doc, term) BM25 weight; a query score is a column sum.
        self.W = sparse.csc_matrix((w * self.idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vec.vocabulary_[t] for t in tokenize(query) if t in self.vec.vocabulary_]
        if not cols:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        return np.asarray(self.W[:, cols].sum(axis=1)).ravel()

    def search(self, query: str, k: int = 10, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[: k + (len(exclude) if exclude else 0) + 50]
        out = []
        for i in top:
            if exclude and self.ids[i] in exclude:
                continue
            if s[i] <= 0:
                break
            out.append((str(self.ids[i]), float(s[i])))
            if len(out) == k:
                break
        return out
