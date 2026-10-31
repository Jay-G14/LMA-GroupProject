"""Deterministic passage corpus over PUBHEALTH main_text with provenance."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ..data.masking import mask_sentences, split_sentences


@dataclass(frozen=True)
class Passage:
    passage_id: str          # "{doc_id}::p{k}"
    doc_id: str              # example_id of the article's own record
    split: str               # split of the record the article belongs to
    sentence_ids: list[int]  # indices into the doc's *masked* sentence list
    text: str


def chunk_document(doc_id: str, split: str, main_text: str, sents_per_passage: int = 4,
                   stride: int | None = None, mask: bool = True) -> tuple[list[Passage], dict]:
    sents = split_sentences(main_text)
    removed: list[int] = []
    if mask:
        sents, removed = mask_sentences(sents)
    stride = stride or sents_per_passage
    out = []
    for k, start in enumerate(range(0, max(len(sents), 1), stride)):
        ids = list(range(start, min(start + sents_per_passage, len(sents))))
        if not ids:
            break
        out.append(Passage(f"{doc_id}::p{k}", doc_id, split, ids, " ".join(sents[i] for i in ids)))
        if start + sents_per_passage >= len(sents):
            break
    return out, {"doc_id": doc_id, "n_sentences": len(sents), "n_masked": len(removed)}


def build_corpus(frames: dict[str, pd.DataFrame], sents_per_passage: int = 4,
                 stride: int | None = None, mask: bool = True) -> tuple[list[Passage], pd.DataFrame]:
    passages, stats = [], []
    for split, df in frames.items():
        for doc_id, text in zip(df["example_id"], df["main_text"]):
            ps, st = chunk_document(doc_id, split, text, sents_per_passage, stride, mask)
            passages.extend(ps)
            stats.append(st)
    return passages, pd.DataFrame(stats)


def corpus_fingerprint(passages: list[Passage]) -> str:
    h = hashlib.sha256()
    for p in passages:
        h.update(p.passage_id.encode())
        h.update(p.text.encode())
    return h.hexdigest()[:16]


def save_corpus(passages: list[Passage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def load_corpus(path: Path) -> list[Passage]:
    with open(path, encoding="utf-8") as f:
        return [Passage(**json.loads(l)) for l in f]
