"""BM25 retrieval pilot on PUBHEALTH dev: own-article recall across passage granularities.

The pooled corpus contains every split's (verdict-masked) main_text. The "own
article" of a claim is the main_text of its record: a document-level *silver*
evidence target, not a semantic gold-evidence judgement.

Usage: python scripts/eval_retrieval.py [--n 1214]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.pubhealth import SPLITS, load_all  # noqa: E402
from oblig.retrieval.bm25 import BM25  # noqa: E402
from oblig.retrieval.corpus import build_corpus  # noqa: E402

KS = (1, 5, 10, 20)


def evaluate(passages, claims, granularity: str) -> dict:
    t0 = time.perf_counter()
    bm = BM25().fit([p.text for p in passages], [p.passage_id for p in passages])
    build_s = time.perf_counter() - t0
    doc_of = {p.passage_id: p.doc_id for p in passages}
    hits = {k: [] for k in KS}
    rr, lat, dup = [], [], []
    for ex_id, claim in claims:
        t = time.perf_counter()
        res = bm.search(claim, k=max(KS))
        lat.append(time.perf_counter() - t)
        docs = [doc_of[pid] for pid, _ in res]
        rank = next((i + 1 for i, d in enumerate(docs) if d == ex_id), None)
        rr.append(1 / rank if rank else 0.0)
        for k in KS:
            hits[k].append(rank is not None and rank <= k)
        top10 = docs[:10]
        dup.append(1 - len(set(top10)) / max(len(top10), 1))
    return {"granularity": granularity, "n_passages": len(passages), "index_build_s": round(build_s, 1),
            **{f"own_article_recall@{k}": round(float(np.mean(v)), 4) for k, v in hits.items()},
            "own_article_mrr@20": round(float(np.mean(rr)), 4),
            "same_doc_redundancy@10": round(float(np.mean(dup)), 4),
            "latency_ms_mean": round(1000 * float(np.mean(lat)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="number of dev claims (default all)")
    args = ap.parse_args()
    frames = {s: sp.data for s, sp in load_all().items()}
    dev = frames["dev"] if args.n is None else frames["dev"].sample(args.n, random_state=13)
    claims = list(zip(dev["example_id"], dev["claim"]))
    results = []
    for name, spp in [("2-sentence", 2), ("4-sentence", 4), ("8-sentence", 8), ("whole-doc", 10_000)]:
        passages, stats = build_corpus(frames, sents_per_passage=spp)
        r = evaluate(passages, claims, name)
        if name == "4-sentence":
            r["docs_with_masked_sentences"] = round(float((stats.n_masked > 0).mean()), 4)
            r["masked_sentences_total"] = int(stats.n_masked.sum())
        results.append(r)
        print(json.dumps(r))
    out = ROOT / "results" / "metrics" / "retrieval_bm25_dev.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": "dev", "n_claims": len(claims), "corpus_splits": list(SPLITS),
                               "masking": True, "results": results}, indent=2))


if __name__ == "__main__":
    main()
