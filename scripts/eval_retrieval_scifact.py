"""SciFact abstract-retrieval evaluation against gold evidence abstracts.

Retrieval settings are compared on the leak-free prompt-development split (prompt_dev). The held-out
sets (claims_dev, SciFact-Open test claims) are scored only with --heldout, after the retrieval
configuration is frozen.

Usage:
  python scripts/eval_retrieval_scifact.py --eval-set prompt_dev
  python scripts/eval_retrieval_scifact.py --eval-set dev --heldout --configs hybrid_bm25_bge_base
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.scifact import DEV_SPLITS, HELD_OUT_SETS, load_corpus, load_eval_set  # noqa: E402
from oblig.retrieval.factory import build_retriever  # noqa: E402

KS = (1, 3, 5, 10, 20)


def evaluate(retriever, claims) -> dict:
    with_ev = [c for c in claims if c.evidence]
    rec = {k: [] for k in KS}
    full = {k: [] for k in KS}
    sent_cov = {k: [] for k in KS}
    rr, ndcg, lat = [], [], []
    for c in with_ev:
        t = time.perf_counter()
        ranked = [d for d, _ in retriever.search(c.claim, k=max(KS))]
        lat.append(time.perf_counter() - t)
        gold = set(c.evidence)
        n_sent = sum(len(v) for v in c.evidence.values())
        for k in KS:
            got = gold & set(ranked[:k])
            rec[k].append(len(got) / len(gold))
            full[k].append(got == gold)
            if n_sent:   # SciFact-Open evidence found by pooling has no gold rationale sentences
                sent_cov[k].append(sum(len(c.evidence[d]) for d in got) / n_sent)
        first = next((i + 1 for i, d in enumerate(ranked) if d in gold), None)
        rr.append(1 / first if first else 0.0)
        dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in gold)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gold), 10)))
        ndcg.append(dcg / idcg)
    r = {"n_claims_with_evidence": len(with_ev), "n_claims_with_rationales": len(sent_cov[KS[0]])}
    for k in KS:
        r[f"recall@{k}"] = round(float(np.mean(rec[k])), 4)
        r[f"all_gold_found@{k}"] = round(float(np.mean(full[k])), 4)
        r[f"rationale_sentence_coverage@{k}"] = round(float(np.mean(sent_cov[k])), 4) if sent_cov[k] else None
    r["mrr@20"] = round(float(np.mean(rr)), 4)
    r["ndcg@10"] = round(float(np.mean(ndcg)), 4)
    r["latency_ms_mean"] = round(1000 * float(np.mean(lat)), 2)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", choices=list(DEV_SPLITS + HELD_OUT_SETS), default="prompt_dev")
    ap.add_argument("--heldout", action="store_true", help="required for held-out sets")
    ap.add_argument("--configs", nargs="+",
                    default=["bm25", "bm25_k1.2_b.75", "dense_minilm", "hybrid_bm25_minilm"])
    args = ap.parse_args()
    if args.eval_set in HELD_OUT_SETS and not args.heldout:
        sys.exit(f"{args.eval_set} is a held-out set: pass --heldout only after the retrieval setup is frozen")
    corpus = load_corpus()
    claims = load_eval_set(args.eval_set, corpus.keys())
    results = []
    for name in args.configs:
        t0 = time.perf_counter()
        ret = build_retriever(name, corpus)
        build_s = time.perf_counter() - t0
        r = {"config": name, "index_build_s": round(build_s, 1), **evaluate(ret, claims)}
        results.append(r)
        print(json.dumps(r))
    out = ROOT / "results" / "metrics" / f"retrieval_scifact_{args.eval_set}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"eval_set": args.eval_set, "n_claims": len(claims), "corpus_size": len(corpus),
                               "unit": "abstract (title + sentences)",
                               "run_utc": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
