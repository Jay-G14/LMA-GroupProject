"""Run B0 (zero-shot) or B1 (direct RAG) on a PUBHEALTH split and write one prediction file.

Usage:
  python scripts/run_baselines.py --system b1 --split dev --n 20 --profile smoke_qwen25_05b_cpu --exp-id smoke_b1
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.pubhealth import load_all  # noqa: E402
from oblig.evaluation.classification import INVALID, classification_report  # noqa: E402
from oblig.llm import prompts  # noqa: E402
from oblig.llm.client import LLMClient, LLMConfig  # noqa: E402
from oblig.retrieval.bm25 import BM25  # noqa: E402
from oblig.retrieval.corpus import build_corpus, corpus_fingerprint  # noqa: E402
from oblig.utils.cache import Cache  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["b0", "b1"], required=True)
    ap.add_argument("--split", default="dev", choices=["train", "dev"])  # test only after Gate 3
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="evidence budget (passages)")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    prof = yaml.safe_load(open(ROOT / "configs" / "model.yaml"))[args.profile]
    rcfg = yaml.safe_load(open(ROOT / "configs" / "retrieval.yaml"))["pubhealth"]
    k = args.k or rcfg["evidence_budget_passages"]
    cfg = LLMConfig(**prof)
    llm = LLMClient(cfg, Cache(ROOT / "results" / "cache" / "llm.sqlite"))

    frames = {s: sp.data for s, sp in load_all().items()}
    data = frames[args.split]
    if args.n:
        data = data.sample(args.n, random_state=args.seed)
    bm = fp = None
    if args.system == "b1":
        passages, _ = build_corpus({s: frames[s] for s in rcfg["corpus_splits"]},
                                   rcfg["sentences_per_passage"], rcfg["stride"], rcfg["mask_verdict_sentences"])
        text_of = {p.passage_id: p.text for p in passages}
        bm = BM25(**rcfg["bm25"]).fit([p.text for p in passages], [p.passage_id for p in passages])
        fp = corpus_fingerprint(passages)

    out = ROOT / "results" / "predictions" / f"{args.exp_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    gold, pred = [], []
    with open(out, "w", encoding="utf-8") as f:
        for _, r in data.iterrows():
            retrieved, alias = [], {}
            if args.system == "b0":
                msgs, schema, pv = prompts.b0_messages(r.claim), prompts.B0_SCHEMA, prompts.B0_VERSION
            else:
                retrieved = bm.search(r.claim, k=k)
                ev, alias = prompts.alias_passages([(pid, text_of[pid]) for pid, _ in retrieved])
                msgs, schema, pv = prompts.b1_messages(r.claim, ev), prompts.B1_SCHEMA, prompts.B1_VERSION
            res = llm.generate_json(msgs, schema, pv)
            verdict = res.parsed["verdict"] if res.parsed else INVALID
            cited = res.parsed.get("cited_passages", []) if res.parsed else []
            rec = {"example_id": r.example_id, "gold_label": r.label, "pred_label": verdict,
                   "system": args.system, "retrieved": [{"passage_id": p, "score": round(s, 4)} for p, s in retrieved],
                   "cited_passages": [alias[c] for c in cited if c in alias],
                   "invalid_citations": [c for c in cited if c not in alias],
                   "own_article_retrieved": any(p.split("::")[0] == r.example_id for p, _ in retrieved),
                   "justification": res.parsed.get("justification") if res.parsed else None,
                   "prompt_version": pv, "model": cfg.model, "prompt_tokens": res.prompt_tokens,
                   "completion_tokens": res.completion_tokens, "latency_s": round(res.latency_s, 3),
                   "cached": res.cached, "error": res.error, "raw_output": res.text if res.error else None}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            gold.append(r.label); pred.append(verdict)
    wall = time.perf_counter() - t_start

    metrics = classification_report(gold, pred)
    manifest = {"exp_id": args.exp_id, "system": args.system, "split": args.split, "n": len(gold),
                "sample_seed": args.seed if args.n else None, "profile": args.profile, "llm": prof,
                "retrieval": rcfg if args.system == "b1" else None, "evidence_budget": k if args.system == "b1" else 0,
                "corpus_fingerprint": fp, "git_commit": git_commit(), "python": platform.python_version(),
                "platform": platform.platform(), "run_utc": datetime.now(timezone.utc).isoformat(),
                "wall_clock_s": round(wall, 1), "predictions": str(out.relative_to(ROOT)).replace("\\", "/")}
    (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "metrics" / f"{args.exp_id}.json").write_text(
        json.dumps({"manifest": manifest, "metrics": metrics}, indent=2))
    print(json.dumps({"exp_id": args.exp_id, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                      "invalid": metrics["invalid_predictions"], "wall_s": round(wall, 1)}))


if __name__ == "__main__":
    main()
