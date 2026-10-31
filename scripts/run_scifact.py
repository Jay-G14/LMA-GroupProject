"""Run a SciFact system and write one prediction file, metrics and a run manifest.

Resumable: claims already in the prediction file are skipped. The held-out sets (claims_dev and the
SciFact-Open test claims) require --heldout, which is only to be used after Gate 3. Development runs use
the leak-free prompt_dev split (protocol amendment 2). Claims run in parallel worker threads (--workers);
each claim's own calls stay sequential, and the server batches concurrent calls.

Usage:
  python scripts/run_scifact.py --system m1 --eval-set prompt_dev --n 20 --profile mock_scifact --exp-id smoke_sf_m1
  python scripts/run_scifact.py --system m1 --eval-set dev --heldout --profile qwen3_8b_vllm --workers 48 \\
      --base-url http://localhost:8000/v1 --results-dir to_transfer/dev --exp-id dev_m1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.scifact import DEV_SPLITS, HELD_OUT_SETS, load_corpus, load_eval_set  # noqa: E402
from oblig.evaluation.scifact import evaluate  # noqa: E402
from oblig.llm import prompts_scifact as P  # noqa: E402
from oblig.llm.client import LLMClient, LLMConfig  # noqa: E402
from oblig.llm.mock_scifact import mock_scifact  # noqa: E402
from oblig.retrieval.factory import build_retriever  # noqa: E402
from oblig.systems.scifact import SYSTEMS, Context  # noqa: E402
from oblig.utils.cache import Cache  # noqa: E402


def git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    commit = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain", "--", "src", "scripts", "configs")
    return {"git_commit": commit or "unknown", "git_dirty": bool(dirty) if dirty is not None else None}


def code_sha256() -> str:
    """Hash of every file under src/, scripts/ and configs/: identifies the exact code even if uncommitted."""
    h = hashlib.sha256()
    for path in sorted(p for d in ("src", "scripts", "configs") for p in (ROOT / d).rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--eval-set", choices=list(DEV_SPLITS + HELD_OUT_SETS), default="prompt_dev")
    ap.add_argument("--heldout", action="store_true", help="required for held-out sets (after Gate 3 only)")
    ap.add_argument("--n", type=int, default=None, help="first n claims by id (development pilots)")
    ap.add_argument("--retriever", default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--base-url", default=None, help="override the profile's server(s), comma-separated")
    ap.add_argument("--workers", type=int, default=1, help="claims processed in parallel")
    ap.add_argument("--sc-samples", type=int, default=None, help="B1-SC samples (token-matched to M1, Gate 3)")
    ap.add_argument("--m2-followup", choices=["fused", "claim"], default=None)
    ap.add_argument("--m2-consolidate", choices=["on", "off"], default=None, help="ablation of amendment 1")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--cache", default=None, help="LLM response cache (default: <results-dir>/cache/llm_<eval-set>.sqlite)")
    ap.add_argument("--exp-id", required=True)
    args = ap.parse_args()
    if args.eval_set in HELD_OUT_SETS and not args.heldout:
        sys.exit(f"{args.eval_set} is a held-out set: pass --heldout only after the protocol is frozen (Gate 3)")

    rcfg = yaml.safe_load(open(ROOT / "configs" / "retrieval.yaml", encoding="utf-8"))["scifact"]
    prof = yaml.safe_load(open(ROOT / "configs" / "model.yaml", encoding="utf-8"))[args.profile]
    cfg = LLMConfig(**prof)
    if args.base_url:
        cfg.base_url = args.base_url
    if args.profile == "mock_scifact":
        cfg.extra["mock_fn"] = mock_scifact
    retriever_name = args.retriever or rcfg.get("retriever", "hybrid_bm25_minilm")
    budget = args.budget or rcfg["evidence_budget_abstracts"]
    results = Path(args.results_dir)
    corpus = load_corpus()
    ctx = Context(LLMClient(cfg, Cache(Path(args.cache) if args.cache else results / "cache" / f"llm_{args.eval_set}.sqlite")),
                  build_retriever(retriever_name, corpus), corpus, budget=budget)
    if args.sc_samples:
        ctx.sc_samples = args.sc_samples
    if args.m2_followup:
        ctx.m2_followup = args.m2_followup
    if args.m2_consolidate:
        ctx.m2_consolidate = args.m2_consolidate == "on"
    claims = load_eval_set(args.eval_set, corpus.keys())
    if args.n:
        claims = claims[: args.n]

    out = results / "predictions" / f"{args.exp_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Resuming is only allowed with an identical configuration (no stale outputs).
    run_cfg = {"system": args.system, "eval_set": args.eval_set, "profile": prof, "retriever": retriever_name,
               "budget": budget, "policy": ctx.policy, "m2_rounds": ctx.m2_rounds,
               "m2_new_per_round": ctx.m2_new_per_round, "sc_samples": ctx.sc_samples,
               "sc_temperature": ctx.sc_temperature, "seed": ctx.seed, "m2_consolidate": ctx.m2_consolidate,
               "m2_followup": ctx.m2_followup, "rrf_k": ctx.rrf_k,
               "prompt_versions": {k: v for k, v in vars(P).items() if k.endswith("_VERSION")}}
    cfg_path = out.with_suffix(".config.json")
    if out.exists() and cfg_path.exists():
        # Self-consistency settings are read by B1-SC only, so they may differ when another system resumes.
        unread = set() if args.system == "b1_sc" else {"sc_samples", "sc_temperature"}
        old = json.loads(cfg_path.read_text())
        changed = [k for k in sorted(set(old) | set(run_cfg)) if k not in unread and old.get(k) != run_cfg.get(k)]
        if changed:
            sys.exit(f"{out.name} was produced with a different configuration ({', '.join(changed)}); use a new --exp-id")
    cfg_path.write_text(json.dumps(run_cfg, indent=2))
    done = {}
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            done[r["claim_id"]] = r
    todo = [c for c in claims if c.id not in done]
    lock, t0 = threading.Lock(), time.perf_counter()

    def run_one(c):
        t = time.perf_counter()
        rec = SYSTEMS[args.system](ctx, c)
        # A dead or failing server is not a model answer: abort instead of recording the claim as invalid,
        # so a rerun resumes it. Parse and schema errors (bad model output) are still scored as invalid.
        down = [x["error"] for x in rec["calls"] if x["error"] and x["error"].startswith("call_error")
                and not any(f"server returned {code}" in x["error"] or f"Client error '{code}" in x["error"]
                            for code in (400, 413, 422))]
        if down:
            raise RuntimeError(f"claim {c.id}: server call failed ({down[0][:200]}); aborting without writing it")
        return {"claim_id": c.id, "gold_label": c.label, "system": args.system, "model": cfg.model,
                **rec, "n_calls": len(rec["calls"]),
                "prompt_tokens": sum(x["prompt_tokens"] for x in rec["calls"]),
                "completion_tokens": sum(x["completion_tokens"] for x in rec["calls"]),
                "n_call_errors": sum(x["error"] is not None for x in rec["calls"]),
                "wall_s": round(time.perf_counter() - t, 3)}

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, c) for c in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except RuntimeError as e:
                for other in futures:
                    other.cancel()
                sys.exit(f"ERROR: {e}")
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[rec["claim_id"]] = rec
            if args.workers > 1 and (i % 25 == 0 or i == len(todo)):
                print(f"  {args.exp_id}: {i}/{len(todo)} claims, {time.perf_counter() - t0:.0f}s", flush=True)
    preds = {c.id: done[c.id] for c in claims}
    metrics = evaluate(claims, preds)
    recs = list(preds.values())
    cost = {"calls_per_claim": round(sum(r["n_calls"] for r in recs) / len(recs), 3),
            "prompt_tokens_per_claim": round(sum(r["prompt_tokens"] for r in recs) / len(recs), 1),
            "completion_tokens_per_claim": round(sum(r["completion_tokens"] for r in recs) / len(recs), 1),
            "call_errors": sum(r["n_call_errors"] for r in recs),
            "invalid_citations": sum(len(r.get("invalid_citations", [])) for r in recs),
            "wall_s_per_claim_mean": round(sum(r["wall_s"] for r in recs) / len(recs), 2)}
    manifest = {"exp_id": args.exp_id, "dataset": claims[0].dataset if claims else "scifact", "system": args.system,
                "eval_set": args.eval_set, "n": len(claims), "profile": args.profile,
                "llm": {k: v for k, v in prof.items()}, "base_url": cfg.base_url,
                "retriever": retriever_name, "evidence_budget_abstracts": budget, "policy": ctx.policy,
                "system_params": {k: run_cfg[k] for k in ("m2_rounds", "m2_new_per_round", "sc_samples", "sc_temperature",
                                                          "seed", "m2_consolidate", "m2_followup", "rrf_k")},
                **git_state(), "code_sha256": code_sha256(), "workers": args.workers,
                "python": platform.python_version(), "platform": platform.platform(),
                "run_utc": datetime.now(timezone.utc).isoformat(), "wall_clock_s": round(time.perf_counter() - t0, 1),
                "predictions": str(out)}
    (results / "metrics").mkdir(parents=True, exist_ok=True)
    (results / "metrics" / f"{args.exp_id}.json").write_text(
        json.dumps({"manifest": manifest, "metrics": metrics, "cost": cost}, indent=2))
    cl, ab = metrics["claim_level"], metrics["abstract_level"]
    print(json.dumps({"exp_id": args.exp_id, "macro_f1": cl["macro_f1"], "ci95": cl["macro_f1_ci95"],
                      "invalid": cl["invalid_predictions"],
                      "unsupported_verdict_rate": metrics["safety"]["unsupported_verdict_rate"],
                      "abstract_label_rationale_f1": (ab["label_rationale"] or ab["label_only"])["f1"],
                      "sentence_selection_label_f1": metrics["sentence_level"]["selection_label"]["f1"], **cost}),
          flush=True)


if __name__ == "__main__":
    main()
