"""Print ledgers from a prediction file for hand-checking (Gate 7) and qualitative examples: claim, gold label,
obligations with type and final status, the cited sentences (corpus text), and the verdict with its reason.

Usage:
  python scripts/inspect_ledgers.py to_transfer/pilot/predictions/pilot_m1.jsonl --n 8 --seed 1
  python scripts/inspect_ledgers.py to_transfer/dev/predictions/dev_m2.jsonl --only-wrong --gold NEI
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.scifact import load_corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only-wrong", action="store_true")
    ap.add_argument("--gold", default=None, help="only claims with this gold label")
    ap.add_argument("--claims", nargs="*", type=int, default=None)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.predictions, encoding="utf-8")]
    recs = [r for r in recs if "ledger" in r]
    if args.claims:
        recs = [r for r in recs if r["claim_id"] in args.claims]
    if args.only_wrong:
        recs = [r for r in recs if r["pred_label"] != r["gold_label"]]
    if args.gold:
        recs = [r for r in recs if r["gold_label"] == args.gold]
    random.Random(args.seed).shuffle(recs)
    corpus = load_corpus()
    wrap = lambda s, ind: textwrap.fill(s, 110, initial_indent=ind, subsequent_indent=ind)
    for r in recs[: args.n]:
        led = r["ledger"]
        print("=" * 110)
        print(wrap(f"claim {r['claim_id']} [gold {r['gold_label']}] {led['claim']}", ""))
        print(f"verdict {r['pred_label']} ({r.get('verdict_reason')}); M1 verdict {r.get('m1_label', '-')}; "
              f"{r['n_calls']} calls; retrieved {r['retrieved']}")
        final = {}
        for a in led["audits"]:
            if a.get("kind", "audit") == "audit":
                final[a["obligation_id"]] = a
        for o in led["obligations"]:
            a = final.get(o["obligation_id"])
            print(wrap(f"{o['obligation_id']} [{o['criticality']}/{o['type']}] {o['description']}  ->  "
                       f"{a['effective_status'] if a else 'not audited'}"
                       f"{' (raw ' + a['status'] + ')' if a and a['status'] != a['effective_status'] else ''}", "  "))
            if a and a.get("rationale"):
                print(wrap(f"why: {a['rationale']}", "      "))
            for e in (a["evidence"] if a else []):
                for i in e["sentence_ids"]:
                    print(wrap(f"[{e['doc_id']}:{i}] {corpus[e['doc_id']].sentences[i]}", "      "))
        for rr in led.get("retrieval_rounds", []):
            if rr.get("new_doc_ids") or rr.get("queries"):
                print(wrap(f"round {rr['round']}: targets {rr.get('targets')} queries {rr.get('queries')} "
                           f"new {rr.get('new_doc_ids')} consolidations {rr.get('consolidations')}", "  "))
        if led.get("planner_error"):
            print(wrap(f"planner note: {led['planner_error']}", "  "))


if __name__ == "__main__":
    main()
