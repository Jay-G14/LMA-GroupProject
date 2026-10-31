"""Re-score stored ledger runs under another verdict rule, without new LLM calls (protocol amendment 5).

Each re-scored run is written next to its source as <prefix><system>_<tag>.jsonl, with the rule in its config, so
compare_systems.py treats it as a variant of the system. The frozen rule must first reproduce every stored verdict.
M2 records also get their first-pass (round-0) verdict re-scored, so the loop gain compares like with like; the
M2 loop itself is not re-run (an obligation whose contradiction no longer counts is not re-audited).

Also writes the ledger-as-check fallback B1-checked: B1's verdict where M1 gives the same verdict, NEI otherwise,
with M1 under the frozen rule (b1_checked) and under the anchored rule (b1_checked_anch). It costs both runs.
Amendment 6 adds the same check by systems without a ledger, B2 (b1_checked_b2) and B1-9 (b1_checked_b1_9), to
test whether the ledger is a better check than any second opinion (scripts/veto_analysis.py).

  python scripts/rescore_ledgers.py --results-dir to_transfer/dev --prefix dev_
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.ledger.aggregation import POLICIES, POLICY_TAGS  # noqa: E402
from oblig.ledger.schema import Ledger  # noqa: E402

AMENDED = "scifact_v1_anchored"
LEDGER_RUNS = ["m1", "m1_shared", "m1_aligned", "m1_untyped", "m2", "m2_random", "m2_noconsol", "m2_claimfollow"]
RULE_ABLATION_RUNS = ["m1"]        # the three rules not chosen on prompt_dev are applied to M1's ledgers only
# Checks on B1: checking run -> (output suffix, rule of that run). m1_anch is written above.
CHECKERS = {"m1": ("", "scifact_v1"), f"m1_{POLICY_TAGS[AMENDED]}": (f"_{POLICY_TAGS[AMENDED]}", AMENDED),
            "b2": ("_b2", "scifact_v1"), "b1_9": ("_b1_9", "scifact_v1")}


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write(path: Path, cfg: dict, recs: list[dict]) -> None:
    path.with_suffix(".config.json").write_text(json.dumps(cfg, indent=2))
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)


def first_pass(ledger: dict) -> Ledger:
    """The M1 ledger an M2 run started from: its round-0 status audits only."""
    d = copy.deepcopy(ledger)
    d["audits"] = [a for a in d["audits"] if a.get("kind", "audit") == "audit" and a["round"] == 0]
    d["retrieval_rounds"] = []
    return Ledger.from_dict(d)


def rescore(rec: dict, policy: str) -> dict:
    led, frozen = Ledger.from_dict(rec["ledger"]), POLICIES["scifact_v1"]
    if frozen(led).label != rec["pred_label"]:
        sys.exit(f"claim {rec['claim_id']}: the frozen rule does not reproduce the stored verdict")
    v = POLICIES[policy](led)
    out = {**rec, "pred_label": v.label, "verdict_reason": v.reason, "rationale": v.rationale_docs,
           "frozen_label": rec["pred_label"]}
    if "m1_label" in rec:
        led0 = first_pass(rec["ledger"])
        if frozen(led0).label != rec["m1_label"]:
            sys.exit(f"claim {rec['claim_id']}: the frozen rule does not reproduce the stored first-pass verdict")
        v0 = POLICIES[policy](led0)
        out.update({"m1_label": v0.label, "m1_rationale": v0.rationale_docs, "verdict_changed": v.label != v0.label})
    return out


def checked(b1: list[dict], checker: dict[int, dict], name: str) -> list[dict]:
    out = []
    for r in b1:
        m = checker[r["claim_id"]]
        keep = r["pred_label"] == m["pred_label"]
        out.append({"claim_id": r["claim_id"], "gold_label": r["gold_label"], "system": "b1_checked",
                    "model": r["model"], "pred_label": r["pred_label"] if keep else "NEI",
                    "verdict_reason": f"B1 and {name} agree" if keep else f"{name} gives {m['pred_label']}",
                    "rationale": r["rationale"] if keep else {}, "retrieved": r["retrieved"],
                    "b1_label": r["pred_label"], "check_label": m["pred_label"], "calls": r["calls"] + m["calls"],
                    "n_calls": len(r["calls"]) + len(m["calls"]),
                    "prompt_tokens": r.get("prompt_tokens", 0) + m.get("prompt_tokens", 0),
                    "completion_tokens": r.get("completion_tokens", 0) + m.get("completion_tokens", 0)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()
    pred = Path(args.results_dir) / "predictions"

    def run(name: str) -> tuple[dict, list[dict]]:
        src = pred / f"{args.prefix}{name}.jsonl"
        return json.loads(src.with_suffix(".config.json").read_text()), read(src)

    for name in LEDGER_RUNS:
        cfg, recs = run(name)
        for policy in [AMENDED] + ([p for p in POLICY_TAGS if p != AMENDED] if name in RULE_ABLATION_RUNS else []):
            new = [rescore(r, policy) for r in recs]
            write(pred / f"{args.prefix}{name}_{POLICY_TAGS[policy]}.jsonl",
                  {**cfg, "policy": policy, "rescored_from": f"{args.prefix}{name}"}, new)
            moves = Counter((r["gold_label"], r["frozen_label"], r["pred_label"]) for r in new
                            if r["pred_label"] != r["frozen_label"])
            fixed = sum(n for (g, _, p), n in moves.items() if g == p)
            broken = sum(n for (g, o, _), n in moves.items() if g == o)
            print(f"{args.prefix}{name} under {policy}: {sum(moves.values())} verdicts changed, "
                  f"{fixed} fixed, {broken} broken")

    b1_cfg, b1 = run("b1")
    for checker, (suffix, policy) in CHECKERS.items():
        _, recs = run(checker)
        recs = checked(b1, {r["claim_id"]: r for r in recs}, checker)
        write(pred / f"{args.prefix}b1_checked{suffix}.jsonl",
              {**b1_cfg, "system": "b1_checked", "policy": policy, "checker": checker,
               "checked_by": f"{args.prefix}{checker}"}, recs)
        agree = sum(r["b1_label"] == r["check_label"] for r in recs)
        print(f"{args.prefix}b1_checked{suffix}: B1 and {checker} agree on {agree} of {len(recs)} claims")


if __name__ == "__main__":
    main()
