"""Error analysis for the reports (descriptive; nothing here selects or tunes a system).

  python scripts/error_analysis.py to_transfer/dev dev
prints, for every run <tag>_<system>.jsonl in <results>/predictions:
  confusion matrices (gold -> predicted), the verdict reasons of ledger systems' errors, which obligation types
  the wrong REFUTES verdicts rest on, B1-vs-M1 discordant claims, M2's verdict changes against its own first pass,
  and how often M2-random's targets differ from M2's.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

LABELS = ("SUPPORTS", "REFUTES", "NEI")


def load(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["claim_id"]: r for r in map(json.loads, f)}


def final_status(led: dict) -> dict[str, str]:
    st = {}
    for a in led["audits"]:
        if a.get("kind", "audit") == "audit":
            st[a["obligation_id"]] = a["effective_status"]
    return st


def main() -> None:
    results, tag = Path(sys.argv[1]), sys.argv[2]
    runs = {p.stem[len(tag) + 1:]: load(p) for p in sorted((results / "predictions").glob(f"{tag}_*.jsonl"))}
    n_full = max(len(r) for r in runs.values())
    runs = {k: v for k, v in runs.items() if len(v) == n_full}          # finished runs only
    print(f"# {tag}: {len(runs)} finished runs of {n_full} claims\n")

    print("## Confusion (rows gold S/R/NEI; columns predicted S/R/NEI/other)")
    for name, recs in runs.items():
        c = Counter((r["gold_label"], r["pred_label"]) for r in recs.values())
        rows = [" ".join(f"{c[(g, p)]:3d}" for p in LABELS) + f" {sum(v for (g2, p), v in c.items() if g2 == g and p not in LABELS):3d}"
                for g in LABELS]
        print(f"{name:16s} | " + " | ".join(rows))

    print("\n## Ledger systems: verdict reasons of wrong verdicts, and what wrong REFUTES rest on")
    for name, recs in runs.items():
        if not any("ledger" in r for r in recs.values()):
            continue
        reasons = Counter((r["gold_label"], r["pred_label"], r.get("verdict_reason")) for r in recs.values()
                          if r["pred_label"] != r["gold_label"])
        types = Counter()
        for r in recs.values():
            if r["pred_label"] == "REFUTES" and r["gold_label"] != "REFUTES" and "ledger" in r:
                st = final_status(r["ledger"])
                for o in r["ledger"]["obligations"]:
                    if st.get(o["obligation_id"]) == "CONTRADICTED":
                        types[f"{o['criticality']}/{o['type']}"] += 1
        print(f"- {name}: " + "; ".join(f"{g}->{p} ({why}) {n}" for (g, p, why), n in reasons.most_common(6)))
        print(f"  contradicted obligations in wrong REFUTES: {dict(types.most_common())}")

    if "b1" in runs and "m1" in runs:
        b1, m1 = runs["b1"], runs["m1"]
        ok = lambda r: r["pred_label"] == r["gold_label"]
        lost = Counter((m1[i]["gold_label"], m1[i]["pred_label"], m1[i].get("verdict_reason"))
                       for i in b1 if i in m1 and ok(b1[i]) and not ok(m1[i]))
        won = Counter((m1[i]["gold_label"], b1[i]["pred_label"]) for i in b1 if i in m1 and ok(m1[i]) and not ok(b1[i]))
        print(f"\n## B1 vs M1: B1 right / M1 wrong {sum(lost.values())}, M1 right / B1 wrong {sum(won.values())}")
        print("  M1 losses: " + "; ".join(f"{g}->{p} ({why}) {n}" for (g, p, why), n in lost.most_common(6)))
        print("  M1 wins (gold, B1's wrong label): " + "; ".join(f"{g}/{p} {n}" for (g, p), n in won.most_common(6)))

    for name in ("m2", "m2_random", "m2_claimiter", "m2_noconsol", "m2_claimfollow"):
        if name not in runs:
            continue
        recs = runs[name].values()
        first = lambda r: r.get("m1_label", r.get("first_label"))       # M2 family / M2-claimiter
        changed = [r for r in recs if r.get("verdict_changed")]
        fixed = Counter((r["gold_label"], first(r), r["pred_label"]) for r in changed if r["pred_label"] == r["gold_label"])
        broken = Counter((r["gold_label"], first(r), r["pred_label"]) for r in changed
                         if first(r) == r["gold_label"] and r["pred_label"] != r["gold_label"])
        entered = sum(bool(r["entered_m2"]) if "entered_m2" in r else bool(r.get("rounds")) for r in recs)
        print(f"\n## {name}: entered the loop {entered}; verdicts changed {len(changed)}; fixed {sum(fixed.values())}, "
              f"broken {sum(broken.values())}")
        print("  fixed (gold, first pass, final): " + "; ".join(f"{k} {v}" for k, v in fixed.most_common(5)))
        print("  broken (gold, first pass, final): " + "; ".join(f"{k} {v}" for k, v in broken.most_common(5)))

    if "m2" in runs and "m2_random" in runs:
        m2, rnd = runs["m2"], runs["m2_random"]
        looped = differ = same = 0
        for i in m2:
            ta = [x["targets"] for x in m2[i]["ledger"]["retrieval_rounds"] if x["targets"]]
            tb = [x["targets"] for x in rnd[i]["ledger"]["retrieval_rounds"] if x["targets"]] if i in rnd else []
            if ta:
                looped += 1
                differ += (not tb) or sorted(ta[0]) != sorted(tb[0])
                same += m2[i]["pred_label"] == rnd[i]["pred_label"]
        print(f"\n## M2-random vs M2: {looped} claims entered the loop; round-1 targets differ in {differ}; "
              f"same final label in {same}")


if __name__ == "__main__":
    main()
