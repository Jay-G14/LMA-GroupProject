"""How well does each check on B1 pick out B1's errors? (protocol amendment 6)

A check turns B1's SUPPORTS/REFUTES answer into NEI wherever the checking system gives a different verdict
(scripts/rescore_ledgers.py writes these runs as <prefix>b1_checked*). For each check: how many of B1's answers it
vetoes, how many of those were wrong (veto precision), and how a random veto of the same number of B1's
SUPPORTS/REFUTES answers does: exact hypergeometric p for the veto precision, and macro-F1 and unsupported-verdict
rate over 10,000 seeded draws. B1-SC unanimity (NEI unless all sampled verdicts equal B1's) is shown for reference.

  python scripts/veto_analysis.py --results-dir to_transfer/dev --prefix dev_ --name dev
writes <results>/comparisons/veto_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.scifact import LABELS  # noqa: E402
from oblig.evaluation.classification import macro_f1  # noqa: E402

CHECKS = {"M1": "b1_checked", "M1-anch": "b1_checked_anch", "B2": "b1_checked_b2", "B1-9": "b1_checked_b1_9"}
VERDICTS = ("SUPPORTS", "REFUTES")


def read(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["claim_id"]: r for r in map(json.loads, f)}


def unsupported(gold: list[str], pred: list[str]) -> float:
    nei = [p for g, p in zip(gold, pred) if g == "NEI"]
    return sum(p in VERDICTS for p in nei) / len(nei)


def p_at_least(n: int, w: int, k: int, x: int) -> float:
    """P(X >= x) for X ~ hypergeometric: k draws without replacement from n items of which w are marked."""
    return sum(comb(w, i) * comb(n - w, k - i) for i in range(x, min(k, w) + 1)) / comb(n, k)


def vectorised_scores(gold: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Macro-F1 and unsupported rate for each row of preds (labels as indices into LABELS)."""
    f1 = []
    for c in range(len(LABELS)):
        tp = ((preds == c) & (gold == c)).sum(1)
        fp = ((preds == c) & (gold != c)).sum(1)
        fn = ((preds != c) & (gold == c)).sum(1)
        f1.append(np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0))
    nei, verdict = LABELS.index("NEI"), np.isin(preds, [LABELS.index(v) for v in VERDICTS])
    unsup = (verdict & (gold == nei)).sum(1) / (gold == nei).sum()
    return np.mean(f1, axis=0), unsup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--draws", type=int, default=10_000)
    args = ap.parse_args()
    pred = Path(args.results_dir) / "predictions"
    b1 = read(pred / f"{args.prefix}b1.jsonl")
    ids = sorted(b1)
    gold = [b1[i]["gold_label"] for i in ids]
    base = [b1[i]["pred_label"] for i in ids]
    answered = [k for k, p in enumerate(base) if p in VERDICTS]
    n, w = len(answered), sum(base[k] != gold[k] for k in answered)

    g_idx = np.array([LABELS.index(g) for g in gold])
    b_idx = np.array([LABELS.index(p) if p in LABELS else -1 for p in base])
    rng = np.random.default_rng(13)
    out = {"n_claims": len(ids), "b1_answers": n, "b1_wrong_answers": w, "b1_error_rate": round(w / n, 4),
           "b1_macro_f1": round(macro_f1(gold, base, labels=LABELS), 4), "b1_unsupported": round(unsupported(gold, base), 4),
           "checks": {}}
    sc = read(pred / f"{args.prefix}b1_sc.jsonl")
    out["sc_unanimous_vetoes"] = sum(set(sc[ids[k]]["votes"]) != {base[k]} for k in answered)
    for name, run in CHECKS.items():
        checked = read(pred / f"{args.prefix}{run}.jsonl")
        new = [checked[i]["pred_label"] for i in ids]
        vetoed = [k for k in answered if new[k] == "NEI"]
        assert all(new[k] == base[k] for k in range(len(ids)) if k not in set(vetoed)), "a check may only veto"
        k_, x = len(vetoed), sum(base[k] != gold[k] for k in vetoed)
        draws = np.tile(b_idx, (args.draws, 1))
        for row in draws:
            row[rng.choice(answered, size=k_, replace=False)] = LABELS.index("NEI")
        f1, un = vectorised_scores(g_idx, draws)
        obs_f1, obs_un = macro_f1(gold, new, labels=LABELS), unsupported(gold, new)
        out["checks"][name] = {
            "run": f"{args.prefix}{run}", "vetoes": k_, "vetoed_wrong": x, "veto_precision": round(x / k_, 4) if k_ else None,
            "p_veto_precision": round(p_at_least(n, w, k_, x), 6) if k_ else None,
            "vetoed_right_lost": k_ - x, "vetoed_gold_nei": sum(gold[k] == "NEI" for k in vetoed),
            "macro_f1": round(obs_f1, 4), "unsupported": round(obs_un, 4),
            "random_macro_f1": [round(float(np.mean(f1)), 4), round(float(np.percentile(f1, 2.5)), 4), round(float(np.percentile(f1, 97.5)), 4)],
            "random_unsupported": [round(float(np.mean(un)), 4), round(float(np.percentile(un, 2.5)), 4), round(float(np.percentile(un, 97.5)), 4)],
            "share_random_at_least_as_good_f1": round(float(np.mean(f1 >= obs_f1 - 1e-12)), 4),
            "share_random_at_least_as_safe": round(float(np.mean(un <= obs_un + 1e-12)), 4)}
    out_dir = Path(args.results_dir) / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"veto_{args.name}.json").write_text(json.dumps(out, indent=2))

    print(f"{args.name}: B1 answers {n} claims with SUPPORTS/REFUTES, {w} wrong (error rate {w / n:.3f}); "
          f"B1-SC unanimity would veto {out['sc_unanimous_vetoes']}")
    print("| check | vetoes | of which B1 wrong | veto precision | p (random veto) | macro-F1 (random) | unsupported (random) |")
    print("|---|---|---|---|---|---|---|")
    for name, c in out["checks"].items():
        rf, ru = c["random_macro_f1"], c["random_unsupported"]
        print(f"| {name} | {c['vetoes']} | {c['vetoed_wrong']} | {c['veto_precision']:.3f} | {c['p_veto_precision']:.4f} "
              f"| {c['macro_f1']:.3f} ({rf[0]:.3f} [{rf[1]:.3f}, {rf[2]:.3f}]) "
              f"| {c['unsupported']:.3f} ({ru[0]:.3f} [{ru[1]:.3f}, {ru[2]:.3f}]) |")


if __name__ == "__main__":
    main()
