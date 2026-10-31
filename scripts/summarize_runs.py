"""Cross-eval-set summary for reports: reads results/comparisons-style JSON written by compare_systems.py for
several eval sets and prints (and writes) Markdown tables: macro-F1 [95% CI] per system and set, the
capability metrics (RAG: sentence-level F1, agentic: loop gain, safety: unsupported-verdict rate), cost, and
the paired comparisons.

Usage:
  python scripts/summarize_runs.py --comparisons to_transfer/pilot/comparisons/pilot.json \\
      to_transfer/dev/comparisons/dev.json to_transfer/open/comparisons/open.json --out reports/summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["B0", "B1", "B1-9", "B1-SC", "B2", "M1", "M1-shared", "M1-aligned", "M1-untyped", "M1-LLMagg", "M2",
         "M2-random", "M2-claimiter"]


def fmt(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparisons", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = [json.loads(Path(p).read_text()) for p in args.comparisons]
    names = [f"{s['eval_set']} (n={s['n_claims']})" for s in sets]
    rows = {}
    for i, s in enumerate(sets):
        for t in s["table"]:
            rows.setdefault(t["system"], {})[i] = t
    systems = [x for x in ORDER if x in rows] + [x for x in rows if x not in ORDER]

    lines = ["## Claim-level macro-F1 [95% bootstrap CI]", "",
             "| system | " + " | ".join(names) + " |", "|---|" + "---|" * len(sets)]
    for sname in systems:
        cells = []
        for i in range(len(sets)):
            t = rows[sname].get(i)
            cells.append("–" if t is None else f"{t['macro_f1']:.3f} [{t['macro_f1_ci95'][0]:.3f}, {t['macro_f1_ci95'][1]:.3f}]")
        lines.append(f"| {sname} | " + " | ".join(cells) + " |")

    for i, s in enumerate(sets):
        lines += ["", f"## {names[i]}: capability and cost metrics", "",
                  "| system | acc | F1 S / R / NEI | sentence F1 (RAG) | abstract F1 | unsupported (safety) | "
                  "gold-abstract recall | calls / claim | tokens / claim |", "|---|---|---|---|---|---|---|---|---|"]
        for sname in systems:
            t = rows[sname].get(i)
            if t is None:
                continue
            lines.append(f"| {sname} | {t['accuracy']:.3f} | {t['f1_SUPPORTS']:.2f} / {t['f1_REFUTES']:.2f} / {t['f1_NEI']:.2f} "
                         f"| {fmt(t['sentence_f1'])} | {fmt(t['abstract_f1'])} ({t['abstract_metric']}) "
                         f"| {fmt(t['unsupported_verdict_rate'])} | {fmt(t['gold_abstract_recall'])} "
                         f"| {t['calls_per_claim']} | {t['tokens_per_claim']:.0f} |")
        loops = {k: m["iteration"] for k, m in s["metrics"].items() if "iteration" in m}
        if loops:
            lines += ["", "Agentic loop (first pass -> iterated): Δ macro-F1 [95% CI], Δ gold-abstract recall, "
                          "extra calls per claim, fixed / broken verdicts", ""]
            for k, it in loops.items():
                g = it["loop_gain_macro_f1"]
                lines.append(f"- {k}: {g['delta']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}], "
                             f"{fmt(it['loop_gain_recall'])}, {it['extra_calls_per_claim']}, "
                             f"{it['changes_fixing_error']} / {it['changes_introducing_error']}")
        if s["comparisons"]:
            lines += ["", "Paired comparisons (b - a): Δ macro-F1 [95% CI], Holm-adjusted p (bootstrap; McNemar)", ""]
            for c in s["comparisons"]:
                f, mc = c["macro_f1"], c["accuracy_mcnemar"]
                lines.append(f"- {c['name']} ({c['rq']}, {c.get('family', 'primary')} family): "
                             f"{f['delta']:+.3f} [{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}], "
                             f"p = {f['p_holm']:.3f}; {mc['p_holm']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
