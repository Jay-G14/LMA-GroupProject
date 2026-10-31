"""Compare SciFact runs on one evaluation set: a metrics table for every system and paired tests for the
comparisons in configs/experiment_matrix.yaml (protocol section 5, amendment 2).

Every comparison reports the paired-bootstrap 95% CI of the macro-F1 difference, an exact McNemar test on
per-claim correctness, and the change in unsupported-verdict rate on gold-NEI claims. p-values are
Holm-adjusted across all comparisons that were run. Writes results/comparisons/<name>.{json,md}.

Usage:
  python scripts/compare_systems.py --eval-set prompt_dev --prefix smoke2_ --name smoke2
  python scripts/compare_systems.py --eval-set dev --runs dev_b0 dev_b1 dev_m1 dev_m2 --name dev_main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.scifact import DEV_SPLITS, HELD_OUT_SETS, LABELS, load_corpus, load_eval_set  # noqa: E402
from oblig.evaluation.classification import holm, mcnemar, paired_bootstrap  # noqa: E402
from oblig.evaluation.scifact import evaluate  # noqa: E402
from oblig.ledger.aggregation import POLICY_TAGS  # noqa: E402

NAMES = {"b0": "B0", "b1": "B1", "b1_9": "B1-9", "b1_sc": "B1-SC", "b2": "B2", "m1": "M1", "m1_shared": "M1-shared",
         "m1_aligned": "M1-aligned", "m1_untyped": "M1-untyped", "m1_llmagg": "M1-LLMagg", "m2": "M2",
         "m2_random": "M2-random", "m2_claimiter": "M2-claimiter", "b1_checked": "B1-checked",
         "b1_checked_b2": "B1-checked-by-B2", "b1_checked_b1_9": "B1-checked-by-B1-9"}
# Settings frozen at Gate 3 (docs/EXPERIMENT_PROTOCOL.md); a run that deviates is an ablation variant of its system.
FROZEN = {"sc_samples": 4, "m2_followup": "fused", "m2_consolidate": True, "policy": "scifact_v1"}


def run_key(cfg: dict) -> str:
    """System id plus the variant the run's configuration encodes, e.g. m2_noconsol or b1_sc_5samples."""
    s, tags = cfg["system"], []
    if s == "b1_sc" and cfg.get("sc_samples", FROZEN["sc_samples"]) != FROZEN["sc_samples"]:
        tags.append(f"{cfg['sc_samples']}samples")
    if s.startswith("m2"):
        if cfg.get("m2_followup", FROZEN["m2_followup"]) != FROZEN["m2_followup"]:
            tags.append(f"{cfg['m2_followup']}follow")
        if cfg.get("m2_consolidate", FROZEN["m2_consolidate"]) != FROZEN["m2_consolidate"]:
            tags.append("noconsol")
    if cfg.get("policy", FROZEN["policy"]) != FROZEN["policy"]:      # re-scored ledgers (amendment 5)
        tags.append(POLICY_TAGS[cfg["policy"]])
    if s == "b1_checked" and not cfg.get("checker", "m1").startswith("m1"):   # checks without a ledger (amendment 6)
        tags.append(cfg["checker"])
    return "_".join([s] + tags)


def display(key: str) -> str:
    if key in NAMES:
        return NAMES[key]
    base = max((s for s in NAMES if key.startswith(s + "_")), key=len, default=None)
    return f"{NAMES[base]}-{key[len(base) + 1:].replace('_', '-')}" if base else key


def order(key: str) -> tuple:
    base = key if key in NAMES else max((s for s in NAMES if key.startswith(s + "_")), key=len, default=None)
    return (list(NAMES).index(base) if base else 99, key)


def load_run(exp_id: str, results: Path) -> tuple[dict, dict[int, dict]]:
    pred_path = results / "predictions" / f"{exp_id}.jsonl"
    cfg = json.loads(pred_path.with_suffix(".config.json").read_text(encoding="utf-8"))
    with open(pred_path, encoding="utf-8") as f:
        preds = {r["claim_id"]: r for r in map(json.loads, f)}
    return cfg, preds


def paired_rate_delta(x_a: np.ndarray, x_b: np.ndarray, n_boot: int, seed: int = 13) -> dict | None:
    """Bootstrap CI of mean(x_b) - mean(x_a) for paired 0/1 indicators."""
    if not len(x_a):
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x_a), (n_boot, len(x_a)))
    d = x_b[idx].mean(1) - x_a[idx].mean(1)
    return {"delta": round(float(x_b.mean() - x_a.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}


def fmt(x, digits=3):
    return "–" if x is None else f"{x:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", choices=list(DEV_SPLITS + HELD_OUT_SETS), required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--runs", nargs="+", help="experiment ids")
    g.add_argument("--prefix", help="every prediction file whose experiment id starts with this prefix")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    args = ap.parse_args()
    results = Path(args.results_dir)

    exp_ids = args.runs or sorted(p.stem for p in (results / "predictions").glob(f"{args.prefix}*.jsonl"))
    runs = {}
    for exp_id in exp_ids:
        cfg, preds = load_run(exp_id, results)
        if cfg.get("eval_set") != args.eval_set:
            print(f"skip {exp_id}: eval set {cfg.get('eval_set')!r}")
            continue
        key = run_key(cfg)
        if key in runs:
            sys.exit(f"two runs of {key} ({runs[key]['exp_id']}, {exp_id}); pass --runs")
        runs[key] = {"exp_id": exp_id, "cfg": cfg, "preds": preds}
    if not runs:
        sys.exit("no runs found")
    claims = load_eval_set(args.eval_set, load_corpus().keys())
    common = set.intersection(*(set(r["preds"]) for r in runs.values()))
    claims = [c for c in claims if c.id in common]
    print(f"{len(runs)} systems, {len(claims)} claims scored by all of them")

    table, metrics = [], {}
    for sysid in sorted(runs, key=order):
        r = runs[sysid]
        m = evaluate(claims, r["preds"], n_boot=args.n_boot)
        metrics[sysid] = m
        recs = [r["preds"][c.id] for c in claims]
        cl, ab, sl, ev = m["claim_level"], m["abstract_level"], m["sentence_level"], m["evidence"]
        table.append({"system": display(sysid), "exp_id": r["exp_id"], "macro_f1": cl["macro_f1"],
                      "macro_f1_ci95": cl["macro_f1_ci95"], "accuracy": cl["accuracy"],
                      **{f"f1_{l}": cl["per_class"][l]["f1"] for l in LABELS},
                      "unsupported_verdict_rate": m["safety"]["unsupported_verdict_rate"],
                      "abstract_f1": (ab["label_rationale"] or ab["label_only"])["f1"],
                      "abstract_metric": "label+rationale" if ab["label_rationale"] else "label-only",
                      "sentence_f1": sl["selection_label"]["f1"], "gold_abstract_recall": ev["gold_abstract_recall"],
                      "calls_per_claim": round(sum(len(x["calls"]) for x in recs) / len(recs), 2),
                      "tokens_per_claim": round(sum(x.get("prompt_tokens", 0) + x.get("completion_tokens", 0)
                                                    for x in recs) / len(recs), 1)})

    gold = [c.label for c in claims]
    nei = [i for i, c in enumerate(claims) if c.label == "NEI"]
    matrix = yaml.safe_load(open(ROOT / "configs" / "experiment_matrix.yaml", encoding="utf-8"))
    comps = []
    ind = lambda p: np.array([p[i] in ("SUPPORTS", "REFUTES") for i in nei], dtype=float)
    # The pre-registered comparisons form one Holm family; the amendment ablations (planned at Gate 3) another,
    # and the post-hoc amendment 5 a third, so adding them leaves the frozen family's adjustment unchanged.
    for family, entries in (("primary", matrix["comparisons"]), ("ablation", matrix.get("ablation_comparisons", [])),
                            ("amendment 5", matrix.get("amendment_comparisons", [])),
                            ("amendment 6", matrix.get("check_comparisons", []))):
        fam = []
        for comp in entries:
            a, b = comp["a"], comp["b"]
            if a not in runs or b not in runs:
                continue
            pa = [runs[a]["preds"][c.id]["pred_label"] for c in claims]
            pb = [runs[b]["preds"][c.id]["pred_label"] for c in claims]
            fam.append({"name": f"{display(a)} vs {display(b)}", "family": family, "rq": comp.get("rq"),
                        "question": comp.get("question"),
                        "macro_f1": paired_bootstrap(gold, pa, pb, labels=LABELS, n_boot=args.n_boot),
                        "accuracy_mcnemar": mcnemar(gold, pa, pb),
                        "unsupported_verdict_rate": paired_rate_delta(ind(pa), ind(pb), args.n_boot)})
        if fam:
            adj_boot = holm({c["name"]: c["macro_f1"]["p_two_sided"] for c in fam})
            adj_mcn = holm({c["name"]: c["accuracy_mcnemar"]["p_value"] for c in fam})
            for c in fam:
                c["macro_f1"]["p_holm"], c["accuracy_mcnemar"]["p_holm"] = adj_boot[c["name"]], adj_mcn[c["name"]]
        comps += fam

    out_dir = results / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.name}.json").write_text(json.dumps(
        {"eval_set": args.eval_set, "n_claims": len(claims), "n_boot": args.n_boot, "table": table,
         "comparisons": comps, "metrics": metrics}, indent=2))

    lines = [f"# {args.name}: {args.eval_set}, {len(claims)} claims", "",
             "Macro-F1 with 95% bootstrap CI. Unsupported = gold-NEI claims given SUPPORTS/REFUTES (safety). "
             "Abstract = SciFact abstract-level F1 (label+rationale, or label-only where rationale sets are merged). "
             "Sentence = SciFact sentence-level selection+label F1 (RAG metric). Recall = gold-abstract recall of the "
             "abstracts shown.", "",
             "| system | macro-F1 [95% CI] | acc | F1 S / R / NEI | unsupported | abstract | sentence | recall | calls | tokens |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        lines.append(f"| {t['system']} | {t['macro_f1']:.3f} [{t['macro_f1_ci95'][0]:.3f}, {t['macro_f1_ci95'][1]:.3f}] "
                     f"| {t['accuracy']:.3f} | {t['f1_SUPPORTS']:.2f} / {t['f1_REFUTES']:.2f} / {t['f1_NEI']:.2f} "
                     f"| {fmt(t['unsupported_verdict_rate'])} | {fmt(t['abstract_f1'])} | {fmt(t['sentence_f1'])} "
                     f"| {fmt(t['gold_abstract_recall'])} | {t['calls_per_claim']} | {t['tokens_per_claim']:.0f} |")
    if comps:
        lines += ["", "## Paired comparisons (b − a)", "",
                  "Holm adjustment within each family: the pre-registered comparisons, the amendment ablations, and "
                  "the post-hoc amendments 5 and 6.", "",
                  "| comparison | family | RQ | Δ macro-F1 [95% CI] | p (Holm) | McNemar a-only / b-only, p (Holm) | Δ unsupported [95% CI] |",
                  "|---|---|---|---|---|---|---|"]
        for c in comps:
            f, mc, u = c["macro_f1"], c["accuracy_mcnemar"], c["unsupported_verdict_rate"]
            u_str = "–" if u is None else "{:+.3f} [{:+.3f}, {:+.3f}]".format(u["delta"], *u["ci95"])
            lines.append(f"| {c['name']} | {c['family']} | {c['rq'] or ''} "
                         f"| {f['delta']:+.3f} [{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}] "
                         f"| {f['p_holm']:.3f} | {mc['only_a_correct']} / {mc['only_b_correct']}, {mc['p_holm']:.3f} "
                         f"| {u_str} |")
    loops = {s: m["iteration"] for s, m in metrics.items() if "iteration" in m}
    if loops:
        lines += ["", "## Agentic loop (first pass → iterated, within each run)", "",
                  "| system | loop gain Δ macro-F1 [95% CI] | Δ gold-abstract recall | extra calls / claim | "
                  "Δ macro-F1 per call | fixed / broken verdicts | entering loop |", "|---|---|---|---|---|---|---|"]
        for s, it in loops.items():
            g_ = it["loop_gain_macro_f1"]
            lines.append(f"| {display(s)} | {g_['delta']:+.3f} [{g_['ci95'][0]:+.3f}, {g_['ci95'][1]:+.3f}] "
                         f"| {fmt(it['loop_gain_recall'])} | {it['extra_calls_per_claim']} "
                         f"| {fmt(it['loop_gain_macro_f1_per_extra_call'], 4)} "
                         f"| {it['changes_fixing_error']} / {it['changes_introducing_error']} "
                         f"| {fmt(it['share_entering_iteration'])} |")
    leds = {s: m["ledger"] for s, m in metrics.items() if "ledger" in m}
    if leds:
        lines += ["", "## Ledger diagnostics", "",
                  "| system | obligations / claim | planner failures | resolving citations on a gold abstract | "
                  "cited sentences in gold rationale |", "|---|---|---|---|---|"]
        for s, l in leds.items():
            lines.append(f"| {display(s)} | {fmt(l['obligations_per_claim'], 2)} | {fmt(l['planner_failure_rate'])} "
                         f"| {fmt(l['resolving_citations_on_gold_abstract'])} "
                         f"| {fmt(l['resolving_cited_sentences_in_gold_rationale'])} |")
    (out_dir / f"{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
