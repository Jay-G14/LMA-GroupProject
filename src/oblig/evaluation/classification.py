"""Classification metrics, bootstrap confidence intervals and paired significance tests.

The label set is always passed explicitly by SciFact code; the PUBHEALTH label set is only the default
for the PUBHEALTH scripts. Invalid / unparseable predictions are scored as wrong, never dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ..data.pubhealth import LABELS

INVALID = "__invalid__"  # prediction for a failed/unparseable call; always counted wrong


def read_predictions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def macro_f1(gold, pred, labels=LABELS) -> float:
    return float(f1_score(gold, pred, labels=list(labels), average="macro", zero_division=0))


def classification_report(gold: list[str], pred: list[str], labels=LABELS) -> dict:
    pred = [p if p in labels else INVALID for p in pred]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=list(labels), zero_division=0)
    return {"n": len(gold), "macro_f1": round(macro_f1(gold, pred, labels), 4),
            "accuracy": round(float(accuracy_score(gold, pred)), 4),
            "invalid_predictions": int(sum(x == INVALID for x in pred)),
            "per_class": {l: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                              "f1": round(float(f[i]), 4), "support": int(s[i])} for i, l in enumerate(labels)},
            "confusion_matrix": {"labels": list(labels) + [INVALID],
                                 "matrix": confusion_matrix(gold, pred, labels=list(labels) + [INVALID]).tolist()}}


# -- fast resampling ------------------------------------------------------------------------
def _codes(seq, labels) -> np.ndarray:
    """Integer codes; anything outside `labels` (invalid outputs) gets the extra code len(labels)."""
    index = {l: i for i, l in enumerate(labels)}
    return np.fromiter((index.get(x, len(labels)) for x in seq), dtype=np.int64)


def _macro_f1_codes(g: np.ndarray, p: np.ndarray, k: int) -> float:
    """Macro-F1 over the k real labels; identical to sklearn's f1_score(average='macro', zero_division=0)."""
    cm = np.bincount(g * (k + 1) + p, minlength=(k + 1) ** 2).reshape(k + 1, k + 1)
    tp, n_pred, n_gold = np.diag(cm)[:k], cm[:, :k].sum(0), cm[:k, :].sum(1)
    prec = np.divide(tp, n_pred, out=np.zeros(k), where=n_pred > 0)
    rec = np.divide(tp, n_gold, out=np.zeros(k), where=n_gold > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return float(f1.mean())


def _metric_fn(metric: str, k: int):
    if metric == "macro_f1":
        return lambda g, p: _macro_f1_codes(g, p, k)
    if metric == "accuracy":
        return lambda g, p: float((g == p).mean())
    raise ValueError(f"unknown metric {metric!r}")


def bootstrap_ci(gold, pred, labels=LABELS, metric: str = "macro_f1", n_boot: int = 10_000,
                 seed: int = 13) -> dict:
    """Percentile 95% CI of a metric for one system (claims resampled with replacement)."""
    g, p = _codes(gold, labels), _codes(pred, labels)
    fn, rng, n = _metric_fn(metric, len(labels)), np.random.default_rng(seed), len(g)
    vals = np.array([fn(g[idx], p[idx]) for idx in rng.integers(0, n, (n_boot, n))])
    return {"value": round(fn(g, p), 4), "ci95": [round(float(np.percentile(vals, 2.5)), 4),
                                                  round(float(np.percentile(vals, 97.5)), 4)], "n_boot": n_boot}


def paired_bootstrap(gold, pred_a, pred_b, labels=LABELS, metric: str = "macro_f1", n_boot: int = 10_000,
                     seed: int = 13) -> dict:
    """95% CI of metric(b) - metric(a), resampling the same claims for both systems.
    p_two_sided is the bootstrap two-sided p-value for 'no difference'."""
    g, a, b = _codes(gold, labels), _codes(pred_a, labels), _codes(pred_b, labels)
    fn, rng, n = _metric_fn(metric, len(labels)), np.random.default_rng(seed), len(g)
    deltas = np.empty(n_boot)
    for i, idx in enumerate(rng.integers(0, n, (n_boot, n))):
        deltas[i] = fn(g[idx], b[idx]) - fn(g[idx], a[idx])
    p_le, p_ge = float((deltas <= 0).mean()), float((deltas >= 0).mean())
    return {"delta": round(fn(g, b) - fn(g, a), 4),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)],
            "p_delta_le_0": round(p_le, 4), "p_two_sided": round(min(1.0, 2 * min(p_le, p_ge)), 4),
            "n": n, "n_boot": n_boot}


def mcnemar(gold, pred_a, pred_b) -> dict:
    """Exact McNemar test on per-claim correctness (two-sided)."""
    ca = np.asarray([x == y for x, y in zip(pred_a, gold)])
    cb = np.asarray([x == y for x, y in zip(pred_b, gold)])
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": round(float(p), 4)}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control across comparisons)."""
    order = sorted(pvalues, key=pvalues.get)
    m, out, running = len(order), {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = round(running, 4)
    return out
