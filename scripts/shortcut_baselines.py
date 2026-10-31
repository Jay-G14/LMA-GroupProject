"""Non-LLM shortcut baselines that quantify how much PUBHEALTH labels leak through
identifiers, metadata and text style. Hyperparameters are chosen on dev; test is
scored once with the dev-selected setting.

Usage: python scripts/shortcut_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.masking import mask_sentences, split_sentences  # noqa: E402
from oblig.data.pubhealth import LABELS, load_all  # noqa: E402

SEED = 13


def scores(y, p) -> dict:
    return {"accuracy": round(accuracy_score(y, p), 4),
            "macro_f1": round(f1_score(y, p, labels=list(LABELS), average="macro", zero_division=0), 4)}


def masked(text: str) -> str:
    return " ".join(mask_sentences(split_sentences(text))[0])


def text_baseline(tr, dv, te, field_fn) -> dict:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    Xtr = vec.fit_transform(field_fn(tr)); Xdv = vec.transform(field_fn(dv)); Xte = vec.transform(field_fn(te))
    best = None
    for C in (0.1, 0.3, 1, 3, 10):
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED).fit(Xtr, tr.label)
        s = scores(dv.label, m.predict(Xdv))
        if best is None or s["macro_f1"] > best[1]["macro_f1"]:
            best = (C, s, m)
    C, dev_s, m = best
    return {"C": C, "dev": dev_s, "test": scores(te.label, m.predict(Xte))}


def main() -> None:
    S = load_all()
    tr, dv, te = (S[s].data.copy() for s in ("train", "dev", "test"))
    R = {"majority": {"dev": scores(dv.label, ["true"] * len(dv)), "test": scores(te.label, ["true"] * len(te))}}

    # claim_id range: majority label per 500-id bin, learned on train.
    for d in (tr, dv, te):
        d["bin"] = d.claim_id.astype(int) // 500
    maj = tr.groupby("bin").label.agg(lambda x: x.mode()[0])
    R["claim_id_bins"] = {s: scores(d.label, d.bin.map(maj).fillna("true")) for s, d in (("dev", dv), ("test", te))}

    R["claim_only_tfidf_lr"] = text_baseline(tr, dv, te, lambda d: d.claim)
    R["masked_main_text_tfidf_lr"] = text_baseline(tr, dv, te, lambda d: d.main_text.map(masked))
    R["metadata_tfidf_lr"] = text_baseline(
        tr, dv, te, lambda d: ("fc_" + d.fact_checkers.str.replace(r"\W+", "_", regex=True) + " "
                               + d.subjects + " src_" + (d.sources.str.strip() != "").astype(str)))
    out = ROOT / "results" / "metrics" / "shortcut_baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=2))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
