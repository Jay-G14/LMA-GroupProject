"""Measure PUBHEALTH (and SciFact) statistics; write reports/dataset_audit.json and figures.

Usage: python scripts/audit_dataset.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data import leakage  # noqa: E402
from oblig.data.pubhealth import (COLUMNS, LABELS, SPLITS, default_root, file_sha256,  # noqa: E402
                                  load_all, split_list)

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"


def words(s: pd.Series) -> pd.Series:
    return s.str.split().str.len().fillna(0).astype(int)


def describe(x: pd.Series) -> dict:
    if len(x) == 0:
        return {}
    q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(x)), "mean": round(float(x.mean()), 1), "std": round(float(x.std()), 1),
            "min": int(x.min()), "p05": float(q[0.05]), "p25": float(q[0.25]), "median": float(q[0.5]),
            "p75": float(q[0.75]), "p95": float(q[0.95]), "max": int(x.max())}


def url_ok(u: str) -> bool:
    p = urlparse(u)
    return p.scheme in ("http", "https") and "." in p.netloc


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    splits = load_all()
    frames = {s: splits[s].data for s in SPLITS}
    allv = pd.concat(frames.values(), ignore_index=True)
    A: dict = {"dataset": "PUBHEALTH", "files": {}, "splits": {}}

    for s in SPLITS:
        p = default_root() / f"{s}.tsv"
        A["files"][s] = {"path": str(p.relative_to(ROOT)).replace("\\", "/"),
                         "sha256": file_sha256(p), "bytes": p.stat().st_size}

    # --- per-split sizes, labels, missingness, lengths ---------------------------------
    for s, sp in splits.items():
        df, raw = sp.data, sp.raw
        lab = df["label"].value_counts().reindex(LABELS, fill_value=0)
        miss = {c: int((df[c].str.strip() == "").sum()) for c in COLUMNS}
        has_src = df["sources"].str.strip() != ""
        A["splits"][s] = {
            "rows_parsed": len(raw), "rows_valid": len(df), "rows_malformed": sp.n_malformed,
            "malformed_raw_labels": dict(Counter(raw.loc[~raw["row"].isin(df["row"]), "label"].str[:40])),
            "labels": lab.to_dict(), "label_pct": (lab / lab.sum() * 100).round(2).to_dict(),
            "missing_in_valid_rows": miss,
            "duplicate_claim_id_within": int(df["claim_id"].duplicated().sum()),
            "duplicate_claim_text_within": int(df["claim"].str.lower().duplicated().sum()),
            "duplicate_main_text_within": int(df.loc[df.main_text.str.strip() != "", "main_text"].duplicated().sum()),
            "words": {"claim": describe(words(df["claim"])),
                      "main_text": describe(words(df["main_text"])),
                      "explanation": describe(words(df["explanation"])),
                      "main_text_with_sources": describe(words(df.loc[has_src, "main_text"])),
                      "main_text_without_sources": describe(words(df.loc[~has_src, "main_text"]))},
            "replacement_char_rate_main_text": round(float(df["main_text"].str.contains("�").mean()), 4),
        }

    # --- pooled statistics ---------------------------------------------------------------
    has_src = allv["sources"].str.strip() != ""
    A["pooled"] = {
        "rows_valid": len(allv),
        "labels": allv["label"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
        "records_without_sources": int((~has_src).sum()),
        "words_main_text": describe(words(allv["main_text"])),
        "words_main_text_without_sources": describe(words(allv.loc[~has_src, "main_text"])),
        "label_by_source_availability": pd.crosstab(has_src.map({True: "has_sources", False: "no_sources"}),
                                                    allv["label"]).to_dict(orient="index"),
        "label_by_fact_checker_missing": pd.crosstab((allv["fact_checkers"].str.strip() == "").map(
            {True: "no_fact_checker", False: "has_fact_checker"}), allv["label"]).to_dict(orient="index"),
    }
    urls = [u for s in allv["sources"] for u in split_list(s)]
    doms = Counter(urlparse(u).netloc.lower().removeprefix("www.") for u in urls if url_ok(u))
    A["pooled"]["source_urls"] = {"total": len(urls), "malformed": sum(not url_ok(u) for u in urls),
                                  "unique": len(set(urls)), "top_domains": doms.most_common(25)}
    fc = Counter(x for f in allv["fact_checkers"] for x in split_list(f))
    sj = Counter(x.lower() for f in allv["subjects"] for x in split_list(f))
    A["pooled"]["fact_checkers"] = {"unique": len(fc), "top": fc.most_common(30)}
    A["pooled"]["subjects"] = {"unique": len(sj), "top": sj.most_common(40)}

    # --- leakage and overlap ------------------------------------------------------------
    L: dict = {}
    L["exact_claim_overlap"] = leakage.cross_split_exact(frames, lambda d: d["claim"].map(leakage.norm_claim))
    L["claim_id_overlap"] = leakage.cross_split_exact(frames, lambda d: d["claim_id"])
    L["main_text_overlap"] = leakage.cross_split_exact(frames, lambda d: d["main_text"].str.strip())
    L["explanation_overlap"] = leakage.cross_split_exact(frames, lambda d: d["explanation"].str.strip())
    L["source_url_overlap"] = leakage.source_url_overlap(frames)
    nd = leakage.near_duplicate_pairs(frames, threshold=0.9)
    nd.to_csv(REPORTS / "near_duplicate_claims.csv", index=False)
    L["near_duplicate_claim_pairs_cos>=0.9"] = {
        f"{a}&{b}": int(n) for (a, b), n in nd.groupby(["split_a", "split_b"]).size().items()}
    test_ids_nd = set(nd.loc[nd.split_b == "test", "id_b"]) | set(nd.loc[nd.split_a == "test", "id_a"])
    L["test_examples_with_near_duplicate_elsewhere"] = len(test_ids_nd)
    lab = allv.set_index("example_id")["label"]
    if len(nd):
        L["near_duplicate_label_agreement"] = round(float((lab[nd.id_a].values == lab[nd.id_b].values).mean()), 3)
    L["verdict_phrases_in_main_text"] = {s: leakage.verdict_phrase_rates(frames[s]) for s in SPLITS}
    L["verdict_phrases_by_label"] = {l: leakage.verdict_phrase_rates(allv[allv.label == l]) for l in LABELS}
    exp_in = leakage.explanation_in_text(allv)
    lede = leakage.explanation_is_lede(allv)
    L["explanation_verbatim_in_main_text"] = round(float(exp_in.mean()), 4)
    L["explanation_is_lede_of_main_text"] = round(float(lede.mean()), 4)
    L["explanation_is_lede_by_label"] = allv.groupby("label").apply(
        lambda d: round(float(lede[d.index].mean()), 4), include_groups=False).to_dict()
    L["claim_verbatim_in_main_text"] = round(float(np.mean(
        [len(c) > 20 and c.lower() in t.lower() for c, t in zip(allv.claim, allv.main_text)])), 4)
    A["leakage"] = L

    # --- SciFact (kept separate; different label space) ---------------------------------
    sf = ROOT / "data" / "raw" / "scifact" / "data"
    SF = {}
    for s in ("train", "dev", "test"):
        claims = [json.loads(l) for l in open(sf / f"claims_{s}.jsonl", encoding="utf-8")]
        labs = Counter()
        for c in claims:
            if "evidence" not in c:
                labs["unlabelled"] += 1
            elif not c["evidence"]:
                labs["NEI"] += 1
            else:
                ls = {e["label"] for ev in c["evidence"].values() for e in ev}
                labs["SUPPORTS" if ls == {"SUPPORT"} else "REFUTES" if ls == {"CONTRADICT"} else "MIXED"] += 1
        SF[s] = {"claims": len(claims), "claim_level_labels": dict(labs)}
    SF["corpus_abstracts"] = sum(1 for _ in open(sf / "corpus.jsonl", encoding="utf-8"))
    A["scifact"] = SF

    (REPORTS / "dataset_audit.json").write_text(json.dumps(A, indent=2, default=int), encoding="utf-8")
    figures(frames, allv)
    print(json.dumps({s: A["splits"][s]["labels"] for s in SPLITS}, indent=1))


def figures(frames: dict[str, pd.DataFrame], allv: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(LABELS))
    for i, s in enumerate(SPLITS):
        c = frames[s]["label"].value_counts().reindex(LABELS, fill_value=0)
        ax.bar(x + (i - 1) * 0.27, c / c.sum() * 100, width=0.27, label=f"{s} (n={c.sum()})")
    ax.set_xticks(x, LABELS); ax.set_ylabel("% of split"); ax.set_title("PUBHEALTH label distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "label_distribution.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, f, cap in zip(axes, ["claim", "main_text", "explanation"], [80, 3000, 150]):
        w = words(allv[f]).clip(upper=cap)
        for l in LABELS:
            ax.hist(w[allv.label == l], bins=50, histtype="step", density=True, label=l)
        ax.set_title(f"{f} words (clipped at {cap})")
    axes[0].legend(); fig.tight_layout(); fig.savefig(FIG / "length_distributions.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    miss = pd.DataFrame({s: (frames[s][COLUMNS].apply(lambda c: c.str.strip() == "").mean() * 100) for s in SPLITS})
    miss.plot.barh(ax=ax); ax.set_xlabel("% empty"); ax.set_title("Missing values (valid rows)")
    fig.tight_layout(); fig.savefig(FIG / "missing_values.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fc = Counter(x for f in allv["fact_checkers"] for x in split_list(f)).most_common(20)
    sj = Counter(x.lower() for f in allv["subjects"] for x in split_list(f)).most_common(20)
    for ax, data, t in zip(axes, [fc, sj], ["Top fact-checkers", "Top subjects"]):
        ax.barh([d[0][:30] for d in data][::-1], [d[1] for d in data][::-1]); ax.set_title(t)
    fig.tight_layout(); fig.savefig(FIG / "factcheckers_subjects.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
