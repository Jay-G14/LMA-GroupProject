"""Reasoning capability metric (proposal Table 2): obligation-status accuracy against two human annotators.

Export is blind: for a label-stratified sample of claims, annotators see the claim, each obligation and the
abstracts the Auditor saw, never the model's status (kept in key.json). Each annotator labels the true
status of every obligation, and whether the obligations together cover the claim. Scoring reports Cohen's
kappa between the annotators, then the model's accuracy and macro-F1 against the gold labels (the
adjudicated label where given, else the annotators' shared label; disagreements without adjudication are
reported as pending, never guessed).
"""
from __future__ import annotations

import random

from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from ..ledger.schema import STATUSES

SHEET_FIELDS = ["item_id", "claim_id", "claim", "obligation_id", "type", "criticality", "description",
                "expected_condition", "audited_on", "annotator_A_status", "annotator_B_status",
                "adjudicated_status", "notes"]
CLAIM_FIELDS = ["claim_id", "claim", "obligations", "annotator_A_covers", "annotator_B_covers",
                "adjudicated_covers", "notes"]


def sample_records(records: list[dict], n: int, seed: int = 13) -> list[dict]:
    """Label-stratified (proportional) sample of ledger records with at least one obligation."""
    eligible = sorted((r for r in records if r.get("ledger", {}).get("obligations")), key=lambda r: r["claim_id"])
    by_label: dict[str, list[dict]] = {}
    for r in eligible:
        by_label.setdefault(r["gold_label"], []).append(r)
    rng, out = random.Random(seed), []
    labels, total = sorted(by_label), min(n, len(eligible))
    exact = {l: total * len(by_label[l]) / len(eligible) for l in labels}
    quotas = {l: int(exact[l]) for l in labels}                 # largest-remainder allocation: exactly `total`
    for l in sorted(labels, key=lambda l: (int(exact[l]) - exact[l], l))[: total - sum(quotas.values())]:
        quotas[l] += 1
    for l in labels:
        group = list(by_label[l])
        rng.shuffle(group)
        out += group[: quotas[l]]
    return sorted(out, key=lambda r: r["claim_id"])


def _final_audits(ledger: dict) -> dict[str, dict]:
    out = {}
    for a in ledger["audits"]:
        if a.get("kind", "audit") == "audit":
            out[a["obligation_id"]] = a
    return out


def build_items(record: dict, corpus: dict) -> tuple[list[dict], dict, dict, str]:
    """Sheet rows, the claim row, the hidden key and a Markdown packet section for one ledger record."""
    led, final = record["ledger"], _final_audits(record["ledger"])
    docs = list(dict.fromkeys(d for a in final.values() for d in a["shown_doc_ids"]))
    alias = {d: f"A{i + 1}" for i, d in enumerate(docs)}
    rows, key = [], {}
    for o in led["obligations"]:
        item = f"{record['claim_id']}:{o['obligation_id']}"
        a = final.get(o["obligation_id"])
        rows.append({"item_id": item, "claim_id": record["claim_id"], "claim": led["claim"],
                     "obligation_id": o["obligation_id"], "type": o["type"], "criticality": o["criticality"],
                     "description": o["description"], "expected_condition": o["expected_condition"],
                     "audited_on": " ".join(alias[d] for d in (a["shown_doc_ids"] if a else [])),
                     **{f: "" for f in SHEET_FIELDS[9:]}})
        key[item] = {"model_status": a["effective_status"] if a else None, "raw_status": a["status"] if a else None,
                     "criticality": o["criticality"],
                     "citations": [{"doc": alias.get(e["doc_id"], e["doc_id"]), "sentences": e["sentence_ids"]}
                                   for e in (a["evidence"] if a else [])]}
    claim_row = {"claim_id": record["claim_id"], "claim": led["claim"], "obligations": len(led["obligations"]),
                 **{f: "" for f in CLAIM_FIELDS[3:]}}
    md = [f"## Claim {record['claim_id']}", "", f"**Claim:** {led['claim']}", "", "**Obligations to label:**", ""]
    md += [f"- `{r['item_id']}` ({r['criticality']}, {r['type']}): {r['description']}. "
           f"Evidence must show: {r['expected_condition']}. Audited on: {r['audited_on'] or 'nothing'}" for r in rows]
    md += ["", "**Abstracts:**", ""]
    for d in docs:
        a = corpus[d]
        md += [f"[{alias[d]}] {a.title}"] + [f"  ({i}) {s}" for i, s in enumerate(a.sentences)] + [""]
    return rows, claim_row, key, "\n".join(md)


def _status(x: str) -> str | None:
    x = (x or "").strip().upper()
    if not x:
        return None
    if x not in STATUSES:
        raise ValueError(f"unknown status {x!r}; use one of {STATUSES}")
    return x


def _yes_no(x: str) -> str | None:
    x = (x or "").strip().lower()
    if not x:
        return None
    if x not in ("yes", "no"):
        raise ValueError(f"coverage must be yes or no, not {x!r}")
    return x


def _kappa(a: list, b: list, labels) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    if len(set(xs) | set(ys)) < 2:
        return 1.0 if xs == ys else 0.0
    return round(float(cohen_kappa_score(xs, ys, labels=list(labels))), 4)


def _gold(adjudicated, a, b):
    return adjudicated or (a if a and a == b else None)


def score(sheet: list[dict], claims: list[dict], key: dict) -> dict:
    a = [_status(r["annotator_A_status"]) for r in sheet]
    b = [_status(r["annotator_B_status"]) for r in sheet]
    gold = [_gold(_status(r["adjudicated_status"]), x, y) for r, x, y in zip(sheet, a, b)]
    model = [key[r["item_id"]]["model_status"] for r in sheet]
    scored = [(g, m, key[r["item_id"]]["criticality"]) for r, g, m in zip(sheet, gold, model) if g]
    out = {"n_items": len(sheet), "labelled_A": sum(map(bool, a)), "labelled_B": sum(map(bool, b)),
           "raw_agreement": round(sum(x == y for x, y in zip(a, b) if x and y) / max(1, sum(bool(x and y) for x, y in zip(a, b))), 4),
           "cohen_kappa": _kappa(a, b, STATUSES), "n_gold": len(scored),
           "pending_disagreements": sum(bool(x and y and x != y and not g) for x, y, g in zip(a, b, gold))}
    if scored:
        g_, m_, crit = zip(*scored)
        out.update({"model_status_accuracy": round(sum(x == y for x, y in zip(g_, m_)) / len(scored), 4),
                    "model_status_macro_f1": round(float(f1_score(          # over statuses that occur
                        g_, m_, labels=[x for x in STATUSES if x in set(g_) | set(m_)], average="macro",
                        zero_division=0)), 4),
                    "accuracy_by_criticality": {c: round(sum(x == y for x, y, k in scored if k == c) /
                                                         sum(k == c for *_, k in scored), 4)
                                                for c in sorted(set(crit))},
                    "confusion_matrix": {"rows_gold_cols_model": list(STATUSES),
                                         "matrix": confusion_matrix(g_, m_, labels=list(STATUSES)).tolist()}})
    ca = [_yes_no(r["annotator_A_covers"]) for r in claims]
    cb = [_yes_no(r["annotator_B_covers"]) for r in claims]
    cg = [_gold(_yes_no(r["adjudicated_covers"]), x, y) for r, x, y in zip(claims, ca, cb)]
    out["coverage"] = {"n_claims": len(claims), "cohen_kappa": _kappa(ca, cb, ("yes", "no")),
                       "n_gold": sum(map(bool, cg)),
                       "share_covering": round(sum(x == "yes" for x in cg) / max(1, sum(map(bool, cg))), 4)}
    return out
