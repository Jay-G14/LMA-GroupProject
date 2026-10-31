"""SciFact evaluation: every metric in the proposal plus diagnostics (protocol amendment 2, 2026-10-29).

Metric groups, each computed from one prediction file and the gold claims:

  claim_level     primary task metric: macro-F1 over SUPPORTS/REFUTES/NEI with a 95% bootstrap CI; accuracy,
                  per-class P/R/F1, confusion matrix.
  safety          proposal's safety metric: unsupported-verdict rate (gold-NEI claims predicted SUPPORTS or
                  REFUTES); also polarity errors, verdicts without a valid citation, invalid outputs.
  abstract_level  SciFact's official label+rationale P/R/F1 (only the first 3 predicted sentences of an
                  abstract count); label-only (SciFact paper); uncapped label+rationale (diagnostic).
  sentence_level  SciFact's official selection+label P/R/F1 (all predicted sentences count; a sentence is
                  correct only if its whole gold rationale set is predicted); selection-only. The RAG
                  capability metric is sentence_level.selection_label.f1.
  evidence        gold-abstract recall of the abstracts shown to the verdict step, share of claims with every
                  gold abstract shown, share of gold rationale sentences shown.
  ledger          obligation counts, types and statuses, planner failures, verdict reasons, and how often
                  resolving audits cite gold evidence abstracts and gold rationale sentences.
  iteration       agentic capability metric, loop gain: change in macro-F1 and in gold-abstract recall from the
                  first pass (M1 / B1) to the iterated result (M2 / M2-claimiter), per extra LLM call; verdict
                  flips, status transitions, stop reasons.
  strata          accuracy split by whether a gold abstract was shown and by number of obligations.
  robustness      minimal pairs: near-identical claims sharing an abstract (pair accuracy, contrast consistency).

The reasoning capability metric (obligation-status accuracy against two annotators) needs human labels:
scripts/export_annotation.py and scripts/score_annotation.py.

SciFact-Open merges each abstract's rationale sets and has gold rationales only for citation evidence, so
its label+rationale metric is not computed and its sentence metrics score citation evidence per sentence.
"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

from ..data.scifact import LABELS, Claim
from .classification import INVALID, bootstrap_ci, classification_report, macro_f1, paired_bootstrap

MAX_RATIONALE = 3
VERDICTS = ("SUPPORTS", "REFUTES")
TERMINAL = ("SATISFIED", "CONTRADICTED")
PAIR_SIMILARITY = 0.75


def _f1(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def _rationale(p: dict) -> dict[str, list[int]]:
    """Predicted rationale of a SUPPORTS/REFUTES prediction, sentence order kept (NEI predicts none)."""
    if p["pred_label"] not in VERDICTS:
        return {}
    return {d: list(dict.fromkeys(s)) for d, s in (p.get("rationale") or {}).items()}


# -- task --------------------------------------------------------------------------------
def claim_level(claims: list[Claim], preds: dict[int, dict], n_boot: int = 10_000) -> dict:
    gold = [c.label for c in claims]
    pred = [preds[c.id]["pred_label"] for c in claims]
    rep = classification_report(gold, pred, labels=LABELS)
    rep["macro_f1_ci95"] = bootstrap_ci(gold, pred, labels=LABELS, n_boot=n_boot)["ci95"]
    return rep


def safety(claims: list[Claim], preds: dict[int, dict]) -> dict:
    lab = {c.id: preds[c.id]["pred_label"] for c in claims}
    nei = [c for c in claims if c.label == "NEI"]
    sr = [c for c in claims if c.label in VERDICTS]
    verdicts = [c for c in claims if lab[c.id] in VERDICTS]
    return {"unsupported_verdict_rate": _rate(sum(lab[c.id] in VERDICTS for c in nei), len(nei)),
            "false_supports_on_nei": _rate(sum(lab[c.id] == "SUPPORTS" for c in nei), len(nei)),
            "false_refutes_on_nei": _rate(sum(lab[c.id] == "REFUTES" for c in nei), len(nei)),
            "polarity_error_rate": _rate(sum(lab[c.id] in VERDICTS and lab[c.id] != c.label for c in sr), len(sr)),
            "ungrounded_verdict_rate": _rate(sum(not _rationale(preds[c.id]) for c in verdicts), len(verdicts)),
            "invalid_rate": _rate(sum(lab[c.id] not in LABELS for c in claims), len(claims)),
            "n_gold_nei": len(nei), "n_verdicts": len(verdicts)}


# -- SciFact official rationale metrics ----------------------------------------------------------
def abstract_level(claims: list[Claim], preds: dict[int, dict]) -> dict:
    exact = not any(c.merged_rationales for c in claims)
    tp_label = tp_rat = tp_unc = n_pred = n_gold = 0
    for c in claims:
        n_gold += len(c.evidence)
        p = preds[c.id]
        for doc, sents in _rationale(p).items():
            n_pred += 1
            if doc in c.evidence and c.doc_labels[doc] == p["pred_label"]:
                tp_label += 1
                if exact:
                    tp_rat += any(set(g) <= set(sents[:MAX_RATIONALE]) for g in c.evidence_sets[doc])
                    tp_unc += any(set(g) <= set(sents) for g in c.evidence_sets[doc])
    out = {"label_only": _f1(tp_label, n_pred, n_gold)}
    if exact:
        out["label_rationale"] = _f1(tp_rat, n_pred, n_gold)
        out["label_rationale_uncapped"] = _f1(tp_unc, n_pred, n_gold)
    else:
        out["label_rationale"] = None
        out["note"] = "rationale sets are merged in this test set; only label-only is scored"
    return out


def sentence_level(claims: list[Claim], preds: dict[int, dict]) -> dict:
    merged = any(c.merged_rationales for c in claims)
    tp_sel = tp_lab = n_pred = n_gold = 0
    for c in claims:
        n_gold += sum(len(c.evidence[d]) for d in c.evidence if c.has_gold_rationale(d))
        p = preds[c.id]
        for doc, sents in _rationale(p).items():
            if doc in c.evidence and not c.has_gold_rationale(doc):
                continue                      # labelled evidence without gold sentences: unscorable
            n_pred += len(sents)
            if doc not in c.evidence:
                continue
            sets = [[s] for s in c.evidence[doc]] if c.merged_rationales else c.evidence_sets[doc]
            correct = {s for g in sets if set(g) <= set(sents) for s in g}
            tp_sel += len(correct)
            tp_lab += len(correct) if c.doc_labels[doc] == p["pred_label"] else 0
    return {"selection_label": _f1(tp_lab, n_pred, n_gold), "selection_only": _f1(tp_sel, n_pred, n_gold),
            "criterion": ("per sentence; citation evidence only (merged rationale sets)" if merged
                          else "official: complete gold rationale sets")}


# -- retrieval as seen by the verdict step -----------------------------------------------------
def _gold_recall(c: Claim, docs) -> float:
    return len(set(c.evidence) & set(docs)) / len(c.evidence)


def evidence(claims: list[Claim], preds: dict[int, dict], key: str = "retrieved") -> dict:
    with_ev = [c for c in claims if c.evidence]
    with_sents = [c for c in with_ev if any(c.has_gold_rationale(d) and c.evidence[d] for d in c.evidence)]

    def coverage(c):
        shown = set(preds[c.id].get(key) or [])
        gold = {d: c.evidence[d] for d in c.evidence if c.has_gold_rationale(d)}
        return sum(len(s) for d, s in gold.items() if d in shown) / sum(len(s) for s in gold.values())
    return {"n_claims_with_evidence": len(with_ev),
            "gold_abstract_recall": _mean(_gold_recall(c, preds[c.id].get(key) or []) for c in with_ev),
            "all_gold_shown": _mean(set(c.evidence) <= set(preds[c.id].get(key) or []) for c in with_ev),
            "rationale_sentence_coverage": _mean(coverage(c) for c in with_sents),
            "abstracts_shown_mean": _mean(len(preds[c.id].get(key) or []) for c in claims)}


# -- ledger diagnostics ---------------------------------------------------------------------
def _final_audits(ledger: dict) -> dict[str, dict]:
    out = {}
    for a in ledger["audits"]:
        if a.get("kind", "audit") == "audit":
            out[a["obligation_id"]] = a
    return out


def ledger_metrics(claims: list[Claim], preds: dict[int, dict]) -> dict | None:
    recs = [(c, preds[c.id]) for c in claims if "ledger" in preds[c.id]]
    if not recs:
        return None
    n_obs = [len(r["ledger"]["obligations"]) for _, r in recs]
    obs = [o for _, r in recs for o in r["ledger"]["obligations"]]
    refs = gold_doc_refs = sents_on_gold = gold_sents = 0
    for c, r in recs:
        for a in _final_audits(r["ledger"]).values():
            if a["effective_status"] not in TERMINAL:
                continue
            for e in a["evidence"]:
                refs += 1
                if e["doc_id"] in c.evidence:
                    gold_doc_refs += 1
                    if c.has_gold_rationale(e["doc_id"]):
                        sents_on_gold += len(e["sentence_ids"])
                        gold_sents += len(set(e["sentence_ids"]) & set(c.evidence[e["doc_id"]]))
    return {"obligations_per_claim": _mean(n_obs), "obligation_count_distribution": dict(sorted(Counter(n_obs).items())),
            "planner_failure_rate": _rate(sum(n == 0 for n in n_obs), len(recs)),
            "duplicates_dropped": sum("duplicate" in (r["ledger"].get("planner_error") or "") for _, r in recs),
            "type_counts": dict(Counter(o["type"] for o in obs)),
            "status_counts_core": dict(Counter(o["status"] for o in obs if o["criticality"] == "core")),
            "status_counts_qualifier": dict(Counter(o["status"] for o in obs if o["criticality"] == "qualifier")),
            "verdict_reasons": dict(Counter(r.get("verdict_reason") for _, r in recs)),
            "resolving_citations_on_gold_abstract": _rate(gold_doc_refs, refs),
            "resolving_cited_sentences_in_gold_rationale": _rate(gold_sents, sents_on_gold)}


# -- agentic loop -------------------------------------------------------------------------------
def iteration(claims: list[Claim], preds: dict[int, dict], n_boot: int = 10_000) -> dict | None:
    recs = [(c, preds[c.id]) for c in claims if "m1_label" in preds[c.id] or "first_label" in preds[c.id]]
    if not recs:
        return None
    gold = [c.label for c, _ in recs]
    before = [r.get("m1_label", r.get("first_label")) for _, r in recs]
    after = [r["pred_label"] for _, r in recs]
    gain = paired_bootstrap(gold, before, after, labels=LABELS, n_boot=n_boot)
    extra = _mean(len(r["calls"]) - r.get("round0_calls", len(r["calls"])) for _, r in recs) or 0.0
    with_ev = [(c, r) for c, r in recs if c.evidence]
    rec_before = _mean(_gold_recall(c, r.get("m1_retrieved") or r["retrieved"]) for c, r in with_ev)
    rec_after = _mean(_gold_recall(c, r["retrieved"]) for c, r in with_ev)
    d_recall = round(rec_after - rec_before, 4) if with_ev else None
    transitions = Counter()
    for _, r in recs:
        if "ledger" not in r:
            continue
        first = {}
        for a in r["ledger"]["audits"]:
            if a.get("kind", "audit") == "audit" and a["round"] == 0:
                first[a["obligation_id"]] = a["effective_status"]
        for oid, a in _final_audits(r["ledger"]).items():
            transitions[f"{first.get(oid)}->{a['effective_status']}"] += 1
    changed = [b != a for b, a in zip(before, after)]
    return {"n": len(recs),
            "share_entering_iteration": _rate(sum(bool(r.get("entered_m2", r.get("rounds"))) for _, r in recs), len(recs)),
            "rounds_used_mean": _mean(r.get("rounds_used", len(r.get("rounds", []))) for _, r in recs),
            "extra_calls_per_claim": round(extra, 3),
            "macro_f1_first_pass": round(_macro(gold, before), 4),
            "macro_f1_iterated": round(_macro(gold, after), 4),
            "loop_gain_macro_f1": gain,
            "gold_abstract_recall_first_pass": rec_before, "gold_abstract_recall_iterated": rec_after,
            "loop_gain_recall": d_recall,
            "loop_gain_macro_f1_per_extra_call": round(gain["delta"] / extra, 4) if extra else None,
            "loop_gain_recall_per_extra_call": round(d_recall / extra, 4) if extra and d_recall is not None else None,
            "verdicts_changed": sum(changed),
            "changes_fixing_error": sum(ch and a == g for ch, a, g in zip(changed, after, gold)),
            "changes_introducing_error": sum(ch and b == g for ch, b, g in zip(changed, before, gold)),
            "new_verdicts_on_gold_nei": sum(ch and g == "NEI" and a in VERDICTS
                                            for ch, a, g in zip(changed, after, gold)),
            "consolidation_calls": sum(x["role"] == "consolidation" for _, r in recs for x in r["calls"]),
            "status_transitions": dict(transitions),
            "stop_reasons": dict(Counter(r.get("stop_reason") for _, r in recs))}


def _macro(gold, pred) -> float:
    return macro_f1(gold, [p if p in LABELS else INVALID for p in pred], labels=LABELS)


# -- strata and robustness ----------------------------------------------------------------------
def strata(claims: list[Claim], preds: dict[int, dict]) -> dict:
    def acc(cs):
        return {"n": len(cs), "accuracy": _rate(sum(preds[c.id]["pred_label"] == c.label for c in cs), len(cs))}
    with_ev = [c for c in claims if c.evidence]
    shown = [c for c in with_ev if set(c.evidence) & set(preds[c.id].get("retrieved") or [])]
    out = {"gold_abstract_shown": acc(shown), "gold_abstract_missed": acc([c for c in with_ev if c not in shown]),
           "gold_nei": acc([c for c in claims if not c.evidence])}
    if any("n_obligations" in preds[c.id] for c in claims):
        bucket = lambda n: "0" if n == 0 else "1-2" if n <= 2 else str(n) if n <= 4 else "5-6"
        groups: dict[str, list] = {}
        for c in claims:
            groups.setdefault(bucket(preds[c.id].get("n_obligations", 0)), []).append(c)
        out["by_obligations"] = {k: acc(v) for k, v in sorted(groups.items())}
    return out


def minimal_pairs(claims: list[Claim], threshold: float = PAIR_SIMILARITY) -> list[tuple[Claim, Claim]]:
    """Pairs of claims that cite a common abstract and whose texts are near-identical (often one word
    flipped, e.g. 'increases' vs 'decreases'). Similarity: max of both difflib argument orders."""
    by_doc: dict[str, list[Claim]] = {}
    for c in claims:
        for d in c.cited_doc_ids:
            by_doc.setdefault(d, []).append(c)
    seen, pairs = set(), []
    for group in by_doc.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                key = tuple(sorted((a.id, b.id)))
                if key in seen:
                    continue
                seen.add(key)
                x, y = a.claim.lower(), b.claim.lower()
                if max(SequenceMatcher(None, x, y).ratio(), SequenceMatcher(None, y, x).ratio()) >= threshold:
                    pairs.append((a, b))
    return pairs


def robustness(claims: list[Claim], preds: dict[int, dict]) -> dict:
    pairs = minimal_pairs(claims)
    lab = lambda c: preds[c.id]["pred_label"]
    contrast = [(a, b) for a, b in pairs if a.label != b.label]
    return {"minimal_pairs": len(pairs),
            "pair_accuracy": _rate(sum(lab(a) == a.label and lab(b) == b.label for a, b in pairs), len(pairs)),
            "contrast_pairs": len(contrast),
            "contrast_consistency": _rate(sum(lab(a) != lab(b) for a, b in contrast), len(contrast))}


def evaluate(claims: list[Claim], preds: dict[int, dict], n_boot: int = 10_000) -> dict:
    out = {"claim_level": claim_level(claims, preds, n_boot), "safety": safety(claims, preds),
           "abstract_level": abstract_level(claims, preds), "sentence_level": sentence_level(claims, preds),
           "evidence": evidence(claims, preds), "strata": strata(claims, preds),
           "robustness": robustness(claims, preds)}
    if (led := ledger_metrics(claims, preds)) is not None:
        out["ledger"] = led
    if (it := iteration(claims, preds, n_boot)) is not None:
        out["iteration"] = it
    return out
