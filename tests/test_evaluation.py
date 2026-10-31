"""Metrics added in protocol amendment 2 (2026-10-29): hand-checked examples for every metric group."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.scifact import LABELS, Abstract, Claim, load_claims  # noqa: E402
from oblig.evaluation.annotation import build_items, sample_records, score  # noqa: E402
from oblig.evaluation.classification import (_codes, _macro_f1_codes, bootstrap_ci, holm, macro_f1,  # noqa: E402
                                             mcnemar, paired_bootstrap)
from oblig.evaluation.scifact import (abstract_level, evaluate, evidence, iteration, minimal_pairs, robustness,  # noqa: E402
                                      safety, sentence_level)


def claim(i, label, ev=None, sets=None, cited=None, **kw):
    ev = ev or {}
    return Claim(i, f"claim {i}", label, ev, sets or {d: [s] for d, s in ev.items()},
                 {d: label for d in ev}, cited or list(ev), **kw)


# -- resampling statistics ---------------------------------------------------------------------
def test_fast_macro_f1_matches_sklearn():
    rng = random.Random(0)
    for _ in range(50):
        gold = [rng.choice(LABELS) for _ in range(40)]
        pred = [rng.choice(LABELS + ("__invalid__",)) for _ in range(40)]
        fast = _macro_f1_codes(_codes(gold, LABELS), _codes(pred, LABELS), len(LABELS))
        assert fast == pytest.approx(macro_f1(gold, pred, labels=LABELS))


def test_bootstrap_mcnemar_holm():
    gold = ["SUPPORTS", "REFUTES", "NEI"] * 20
    weak = ["NEI"] * 60
    ci = bootstrap_ci(gold, gold, labels=LABELS, n_boot=200)
    assert ci["value"] == 1.0 and ci["ci95"] == [1.0, 1.0]
    d = paired_bootstrap(gold, weak, gold, labels=LABELS, n_boot=500)
    assert d["delta"] == pytest.approx(1 - macro_f1(gold, weak, labels=LABELS), abs=1e-4)
    assert d["ci95"][0] > 0 and d["p_two_sided"] < 0.01
    m = mcnemar(gold, weak, gold)
    assert (m["only_a_correct"], m["only_b_correct"]) == (0, 40) and m["p_value"] < 1e-6
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}


# -- task, safety and SciFact rationale metrics ---------------------------------------------------
def test_safety_metrics():
    cs = [claim(1, "NEI"), claim(2, "NEI"), claim(3, "SUPPORTS", {"d": [0]}), claim(4, "REFUTES", {"d": [0]})]
    preds = {1: {"pred_label": "SUPPORTS", "rationale": {"d": [0]}}, 2: {"pred_label": "NEI"},
             3: {"pred_label": "REFUTES", "rationale": {}}, 4: {"pred_label": "__invalid__"}}
    s = safety(cs, preds)
    assert s["unsupported_verdict_rate"] == 0.5 and s["false_supports_on_nei"] == 0.5
    assert s["polarity_error_rate"] == 0.5                       # claim 3: SUPPORTS predicted REFUTES
    assert s["ungrounded_verdict_rate"] == 0.5 and s["invalid_rate"] == 0.25


def test_abstract_level_cap_uses_prediction_order():
    c = claim(1, "SUPPORTS", {"d": [5]}, {"d": [[5]]})
    late = abstract_level([c], {1: {"pred_label": "SUPPORTS", "rationale": {"d": [0, 1, 2, 5]}}})
    assert late["label_only"]["f1"] == 1.0 and late["label_rationale"]["f1"] == 0.0   # sentence 5 is 4th
    assert late["label_rationale_uncapped"]["f1"] == 1.0
    early = abstract_level([c], {1: {"pred_label": "SUPPORTS", "rationale": {"d": [5, 0, 1, 2]}}})
    assert early["label_rationale"]["f1"] == 1.0


def test_sentence_level_official_definition():
    c = claim(1, "SUPPORTS", {"d": [1, 2, 4]}, {"d": [[1, 2], [4]]})
    s = sentence_level([c], {1: {"pred_label": "SUPPORTS", "rationale": {"d": [1, 4]}}})
    # 4 completes the set {4}; 1 is in {1, 2} but 2 is missing, so it does not count
    assert (s["selection_label"]["tp"], s["selection_label"]["n_pred"], s["selection_label"]["n_gold"]) == (1, 2, 3)
    wrong = sentence_level([c], {1: {"pred_label": "REFUTES", "rationale": {"d": [1, 2, 4]}}})
    assert wrong["selection_label"]["tp"] == 0 and wrong["selection_only"]["tp"] == 3


def test_scifact_open_rationale_handling():
    c = Claim(1, "x", "SUPPORTS", {"cit": [2, 3], "pool": []}, {"cit": [[2, 3]], "pool": []},
              {"cit": "SUPPORTS", "pool": "SUPPORTS"}, [], dataset="scifact_open", rationale_docs=["cit"],
              merged_rationales=True)
    pred = {1: {"pred_label": "SUPPORTS", "rationale": {"cit": [2], "pool": [0, 1]}}}
    a = abstract_level([c], pred)
    assert a["label_rationale"] is None and a["label_only"]["tp"] == 2 and a["label_only"]["n_gold"] == 2
    s = sentence_level([c], pred)
    # pooled evidence has no gold sentences: its predictions are not scored; merged sets score per sentence
    assert (s["selection_label"]["tp"], s["selection_label"]["n_pred"], s["selection_label"]["n_gold"]) == (1, 1, 2)


# -- evidence, iteration, strata, robustness ---------------------------------------------------
def test_evidence_and_loop_gain():
    cs = [claim(1, "SUPPORTS", {"g1": [0]}), claim(2, "REFUTES", {"g2": [0]}), claim(3, "NEI")]
    calls = lambda n: [{"role": "x"}] * n
    preds = {
        1: {"pred_label": "SUPPORTS", "m1_label": "NEI", "retrieved": ["a", "g1"], "m1_retrieved": ["a"],
            "round0_calls": 3, "calls": calls(5), "entered_m2": True, "rounds_used": 1, "stop_reason": "x"},
        2: {"pred_label": "NEI", "m1_label": "REFUTES", "retrieved": ["g2", "b"], "m1_retrieved": ["g2"],
            "round0_calls": 3, "calls": calls(5), "entered_m2": True, "rounds_used": 1, "stop_reason": "x"},
        3: {"pred_label": "NEI", "m1_label": "NEI", "retrieved": ["c"], "m1_retrieved": ["c"],
            "round0_calls": 3, "calls": calls(3), "entered_m2": False, "rounds_used": 0, "stop_reason": "y"}}
    ev = evidence(cs, preds)
    assert ev["gold_abstract_recall"] == 1.0 and ev["all_gold_shown"] == 1.0 and ev["n_claims_with_evidence"] == 2
    it = iteration(cs, preds, n_boot=200)
    assert it["gold_abstract_recall_first_pass"] == 0.5 and it["loop_gain_recall"] == 0.5
    assert it["extra_calls_per_claim"] == pytest.approx(4 / 3, abs=1e-3)
    assert (it["changes_fixing_error"], it["changes_introducing_error"]) == (1, 1)
    assert it["loop_gain_macro_f1"]["delta"] == pytest.approx(
        macro_f1(["SUPPORTS", "REFUTES", "NEI"], ["SUPPORTS", "NEI", "NEI"], LABELS)
        - macro_f1(["SUPPORTS", "REFUTES", "NEI"], ["NEI", "REFUTES", "NEI"], LABELS), abs=1e-4)


def test_minimal_pairs_and_robustness():
    dev = load_claims("dev")
    assert len(minimal_pairs(dev)) == 31                           # matches the EDA
    perfect = {c.id: {"pred_label": c.label} for c in dev}
    r = robustness(dev, perfect)
    assert r["pair_accuracy"] == 1.0 and r["contrast_consistency"] == 1.0 and r["contrast_pairs"] == 15
    constant = {c.id: {"pred_label": "SUPPORTS"} for c in dev}
    assert robustness(dev, constant)["contrast_consistency"] == 0.0


def test_evaluate_runs_on_baseline_records():
    cs = [claim(1, "SUPPORTS", {"g": [0]}), claim(2, "NEI")]
    preds = {1: {"pred_label": "SUPPORTS", "rationale": {"g": [0]}, "retrieved": ["g"]},
             2: {"pred_label": "NEI", "rationale": {}, "retrieved": ["x"]}}
    m = evaluate(cs, preds, n_boot=100)
    assert m["claim_level"]["macro_f1"] == pytest.approx(macro_f1(["SUPPORTS", "NEI"], ["SUPPORTS", "NEI"], LABELS), abs=1e-4)
    assert "ledger" not in m and "iteration" not in m and m["abstract_level"]["label_rationale"]["f1"] == 1.0


# -- reasoning metric: blind annotation ---------------------------------------------------------
def record(cid, label, statuses):
    obs = [{"obligation_id": f"o{i + 1}", "description": f"c{i}", "type": "subject", "criticality": "core",
            "expected_condition": "x", "query": "q", "status": s} for i, s in enumerate(statuses)]
    audits = [{"obligation_id": f"o{i + 1}", "round": 0, "kind": "audit", "status": s, "effective_status": s,
               "evidence": [], "rationale": "", "needs_more_retrieval": False, "shown_doc_ids": ["d"]}
              for i, s in enumerate(statuses)]
    return {"claim_id": cid, "gold_label": label, "ledger": {"claim": f"claim {cid}", "obligations": obs, "audits": audits}}


def test_annotation_export_is_blind_and_stratified():
    recs = [record(i, ["SUPPORTS", "REFUTES", "NEI"][i % 3], ["SATISFIED", "PARTIAL"]) for i in range(30)]
    chosen = sample_records(recs, 9, seed=1)
    assert len(chosen) == 9 and sorted(r["gold_label"] for r in chosen).count("NEI") == 3
    uneven = recs[:7] + [record(100 + i, "NEI", ["SATISFIED"]) for i in range(4)]   # 3 S / 2 R / 6 NEI
    assert all(len(sample_records(uneven, k, seed=2)) == k for k in range(1, 12))    # exact size, no rounding loss
    rows, claim_row, key, md = build_items(recs[0], {"d": Abstract("d", "T", ["s0", "s1"], False)})
    assert len(rows) == 2 and all("SATISFIED" not in str(r.values()) for r in rows)       # no model answers
    assert key["0:o1"]["model_status"] == "SATISFIED" and "[A1] T" in md and claim_row["obligations"] == 2


def test_annotation_scoring():
    key = {f"1:o{i}": {"model_status": s, "criticality": "core"}
           for i, s in enumerate(["SATISFIED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"], 1)}
    sheet = [{"item_id": "1:o1", "annotator_A_status": "satisfied", "annotator_B_status": "SATISFIED", "adjudicated_status": ""},
             {"item_id": "1:o2", "annotator_A_status": "PARTIAL", "annotator_B_status": "UNSUPPORTED", "adjudicated_status": "UNSUPPORTED"},
             {"item_id": "1:o3", "annotator_A_status": "UNSUPPORTED", "annotator_B_status": "UNSUPPORTED", "adjudicated_status": ""},
             {"item_id": "1:o4", "annotator_A_status": "SATISFIED", "annotator_B_status": "CONTRADICTED", "adjudicated_status": ""}]
    claims = [{"annotator_A_covers": "yes", "annotator_B_covers": "yes", "adjudicated_covers": ""}]
    s = score(sheet, claims, key)
    assert s["n_gold"] == 3 and s["pending_disagreements"] == 1 and s["raw_agreement"] == 0.5
    assert s["model_status_accuracy"] == pytest.approx(2 / 3, abs=1e-4)     # o2 is PARTIAL vs gold UNSUPPORTED
    assert s["coverage"]["share_covering"] == 1.0
    with pytest.raises(ValueError):
        score([dict(sheet[0], annotator_A_status="MAYBE")], claims, key)
