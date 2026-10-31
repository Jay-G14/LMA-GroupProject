import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.agents.citations import alias_abstracts, resolve  # noqa: E402
from oblig.data.scifact import Abstract, Claim  # noqa: E402
from oblig.evaluation.scifact import abstract_level  # noqa: E402
from oblig.ledger.aggregation import POLICIES, ordered_sentences, pubhealth_v1, scifact_v1  # noqa: E402
from oblig.ledger.schema import (CONTRADICTED, PARTIAL, SATISFIED, UNSUPPORTED, Audit, EvidenceRef, Ledger,  # noqa: E402
                                 LedgerError, Obligation)
from oblig.llm.client import LLMClient, LLMConfig  # noqa: E402
from oblig.llm.mock_scifact import mock_scifact  # noqa: E402
from oblig.retrieval.bm25 import BM25  # noqa: E402
from oblig.systems.scifact import SYSTEMS, CallLog, Context, audit, build_ledger, plan, round_robin_pool, rrf_pool  # noqa: E402


def ob(i, t):
    return Obligation(f"o{i}", f"cond {i}", t, "x", f"q{i}")


def au(i, status, doc="d1", sents=(0,), rnd=0):
    ev = [EvidenceRef(doc, list(sents), "t")] if status in (SATISFIED, CONTRADICTED) else []
    return Audit(f"o{i}", rnd, status, status, ev, "r", status not in (SATISFIED, CONTRADICTED), [doc])


def ledger(types, audits):
    led = Ledger("c", "claim")
    for i, t in enumerate(types, 1):
        led.add_obligation(ob(i, t))
    for a in audits:
        led.add_audit(a)
    return led


# -- schema and state transitions ---------------------------------------------------------
def test_obligation_invariants():
    led = ledger(["subject"], [])
    with pytest.raises(LedgerError):
        led.add_obligation(ob(1, "outcome"))          # duplicate id
    with pytest.raises(LedgerError):
        led.add_obligation(ob(2, "dosage"))           # unknown type
    for i in range(2, 7):
        led.add_obligation(ob(i, "scope"))
    with pytest.raises(LedgerError):
        led.add_obligation(ob(7, "scope"))            # cap of 6
    assert led.obligation("o1").criticality == "core" and led.obligation("o2").criticality == "qualifier"


def test_transitions_preserve_history():
    led = ledger(["subject"], [au(1, UNSUPPORTED, rnd=0)])
    with pytest.raises(LedgerError):
        led.add_audit(au(1, PARTIAL, rnd=0))          # rounds must increase
    led.add_audit(au(1, PARTIAL, rnd=1))
    led.add_audit(au(1, SATISFIED, rnd=2))
    with pytest.raises(LedgerError):
        led.add_audit(au(1, CONTRADICTED, rnd=3))     # terminal
    assert [a.status for a in led.audits] == [UNSUPPORTED, PARTIAL, SATISFIED]
    assert led.status("o1") == SATISFIED and led.unresolved() == []


def test_resolving_status_requires_citation():
    led = ledger(["subject"], [])
    with pytest.raises(LedgerError):
        led.add_audit(Audit("o1", 0, SATISFIED, SATISFIED, [], "r", False, ["d1"]))


def test_serialisation_roundtrip():
    led = ledger(["subject", "outcome", "population"], [au(1, SATISFIED), au(2, PARTIAL), au(3, UNSUPPORTED)])
    d = json.loads(json.dumps(led.to_dict()))
    assert Ledger.from_dict(d).to_dict() == d
    assert [o["status"] for o in d["obligations"]] == [SATISFIED, PARTIAL, UNSUPPORTED]


# -- SciFact rule ---------------------------------------------------------------------------
def test_scifact_rule_edge_cases():
    assert scifact_v1(ledger(["population"], [au(1, SATISFIED)])).label == "NEI"           # no core
    assert scifact_v1(ledger(["subject", "outcome"], [au(1, SATISFIED), au(2, SATISFIED)])).label == "SUPPORTS"
    v = scifact_v1(ledger(["subject", "outcome"], [au(1, SATISFIED, "d1", [1]), au(2, SATISFIED, "d1", [3])]))
    assert v.rationale_docs == {"d1": [3, 1]}                         # outcome before subject (amendment 2)
    assert scifact_v1(ledger(["subject", "outcome"],
                             [au(1, SATISFIED, "d1"), au(2, SATISFIED, "d2")])).label == "NEI"  # two abstracts
    assert scifact_v1(ledger(["subject", "outcome"], [au(1, SATISFIED), au(2, PARTIAL)])).label == "NEI"
    assert scifact_v1(ledger(["subject", "outcome", "population"],
                             [au(1, SATISFIED), au(2, SATISFIED), au(3, UNSUPPORTED)])).label == "SUPPORTS"
    v = scifact_v1(ledger(["subject", "outcome", "population"],
                          [au(1, SATISFIED), au(2, SATISFIED), au(3, CONTRADICTED, "d9", [4])]))
    assert v.label == "REFUTES" and v.rationale_docs == {"d9": [4]}                    # any contradiction
    assert scifact_v1(ledger(["subject"], [])).label == "NEI"                          # unaudited


def test_scifact_anchored_rule():
    """Amendment 5: a contradiction counts only on an abstract also cited by a subject obligation."""
    anchored, anchored_sat = POLICIES["scifact_v1_anchored"], POLICIES["scifact_v1_anchoredsat"]
    off = ledger(["subject", "relation"], [au(1, SATISFIED, "d1"), au(2, CONTRADICTED, "d2")])
    assert scifact_v1(off).label == "REFUTES"
    v = anchored(off)
    assert v.label == "NEI" and v.reason == "a contradiction does not count under the rule"
    on = ledger(["subject", "relation", "population"],
                [au(1, SATISFIED, "d1"), au(2, CONTRADICTED, "d1", [5]), au(3, CONTRADICTED, "d9", [4])])
    assert anchored(on).label == "REFUTES" and anchored(on).rationale_docs == {"d1": [5]}   # unanchored d9 not cited
    # Off-subject qualifier contradictions do not block SUPPORTS; any subject status anchors; no subject: all count.
    assert anchored(ledger(["subject", "outcome", "scope"],
                           [au(1, SATISFIED), au(2, SATISFIED), au(3, CONTRADICTED, "d9")])).label == "SUPPORTS"
    weak = ledger(["subject", "relation"], [au(1, CONTRADICTED, "d1"), au(2, CONTRADICTED, "d1")])
    assert anchored(weak).label == "REFUTES" and anchored_sat(weak).label == "NEI"
    assert anchored(ledger(["atomic", "atomic"], [au(1, SATISFIED), au(2, CONTRADICTED, "d2")])).label == "REFUTES"


def test_pubhealth_candidate_rule():
    assert pubhealth_v1(ledger(["subject", "outcome"], [au(1, SATISFIED), au(2, SATISFIED)])).label == "true"
    assert pubhealth_v1(ledger(["subject", "outcome"], [au(1, SATISFIED), au(2, CONTRADICTED)])).label == "mixture"
    assert pubhealth_v1(ledger(["subject", "outcome"], [au(1, PARTIAL), au(2, CONTRADICTED)])).label == "mixture"
    assert pubhealth_v1(ledger(["subject", "outcome"], [au(1, UNSUPPORTED), au(2, CONTRADICTED)])).label == "false"
    assert pubhealth_v1(ledger(["subject", "outcome"], [au(1, SATISFIED), au(2, UNSUPPORTED)])).label == "unproven"
    assert pubhealth_v1(ledger(["scope"], [au(1, CONTRADICTED)])).label == "unproven"


# -- citations and auditor ------------------------------------------------------------------
CORPUS = {"10": Abstract("10", "Aspirin trial", ["Aspirin lowers stroke risk in adults.", "Bleeding rose."], False),
          "20": Abstract("20", "Statins", ["Statins lower cholesterol.", "No effect on stroke."], False),
          "30": Abstract("30", "Diet", ["Salt raises blood pressure."], False)}


def test_citation_resolution():
    _, alias = alias_abstracts(["10", "20"], CORPUS)
    refs, bad = resolve([{"doc": "A1", "sentences": [0, 7]}, {"doc": "A9", "sentences": [0]},
                         {"doc": "A2", "sentences": []}], alias, CORPUS)
    assert [(r.doc_id, r.sentence_ids) for r in refs] == [("10", [0])]
    assert refs[0].quoted_text == "Aspirin lowers stroke risk in adults."
    assert {b["reason"] for b in bad} == {"sentence out of range", "unknown document", "no sentences"}


def mk_ctx(fn=mock_scifact, budget=2):
    bm = BM25().fit([a.title + " " + " ".join(a.sentences) for a in CORPUS.values()], list(CORPUS))
    return Context(LLMClient(LLMConfig(backend="mock", extra={"mock_fn": fn})), bm, CORPUS, budget=budget)


def test_auditor_downgrades_uncited_resolution():
    ctx = mk_ctx(lambda m: json.dumps({"status": "SATISFIED", "citations": [{"doc": "A7", "sentences": [0]}],
                                       "rationale": "r", "needs_more_retrieval": False}))
    led = ledger(["subject"], [])
    a = audit(ctx, led, led.obligation("o1"), ["10"], 0, CallLog())
    assert a.status == SATISFIED and a.effective_status == UNSUPPORTED and a.invalid_citations
    led.add_audit(a)                                   # accepted: it does not resolve the obligation
    assert led.unresolved()


def test_planner_failure_and_dedup_are_recorded():
    claim = Claim(1, "Aspirin lowers stroke risk.", "SUPPORTS")
    bad = plan(mk_ctx(lambda m: "nope"), claim, CallLog())
    assert not bad.obligations and bad.planner_error.startswith("parse_error")
    dup = {"description": "aspirin", "type": "subject", "expected_condition": "x", "query": "aspirin"}
    led = plan(mk_ctx(lambda m: json.dumps({"obligations": [dup, dict(dup, description="Aspirin!")]})), claim,
               CallLog())
    assert len(led.obligations) == 1 and "duplicate" in led.planner_error


def test_round_robin_pool():
    assert round_robin_pool([["a", "b", "c"], ["a", "d"], ["e"]], 4) == ["a", "e", "b", "d"]
    assert round_robin_pool([["a"]], 5) == ["a"]


def test_every_system_respects_budget_and_cites_shown_evidence():
    ctx = mk_ctx(budget=2)
    claim = Claim(1, "Aspirin lowers stroke risk in adults.", "SUPPORTS", {"10": [0]}, {"10": [[0]]},
                  {"10": "SUPPORTS"})
    for name, fn in SYSTEMS.items():
        r = fn(ctx, claim)
        assert len(r["retrieved"]) <= 2, name
        assert set(r["rationale"]) <= set(r["retrieved"]), name
        if "ledger" in r:
            Ledger.from_dict(r["ledger"])
            assert all(set(a["shown_doc_ids"]) <= set(r["retrieved"]) for a in r["ledger"]["audits"])
    assert SYSTEMS["m1"](ctx, claim)["pred_label"] == "SUPPORTS"


def test_abstract_level_metric():
    c = Claim(1, "x", "SUPPORTS", {"10": [0, 1]}, {"10": [[0], [1]]}, {"10": "SUPPORTS"})
    good = abstract_level([c], {1: {"pred_label": "SUPPORTS", "rationale": {"10": [1]}}})
    assert good["label_rationale"]["f1"] == 1.0
    wrong_doc = abstract_level([c], {1: {"pred_label": "SUPPORTS", "rationale": {"20": [0]}}})
    assert wrong_doc["label_only"]["precision"] == 0.0 and wrong_doc["label_only"]["n_gold"] == 1
    no_rat = abstract_level([c], {1: {"pred_label": "SUPPORTS", "rationale": {"10": [5]}}})
    assert no_rat["label_only"]["f1"] == 1.0 and no_rat["label_rationale"]["f1"] == 0.0


def test_consolidation_invariants():
    def cons(i, status, doc, rnd=1):
        ev = [EvidenceRef(doc, [0], "t")] if status == SATISFIED else []
        return Audit(f"o{i}", rnd, status, status, ev, "r", False, [doc], kind="consolidation")
    led = ledger(["subject", "outcome", "population"], [au(1, SATISFIED, "d1"), au(2, UNSUPPORTED), au(3, SATISFIED)])
    with pytest.raises(LedgerError):
        led.add_audit(cons(2, SATISFIED, "d2"))                 # target must be SATISFIED
    with pytest.raises(LedgerError):
        led.add_audit(cons(3, SATISFIED, "d2"))                 # qualifiers are never consolidated
    with pytest.raises(LedgerError):
        led.add_audit(Audit("o1", 1, SATISFIED, SATISFIED, [EvidenceRef("d2", [0], "t")], "r", False,
                            ["d2", "d3"], kind="consolidation"))  # exactly one abstract
    led.add_audit(cons(1, SATISFIED, "d2"))
    with pytest.raises(LedgerError):
        led.add_audit(cons(1, SATISFIED, "d2", rnd=2))          # same abstract twice
    assert led.status("o1") == SATISFIED and [e.doc_id for e in led.evidence("o1")] == ["d1", "d2"]
    led.add_audit(cons(1, CONTRADICTED, "d3"))                  # recorded, adds nothing, status unchanged
    assert led.status("o1") == SATISFIED and [e.doc_id for e in led.evidence("o1")] == ["d1", "d2"]
    d = json.loads(json.dumps(led.to_dict()))
    assert Ledger.from_dict(d).to_dict() == d


def test_scifact_rule_uses_consolidated_evidence():
    led = ledger(["subject", "outcome"], [au(1, SATISFIED, "d1", [2]), au(2, SATISFIED, "d2", [5])])
    assert scifact_v1(led).label == "NEI"
    led.add_audit(Audit("o1", 1, SATISFIED, SATISFIED, [EvidenceRef("d2", [1], "t")], "r", False, ["d2"],
                        kind="consolidation"))
    v = scifact_v1(led)
    assert v.label == "SUPPORTS" and v.rationale_docs == {"d2": [5, 1]}



# -- protocol amendment 2 (2026-10-29) --------------------------------------------------------------
def test_citations_keep_model_order():
    _, alias = alias_abstracts(["10", "20"], CORPUS)
    refs, _ = resolve([{"doc": "A1", "sentences": [1, 0, 1]}], alias, CORPUS)
    assert refs[0].sentence_ids == [1, 0] and refs[0].quoted_text.startswith("Bleeding rose.")


def test_rationale_order_puts_decisive_sentences_first():
    led = ledger(["subject", "outcome", "relation"],
                 [au(1, SATISFIED, "d1", [0, 4]), au(2, SATISFIED, "d1", [2, 4]), au(3, SATISFIED, "d1", [7])])
    # sentence 4 is cited by two obligations; then relation (7), outcome (2), subject (0)
    assert ordered_sentences(led, led.obligations, "d1") == [4, 7, 2, 0]
    assert scifact_v1(led).rationale_docs == {"d1": [4, 7, 2, 0]}


def test_rrf_pool_fuses_rankings():
    assert rrf_pool([["a", "b", "c"], ["b", "d"], ["b", "e"]], 2) == ["b", "a"]
    assert rrf_pool([["a"]], 5) == ["a"] and rrf_pool([], 3) == []


def test_m1_audits_every_obligation_on_the_whole_pool():
    """Proposal: per-obligation rankings are fused into one pool of K abstracts and every obligation
    is audited on all of them; M1-aligned keeps the original per-obligation subsets as an ablation."""
    claim = Claim(1, "Aspirin lowers stroke risk; statins lower cholesterol.", "SUPPORTS")
    fn = lambda m: (json.dumps({"obligations": [
        {"description": "aspirin stroke", "type": "subject", "expected_condition": "x", "query": "aspirin stroke"},
        {"description": "statins cholesterol", "type": "outcome", "expected_condition": "x", "query": "statins cholesterol"}]})
        if m[0]["content"].startswith("ROLE: planner") else json.dumps(
        {"status": "UNSUPPORTED", "citations": [], "rationale": "", "needs_more_retrieval": True}))
    led, pool, per_ob = build_ledger(mk_ctx(fn, budget=2), claim, CallLog())
    assert len(pool) == 2 and all(per_ob[o.obligation_id] == pool for o in led.obligations)
    assert all(a.shown_doc_ids == pool for a in led.audits)
    led_a, pool_a, per_a = build_ledger(mk_ctx(fn, budget=2), claim, CallLog(), evidence="aligned")
    assert all(set(v) <= set(pool_a) for v in per_a.values())
    led_s, pool_s, _ = build_ledger(mk_ctx(fn, budget=2), claim, CallLog(), evidence="shared")
    assert pool_s == [d for d, _ in mk_ctx(fn, budget=2).retriever.search(claim.claim, k=2)]
    with pytest.raises(ValueError):
        build_ledger(mk_ctx(fn), claim, CallLog(), evidence="nope")
