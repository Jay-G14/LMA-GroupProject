import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.scifact import Abstract, Claim  # noqa: E402
from oblig.ledger.schema import UNTYPED, Ledger  # noqa: E402
from oblig.llm.client import LLMClient, LLMConfig  # noqa: E402
from oblig.retrieval.bm25 import BM25  # noqa: E402
from oblig.systems.scifact import SYSTEMS, Context  # noqa: E402

# Doc "k" holds the key fact but only ranks for the follow-up query "zebrafish".
CORPUS = {str(i): Abstract(str(i), f"Paper {i}", [f"Aspirin study {i} reports stroke data.", "Other text."], False)
          for i in range(10)}
CORPUS["k"] = Abstract("k", "Zebrafish", ["Zebrafish trial confirmed the outcome."], False)
CLAIM = Claim(1, "Aspirin has the magicword effect.", "SUPPORTS", {"k": [0]}, {"k": [[0]]}, {"k": "SUPPORTS"})


def scripted(requery="zebrafish", fail_requery=False, verdict_until_magic=True):
    def fn(messages):
        role = messages[0]["content"].split("\n", 1)[0].removeprefix("ROLE: ")
        user = messages[-1]["content"]
        if role == "planner":
            return json.dumps({"obligations": [
                {"description": "aspirin", "type": "subject", "expected_condition": "x", "query": "aspirin"},
                {"description": "magicword effect", "type": "outcome", "expected_condition": "x", "query": "effect"}]})
        if role == "facts":
            return json.dumps({"facts": [{"fact": "aspirin magicword", "query": "aspirin"}]})
        if role in ("requery", "claim_requery"):
            return "broken" if fail_requery else json.dumps({"missing": "m", "query": requery})
        aliases = re.findall(r"^\[(A\d+)\] (.*)$", user, flags=re.MULTILINE)
        magic = next((a for a, t in aliases if t == "Zebrafish"), None)
        if role == "auditor":
            if "aspirin" == re.search(r"Condition to check: (.*)", user).group(1):
                return json.dumps({"status": "SATISFIED", "citations": [{"doc": "A1", "sentences": [0]}],
                                   "rationale": "", "needs_more_retrieval": False})
            if magic:
                return json.dumps({"status": "SATISFIED", "citations": [{"doc": magic, "sentences": [0]}],
                                   "rationale": "", "needs_more_retrieval": False})
            return json.dumps({"status": "UNSUPPORTED", "citations": [], "rationale": "no", "needs_more_retrieval": True})
        if role == "aggregate":
            return json.dumps({"verdict": "REFUTES", "justification": "llm"})
        if magic or not verdict_until_magic:
            return json.dumps({"verdict": "SUPPORTS", "citations": [{"doc": magic or "A1", "sentences": [0]}],
                               "justification": ""})
        return json.dumps({"verdict": "NEI", "citations": [], "justification": ""})
    return fn


def ctx_for(fn, budget=1, corpus=CORPUS):
    bm = BM25().fit([a.title + " " + " ".join(a.sentences) for a in corpus.values()], list(corpus))
    return Context(LLMClient(LLMConfig(backend="mock", extra={"mock_fn": fn})), bm, corpus, budget=budget)


def test_m2_targets_unresolved_and_preserves_history():
    r = SYSTEMS["m2"](ctx_for(scripted()), CLAIM)
    led = Ledger.from_dict(r["ledger"])
    hist = [(a.obligation_id, a.round, a.effective_status) for a in led.audits]
    assert ("o2", 0, "UNSUPPORTED") in hist and ("o2", 1, "SATISFIED") in hist
    assert [a.round for a in led.audits if a.obligation_id == "o1" and a.kind == "audit"] == [0]   # never re-audited
    assert led.retrieval_rounds[0]["targets"] == ["o2"] and "k" in led.retrieval_rounds[0]["new_doc_ids"]
    assert r["stop_reason"] == "all obligations resolved"
    assert len(r["retrieved"]) == len(set(r["retrieved"])) <= 1 + 2 * 2
    roles = [c["role"] for c in r["calls"]]
    assert roles.count("requery") == 1 and roles.count("auditor") == 3
    # Consolidation (amendment 2026-10-27): o1, satisfied on an old abstract, is re-checked on doc k
    # alone; it holds there too, so the single-abstract SUPPORTS rule now fires on k.
    assert roles.count("consolidation") == 1
    assert led.retrieval_rounds[0]["consolidations"] == [{"obligation_id": "o1", "doc_id": "k", "status": "SATISFIED"}]
    assert led.status("o1") == "SATISFIED" and {e.doc_id for e in led.evidence("o1")} >= {"k"}
    assert r["m1_label"] == "NEI" and r["pred_label"] == "SUPPORTS" and r["verdict_changed"]
    assert r["rationale"] == {"k": [0]}
    # first-pass record for the loop-gain metric: M1's pool and calls before iteration
    assert "k" not in r["m1_retrieved"] and set(r["m1_retrieved"]) < set(r["retrieved"])
    assert r["round0_calls"] == 1 + len(led.obligations)


def test_m2_without_consolidation_keeps_the_asymmetry():
    ctx = ctx_for(scripted())
    ctx.m2_consolidate = False
    r = SYSTEMS["m2"](ctx, CLAIM)
    assert r["pred_label"] == "NEI" and not r["verdict_changed"]
    assert r["verdict_reason"] == "core obligations satisfied only across different abstracts"
    assert not any(c["role"] == "consolidation" for c in r["calls"])


def test_failed_consolidation_adds_no_evidence_and_never_changes_status():
    base = scripted()

    def fn(messages):                       # o1 does not hold on the zebrafish abstract
        user = messages[-1]["content"]
        if messages[0]["content"].startswith("ROLE: auditor") and "Condition to check: aspirin" in user                 and "[A1] Zebrafish" in user and "[A2]" not in user:
            return json.dumps({"status": "CONTRADICTED", "citations": [{"doc": "A1", "sentences": [0]}],
                               "rationale": "", "needs_more_retrieval": False})
        return base(messages)
    r = SYSTEMS["m2"](ctx_for(fn), CLAIM)
    led = Ledger.from_dict(r["ledger"])
    cons = [a for a in led.audits if a.kind == "consolidation"]
    assert [(a.obligation_id, a.effective_status) for a in cons] == [("o1", "CONTRADICTED")]
    assert led.status("o1") == "SATISFIED" and "k" not in {e.doc_id for e in led.evidence("o1")}
    assert r["pred_label"] == "NEI"         # a consolidation never turns into REFUTES


def test_m2_caps_and_stop_on_round_limit():
    r = SYSTEMS["m2"](ctx_for(scripted(requery="study")), CLAIM)      # follow-ups never find doc k
    led = Ledger.from_dict(r["ledger"])
    fetched = [d for x in led.retrieval_rounds for d in x["new_doc_ids"]]
    assert r["rounds_used"] == 2 and all(len(x["new_doc_ids"]) <= 2 for x in led.retrieval_rounds)
    assert len(fetched) == len(set(fetched)) and not set(fetched) & set(led.audits[0].shown_doc_ids)
    assert r["stop_reason"] == "round cap reached" and max(a.round for a in led.audits) == 2


def test_m2_stops_when_corpus_exhausted_and_falls_back_on_failed_requery():
    small = {k: CORPUS[k] for k in ("0", "1")}
    r = SYSTEMS["m2"](ctx_for(scripted(fail_requery=True), budget=1, corpus=small), CLAIM)
    led = Ledger.from_dict(r["ledger"])
    assert led.retrieval_rounds[0]["queries"] == ["effect"]             # obligation's own query reused
    assert r["stop_reason"] in ("no new evidence", "round cap reached")
    assert len(r["retrieved"]) <= 2
    assert any(c["error"] for c in r["calls"] if c["role"] == "requery")


def test_m2_no_iteration_when_all_resolved():
    base = scripted()

    def fn(messages):
        if messages[0]["content"].startswith("ROLE: auditor"):
            return json.dumps({"status": "SATISFIED", "citations": [{"doc": "A1", "sentences": [0]}],
                               "rationale": "", "needs_more_retrieval": False})
        return base(messages)
    r = SYSTEMS["m2"](ctx_for(fn), CLAIM)
    assert r["stop_reason"] == "all obligations resolved" and not r["entered_m2"]
    assert not any(c["role"] == "requery" for c in r["calls"]) and r["pred_label"] == "SUPPORTS"


def test_m2_random_is_seeded_and_budget_matched():
    a = SYSTEMS["m2_random"](ctx_for(scripted()), CLAIM)
    b = SYSTEMS["m2_random"](ctx_for(scripted()), CLAIM)
    ra, rb = a["ledger"]["retrieval_rounds"], b["ledger"]["retrieval_rounds"]
    assert ra == rb
    assert all(len(x["targets"]) <= 2 and len(x["new_doc_ids"]) <= 2 for x in ra)
    reaudited = {x["obligation_id"] for x in a["ledger"]["audits"] if x["round"] > 0 and x["kind"] == "audit"}
    assert reaudited <= {"o2"}                                          # only unresolved are re-audited


def test_claimiter_iterates_only_while_nei():
    r = SYSTEMS["m2_claimiter"](ctx_for(scripted()), CLAIM)
    assert r["first_label"] == "NEI" and r["pred_label"] == "SUPPORTS" and r["stop_reason"] == "verdict reached"
    assert "k" in r["retrieved"] and r["rationale"] == {"k": [0]}
    done = SYSTEMS["m2_claimiter"](ctx_for(scripted(verdict_until_magic=False)), CLAIM)
    assert done["rounds"] == [] and len(done["calls"]) == 1
    capped = SYSTEMS["m2_claimiter"](ctx_for(scripted(requery="study")), CLAIM)
    assert len(capped["rounds"]) == 2 and len(capped["retrieved"]) <= 5


def test_b1_9_uses_m2_maximum_budget():
    r = SYSTEMS["b1_9"](ctx_for(scripted(), budget=5), CLAIM)
    assert len(r["retrieved"]) == 9


def test_b1_sc_votes_with_distinct_seeds():
    labels = itertools.cycle(["REFUTES", "SUPPORTS", "REFUTES", "NEI", "SUPPORTS"])
    seen_seeds = []

    class Spy(LLMClient):
        def _mock(self, messages, temperature, seed):
            seen_seeds.append((temperature, seed))
            return json.dumps({"verdict": next(labels), "citations": [{"doc": "A1", "sentences": [0]}],
                               "justification": ""}), 0, 0
    ctx = ctx_for(scripted(), budget=2)
    ctx.sc_samples = 5                       # the frozen setting is 4; 5 exercises a tie between two labels
    ctx.llm = Spy(LLMConfig(backend="mock"))
    r = SYSTEMS["b1_sc"](ctx, CLAIM)
    assert r["votes"] == ["REFUTES", "SUPPORTS", "REFUTES", "NEI", "SUPPORTS"]
    assert r["pred_label"] == "REFUTES"                                  # REFUTES/SUPPORTS tie, REFUTES first
    assert seen_seeds == [(0.7, 13 + k) for k in range(5)]
    labels2 = iter(["NEI", "SUPPORTS"])
    ctx.sc_samples = 2
    ctx.llm = type("S2", (LLMClient,), {"_mock": lambda self, m, t, s: (json.dumps(
        {"verdict": next(labels2), "citations": [], "justification": ""}), 0, 0)})(LLMConfig(backend="mock"))
    assert SYSTEMS["b1_sc"](ctx, CLAIM)["pred_label"] == "NEI"           # tie including NEI -> NEI


def test_m1_untyped_and_llmagg():
    u = SYSTEMS["m1_untyped"](ctx_for(scripted()), CLAIM)
    assert {o["type"] for o in u["ledger"]["obligations"]} == {UNTYPED}
    assert {o["criticality"] for o in u["ledger"]["obligations"]} == {"core"}
    g = SYSTEMS["m1_llmagg"](ctx_for(scripted()), CLAIM)
    assert g["pred_label"] == "REFUTES" and g["rule_label"] == "NEI" and g["rationale"] == {}


def test_m2_followup_search_is_fused_with_and_without_the_claim():
    """Amendment 3: a follow-up query is searched alone and with the claim appended (RRF-fused), so the
    missing condition can pull in abstracts the claim's own ranking would not."""
    queries = []

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def search(self, query, k=10, exclude=None):
            queries.append(query)
            return self.inner.search(query, k=k, exclude=exclude)
    for mode, expect_alone in (("fused", True), ("claim", False)):
        queries.clear()
        ctx = ctx_for(scripted())
        ctx.retriever, ctx.m2_followup = Spy(ctx.retriever), mode
        SYSTEMS["m2"](ctx, CLAIM)
        assert f"zebrafish {CLAIM.claim}" in queries
        assert ("zebrafish" in queries) == expect_alone
