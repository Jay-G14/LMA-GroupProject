"""SciFact systems and controls under matched evidence budgets (protocol sections 3, 9, 10).

Budget: every retrieval system shows the model at most `budget` distinct
abstracts per claim. B2 and M1 build that pool by reciprocal-rank fusion of
their per-query rankings (proposal, design choice D3), so decomposition changes
*which* abstracts are seen, not how many. In M1 every obligation is audited on
the whole pool, as in the proposal (protocol amendment 2, 2026-10-29); the
original per-obligation subset design is kept as the ablation M1-aligned.
Every model call is logged (tokens, latency, cache hit, error).
"""
from __future__ import annotations

import random
import re
import threading
from collections import Counter
from dataclasses import dataclass, replace

from ..agents.citations import alias_abstracts, resolve
from ..data.scifact import Abstract, Claim
from ..ledger.aggregation import POLICIES, cited_evidence
from ..ledger.schema import (CONTRADICTED, MAX_OBLIGATIONS, SATISFIED, TERMINAL, UNSUPPORTED, UNTYPED, Audit,
                             Ledger, LedgerError, Obligation)
from ..llm import prompts_scifact as P
from ..llm.client import LLMClient
from ..retrieval.hybrid import rrf

INVALID = "__invalid__"
RETRIEVAL_DEPTH = 100


@dataclass
class Context:
    llm: LLMClient
    retriever: object
    corpus: dict[str, Abstract]
    budget: int = 5
    policy: str = "scifact_v1"
    m2_rounds: int = 2               # proposal: at most two extra rounds
    m2_new_per_round: int = 2        # proposal: up to two new abstracts per round
    sc_samples: int = 4              # B1-SC draws, token-matched to M1 on the pilot (Gate 3)
    sc_temperature: float = 0.7
    seed: int = 13
    m2_consolidate: bool = True      # protocol amendment 2026-10-27 (section 12)
    rrf_k: int = 60                  # fusion of per-query rankings (proposal D3)
    # M2 follow-up searches: "fused" = RRF of the follow-up query alone and with the claim appended
    # (amendment 3); "claim" = with the claim appended only (the amendment-2 behaviour).
    m2_followup: str = "fused"

    def __post_init__(self):
        self._search_cache: dict[str, list[str]] = {}
        self._search_lock = threading.Lock()


class CallLog(list):
    def call(self, ctx: Context, role: str, messages, schema, version):
        res = ctx.llm.generate_json(messages, schema, version)
        self.append({"role": role, "prompt_version": version, "prompt_tokens": res.prompt_tokens,
                     "completion_tokens": res.completion_tokens, "latency_s": round(res.latency_s, 3),
                     "cached": res.cached, "error": res.error, "raw_output": res.text if res.error else None})
        return res


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def round_robin_pool(rankings: list[list[str]], budget: int) -> list[str]:
    """Take rank-1 of every list, then rank-2, ..., skipping duplicates, until budget."""
    pool: list[str] = []
    for rank in range(max((len(r) for r in rankings), default=0)):
        for r in rankings:
            if rank < len(r) and r[rank] not in pool:
                pool.append(r[rank])
                if len(pool) == budget:
                    return pool
    return pool


def rrf_pool(rankings: list[list[str]], budget: int, k: int = 60) -> list[str]:
    """Reciprocal-rank fusion of per-query rankings (proposal D3); ties broken by doc id."""
    return [d for d, _ in rrf([[(d, 0.0) for d in r] for r in rankings], k)][:budget]


def _search(ctx: Context, query: str) -> list[str]:
    """Top-RETRIEVAL_DEPTH doc ids for a query; deterministic, so results are cached per run."""
    with ctx._search_lock:
        hit = ctx._search_cache.get(query)
    if hit is None:
        hit = [d for d, _ in ctx.retriever.search(query, k=RETRIEVAL_DEPTH)]
        with ctx._search_lock:
            ctx._search_cache[query] = hit
    return list(hit)


def _verdict_record(ctx, res, alias) -> dict:
    if not res.parsed:
        return {"pred_label": INVALID, "rationale": {}, "invalid_citations": []}
    refs, invalid = resolve(res.parsed.get("citations", []), alias, ctx.corpus)
    label = res.parsed["verdict"]
    return {"pred_label": label, "justification": res.parsed.get("justification"),
            "rationale": {} if label == "NEI" else {r.doc_id: r.sentence_ids for r in refs},
            "invalid_citations": invalid}


# -- baselines ----------------------------------------------------------------------------
def run_b0(ctx: Context, claim: Claim) -> dict:
    log = CallLog()
    res = log.call(ctx, "verdict", P.b0_messages(claim.claim), P.B0_SCHEMA, P.B0_VERSION)
    return {"pred_label": res.parsed["verdict"] if res.parsed else INVALID, "retrieved": [], "rationale": {},
            "justification": res.parsed.get("justification") if res.parsed else None, "calls": log}


def run_b1(ctx: Context, claim: Claim, budget: int | None = None) -> dict:
    log = CallLog()
    docs = _search(ctx, claim.claim)[: budget or ctx.budget]
    shown, alias = alias_abstracts(docs, ctx.corpus)
    res = log.call(ctx, "verdict", P.b1_messages(claim.claim, shown), P.VERDICT_SCHEMA, P.B1_VERSION)
    return {**_verdict_record(ctx, res, alias), "retrieved": docs, "calls": log}


def run_b2(ctx: Context, claim: Claim) -> dict:
    log = CallLog()
    dec = log.call(ctx, "subquestions", P.b2_decomp_messages(claim.claim), P.B2_DECOMP_SCHEMA, P.B2_DECOMP_VERSION)
    questions = list(dict.fromkeys(q.strip() for q in dec.parsed["questions"] if q.strip())) if dec.parsed else []
    questions = questions[:MAX_OBLIGATIONS]
    rankings = [_search(ctx, f"{q} {claim.claim}") for q in questions] or [_search(ctx, claim.claim)]
    docs = rrf_pool(rankings, ctx.budget, ctx.rrf_k)
    shown, alias = alias_abstracts(docs, ctx.corpus)
    res = log.call(ctx, "verdict", P.b2_verdict_messages(claim.claim, questions, shown),
                   P.VERDICT_SCHEMA, P.B2_VERDICT_VERSION)
    return {**_verdict_record(ctx, res, alias), "retrieved": docs, "subquestions": questions,
            "decomposition_error": dec.error, "calls": log}


# -- obligation ledger ------------------------------------------------------------------
def plan(ctx: Context, claim: Claim, log: CallLog) -> Ledger:
    led = Ledger(str(claim.id), claim.claim)
    res = log.call(ctx, "planner", P.planner_messages(claim.claim), P.PLANNER_SCHEMA, P.PLANNER_VERSION)
    if not res.parsed:
        led.planner_error = res.error
        return led
    seen = set()
    for o in res.parsed["obligations"]:
        key = (o["type"], _norm(o["description"]))
        if key in seen:                      # redundant obligation: dropped and counted
            led.planner_error = (led.planner_error or "") + f"duplicate obligation dropped: {o['description']}; "
            continue
        seen.add(key)
        led.add_obligation(Obligation(f"o{len(led.obligations) + 1}", o["description"].strip(), o["type"],
                                      o["expected_condition"].strip(), o["query"].strip()))
    return led


def audit(ctx: Context, led: Ledger, ob: Obligation, doc_ids: list[str], rnd: int, log: CallLog,
          role: str = "auditor") -> Audit:
    shown, alias = alias_abstracts(doc_ids, ctx.corpus)
    res = log.call(ctx, role, P.auditor_messages(led.claim, ob.description, ob.expected_condition, shown),
                   P.AUDITOR_SCHEMA, P.AUDITOR_VERSION)
    if not res.parsed:
        return Audit(ob.obligation_id, rnd, UNSUPPORTED, UNSUPPORTED, [], "", True, doc_ids, error=res.error)
    refs, invalid = resolve(res.parsed["citations"], alias, ctx.corpus)
    status = res.parsed["status"]
    # A resolving status needs a valid citation; otherwise it counts as UNSUPPORTED.
    effective = status if status not in TERMINAL or refs else UNSUPPORTED
    return Audit(ob.obligation_id, rnd, status, effective, refs,
                 res.parsed["rationale"], bool(res.parsed["needs_more_retrieval"]), doc_ids, invalid,
                 None if effective == status else "resolving status without valid citation")


def plan_untyped(ctx: Context, claim: Claim, log: CallLog) -> Ledger:
    """M1-untyped: FActScore-style atomic facts; every fact is a core obligation."""
    led = Ledger(str(claim.id), claim.claim)
    res = log.call(ctx, "facts", P.facts_messages(claim.claim), P.FACTS_SCHEMA, P.FACTS_VERSION)
    if not res.parsed:
        led.planner_error = res.error
        return led
    seen = set()
    for f in res.parsed["facts"]:
        if _norm(f["fact"]) in seen:
            led.planner_error = (led.planner_error or "") + f"duplicate fact dropped: {f['fact']}; "
            continue
        seen.add(_norm(f["fact"]))
        led.add_obligation(Obligation(f"o{len(led.obligations) + 1}", f["fact"].strip(), UNTYPED,
                                      f["fact"].strip(), f["query"].strip()))
    return led


def _add_audit(led: Ledger, a: Audit) -> None:
    try:
        led.add_audit(a)
    except LedgerError as e:               # invariant violation is recorded, not hidden
        led.planner_error = (led.planner_error or "") + f"ledger error: {e}; "


EVIDENCE_MODES = ("pool", "aligned", "shared")


def build_ledger(ctx: Context, claim: Claim, log: CallLog, evidence: str = "pool",
                 untyped: bool = False) -> tuple[Ledger, list[str], dict[str, list[str]]]:
    """Plan, retrieve a claim pool of `budget` abstracts, audit every obligation (round 0).

    evidence="pool"    (M1, proposal): per-obligation queries ("{query} {claim}") are fused by RRF into
                       one pool, and every obligation is audited on the whole pool.
    evidence="aligned" (M1-aligned ablation, the original implementation): round-robin pool; each
                       obligation is audited only on the pool abstracts within its own top-`budget`.
    evidence="shared"  (M1-shared control): the claim-level top-`budget` (B1's evidence) for every obligation."""
    if evidence not in EVIDENCE_MODES:
        raise ValueError(f"unknown evidence mode {evidence!r}")
    led = plan_untyped(ctx, claim, log) if untyped else plan(ctx, claim, log)
    if evidence == "shared" or not led.obligations:
        pool = _search(ctx, claim.claim)[: ctx.budget]
        per_ob = {o.obligation_id: list(pool) for o in led.obligations}
    else:
        rankings = {o.obligation_id: _search(ctx, f"{o.query} {claim.claim}") for o in led.obligations}
        if evidence == "pool":
            pool = rrf_pool(list(rankings.values()), ctx.budget, ctx.rrf_k)
            per_ob = {oid: list(pool) for oid in rankings}
        else:
            pool = round_robin_pool(list(rankings.values()), ctx.budget)
            per_ob = {}
            for oid, ranking in rankings.items():
                own = [d for d in ranking[: ctx.budget] if d in pool]
                per_ob[oid] = own or [min(pool, key=lambda d: ranking.index(d) if d in ranking else len(ranking))]
    for o in led.obligations:
        _add_audit(led, audit(ctx, led, o, per_ob[o.obligation_id], 0, log))
    return led, pool, per_ob


def _ledger_record(ctx: Context, led: Ledger, pool: list[str], log: CallLog) -> dict:
    v = POLICIES[ctx.policy](led)
    statuses = [led.status(o.obligation_id) for o in led.obligations]
    return {"pred_label": v.label, "verdict_reason": v.reason, "rationale": v.rationale_docs, "retrieved": pool,
            "ledger": led.to_dict(), "n_obligations": len(led.obligations),
            "n_satisfied": statuses.count(SATISFIED), "n_contradicted": statuses.count(CONTRADICTED),
            "n_unresolved": len(led.unresolved()), "calls": log}


def run_m1(ctx: Context, claim: Claim, evidence: str = "pool", untyped: bool = False) -> dict:
    """M1 (fused obligation pool, every obligation audited on it); M1-aligned (each obligation on its own
    subset); M1-shared (B1's claim-level top-k); M1-untyped (atomic facts instead of typed obligations)."""
    log = CallLog()
    led, pool, _ = build_ledger(ctx, claim, log, evidence, untyped)
    return _ledger_record(ctx, led, pool, log)


def run_m1_llmagg(ctx: Context, claim: Claim) -> dict:
    """Same ledger as M1; the verdict comes from the LLM instead of the deterministic rule."""
    log = CallLog()
    led, pool, _ = build_ledger(ctx, claim, log)
    rule = POLICIES[ctx.policy](led)
    rows = []
    for o in led.obligations:
        a = led.latest(o.obligation_id)
        rows.append({"criticality": o.criticality, "description": o.description,
                     "status": a.effective_status if a else UNSUPPORTED, "rationale": a.rationale if a else "",
                     "evidence": [e.quoted_text for e in a.evidence] if a else []})
    res = log.call(ctx, "aggregate", P.aggregate_messages(claim.claim, rows), P.B0_SCHEMA, P.AGGREGATE_VERSION)
    rec = _ledger_record(ctx, led, pool, log)
    label = res.parsed["verdict"] if res.parsed else INVALID
    status = {"SUPPORTS": SATISFIED, "REFUTES": CONTRADICTED}.get(label)
    rec.update({"pred_label": label, "rule_label": rule.label, "verdict_reason": "llm aggregation",
                "justification": res.parsed.get("justification") if res.parsed else None,
                "rationale": cited_evidence(led, status) if status else {}})
    return rec


# -- M2 and its controls ------------------------------------------------------------------
def consolidate(ctx: Context, led: Ledger, rnd: int, log: CallLog) -> list[dict]:
    """Protocol amendment 2026-10-27: for every abstract cited by a core obligation newly
    SATISFIED in round `rnd`, re-check each other SATISFIED core obligation against that
    abstract alone. Uses only abstracts already retrieved (no new retrieval)."""
    core = [o for o in led.obligations if o.criticality == "core"]
    newly = [o for o in core if led.status(o.obligation_id) == SATISFIED and led.latest(o.obligation_id).round == rnd]
    docs = list(dict.fromkeys(e.doc_id for o in newly for e in led.evidence(o.obligation_id)))
    done = []
    for d in docs:
        for o in core:
            if led.status(o.obligation_id) != SATISFIED or d in {e.doc_id for e in led.evidence(o.obligation_id)}:
                continue
            if any(a.kind == "consolidation" and a.obligation_id == o.obligation_id and a.shown_doc_ids == [d]
                   for a in led.audits):
                continue
            a = replace(audit(ctx, led, o, [d], rnd, log, role="consolidation"), kind="consolidation")
            _add_audit(led, a)
            done.append({"obligation_id": o.obligation_id, "doc_id": d, "status": a.effective_status})
    return done


def _new_docs(ctx: Context, queries: list[str], claim: Claim, seen: set[str]) -> list[str]:
    """Up to `m2_new_per_round` unseen abstracts for the follow-up queries, RRF-fused. With the claim
    appended, a follow-up query mostly re-finds the claim's own next-ranked abstracts (what B1-9 adds);
    fusing in the query alone lets the missing condition pull in new evidence (amendment 3)."""
    rankings = []
    for q in queries:
        rankings.append([d for d in _search(ctx, f"{q} {claim.claim}") if d not in seen])
        if ctx.m2_followup == "fused":
            rankings.append([d for d in _search(ctx, q) if d not in seen])
    return rrf_pool(rankings, ctx.m2_new_per_round, ctx.rrf_k)


def run_m2(ctx: Context, claim: Claim, targeting: str = "status") -> dict:
    """M2: after M1, up to `m2_rounds` rounds; each round writes a follow-up query for the
    targeted obligations, retrieves at most `m2_new_per_round` unseen abstracts for the
    claim, re-audits every unresolved obligation on its evidence plus the new abstracts,
    then runs the consolidation audit (amendment 2026-10-27) for newly satisfied core ones.
    targeting="status": targets are the PARTIAL/UNSUPPORTED obligations (the method).
    targeting="random": M2-random, the same number of targets drawn at random from all
    obligations (seeded per claim); retrieval budget and re-audit are unchanged."""
    log = CallLog()
    led, pool, per_ob = build_ledger(ctx, claim, log)
    m1, m1_pool, round0_calls = POLICIES[ctx.policy](led), list(pool), len(log)
    rng = random.Random(f"{ctx.seed}-{claim.id}")
    stop = "round cap reached"
    for rnd in range(1, ctx.m2_rounds + 1):
        unresolved = led.unresolved()
        if not unresolved:
            stop = "all obligations resolved" if led.obligations else "no obligations"
            break
        targets = unresolved if targeting == "status" else rng.sample(led.obligations, len(unresolved))
        queries = []
        for o in targets:
            a = led.latest(o.obligation_id)
            res = log.call(ctx, "requery", P.requery_messages(
                claim.claim, o.description, o.expected_condition, a.effective_status if a else UNSUPPORTED,
                a.rationale if a else ""), P.REQUERY_SCHEMA, P.REQUERY_VERSION)
            queries.append(res.parsed["query"] if res.parsed else o.query)   # failed requery: reuse own query
        new = _new_docs(ctx, queries, claim, set(pool))
        led.retrieval_rounds.append({"round": rnd, "targets": [o.obligation_id for o in targets],
                                     "queries": queries, "new_doc_ids": new,
                                     "stop_reason": None if new else "no new evidence"})
        if not new:
            stop = "no new evidence"
            break
        pool += new
        for o in unresolved:
            per_ob[o.obligation_id] = per_ob[o.obligation_id] + new
            _add_audit(led, audit(ctx, led, o, per_ob[o.obligation_id], rnd, log))
        if ctx.m2_consolidate:
            led.retrieval_rounds[-1]["consolidations"] = consolidate(ctx, led, rnd, log)
    else:
        if not led.unresolved():
            stop = "all obligations resolved"
    led.retrieval_rounds.append({"round": len(led.retrieval_rounds) + 1, "targets": [], "new_doc_ids": [],
                                 "stop_reason": stop})
    rec = _ledger_record(ctx, led, pool, log)
    rec.update({"m1_label": m1.label, "m1_retrieved": m1_pool, "m1_rationale": m1.rationale_docs,
                "round0_calls": round0_calls, "verdict_changed": rec["pred_label"] != m1.label, "stop_reason": stop,
                "rounds_used": sum(1 for r in led.retrieval_rounds if r["new_doc_ids"]),
                "entered_m2": any(r["targets"] for r in led.retrieval_rounds)})
    return rec


def run_m2_claimiter(ctx: Context, claim: Claim) -> dict:
    """FIRE/QACheck-style control: claim-level verdict; while it is NEI, write one
    claim-level follow-up query and add at most `m2_new_per_round` unseen abstracts,
    for at most `m2_rounds` rounds (M2's caps)."""
    log = CallLog()
    docs = _search(ctx, claim.claim)[: ctx.budget]
    shown, alias = alias_abstracts(docs, ctx.corpus)
    res = log.call(ctx, "verdict", P.b1_messages(claim.claim, shown), P.VERDICT_SCHEMA, P.B1_VERSION)
    first, first_docs = res.parsed["verdict"] if res.parsed else INVALID, list(docs)
    first_rationale = _verdict_record(ctx, res, alias)["rationale"]
    rounds, stop = [], "round cap reached"
    for rnd in range(1, ctx.m2_rounds + 1):
        if not res.parsed or res.parsed["verdict"] != "NEI":
            stop = "verdict reached" if res.parsed else "verdict call failed"
            break
        q = log.call(ctx, "claim_requery", P.claim_requery_messages(claim.claim, res.parsed.get("justification", "")),
                     P.REQUERY_SCHEMA, P.CLAIM_REQUERY_VERSION)
        new = _new_docs(ctx, [q.parsed["query"] if q.parsed else claim.claim], claim, set(docs))
        rounds.append({"round": rnd, "query": q.parsed["query"] if q.parsed else None, "new_doc_ids": new})
        if not new:
            stop = "no new evidence"
            break
        docs += new
        shown, alias = alias_abstracts(docs, ctx.corpus)
        res = log.call(ctx, "verdict", P.b1_messages(claim.claim, shown), P.VERDICT_SCHEMA, P.B1_ITER_VERSION)
    else:
        if res.parsed and res.parsed["verdict"] != "NEI":
            stop = "verdict reached"
    rec = {**_verdict_record(ctx, res, alias), "retrieved": docs, "calls": log, "rounds": rounds,
           "stop_reason": stop, "first_label": first, "m1_retrieved": first_docs, "m1_rationale": first_rationale,
           "round0_calls": 1}
    rec["verdict_changed"] = rec["pred_label"] != first
    return rec


# -- direct-RAG controls ------------------------------------------------------------------
def run_b1_9(ctx: Context, claim: Claim) -> dict:
    """B1 with M2's maximum evidence: budget + rounds x new-per-round abstracts (9 for SciFact)."""
    return run_b1(ctx, claim, budget=ctx.budget + ctx.m2_rounds * ctx.m2_new_per_round)


def run_b1_sc(ctx: Context, claim: Claim) -> dict:
    """B1 with self-consistency: `sc_samples` sampled verdicts on the same evidence, majority
    vote. Ties go to NEI if NEI is tied, else to the label sampled first."""
    log = CallLog()
    docs = _search(ctx, claim.claim)[: ctx.budget]
    shown, alias = alias_abstracts(docs, ctx.corpus)
    msgs = P.b1_messages(claim.claim, shown)
    samples = []
    for k in range(ctx.sc_samples):
        res = ctx.llm.generate_json(msgs, P.VERDICT_SCHEMA, P.B1_VERSION, temperature=ctx.sc_temperature, sample=k)
        log.append({"role": "verdict", "prompt_version": P.B1_VERSION, "sample": k,
                    "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                    "latency_s": round(res.latency_s, 3), "cached": res.cached, "error": res.error,
                    "raw_output": res.text if res.error else None})
        samples.append(res)
    votes = [s.parsed["verdict"] for s in samples if s.parsed]
    if not votes:
        return {"pred_label": INVALID, "rationale": {}, "invalid_citations": [], "retrieved": docs,
                "votes": [], "calls": log}
    counts = Counter(votes)
    top = max(counts.values())
    tied = [l for l in dict.fromkeys(votes) if counts[l] == top]
    label = "NEI" if "NEI" in tied else tied[0]
    chosen = next(s for s in samples if s.parsed and s.parsed["verdict"] == label)
    return {**_verdict_record(ctx, chosen, alias), "retrieved": docs, "votes": votes, "calls": log}


SYSTEMS = {"b0": run_b0, "b1": run_b1, "b1_9": run_b1_9, "b1_sc": run_b1_sc, "b2": run_b2,
           "m1": run_m1,
           "m1_shared": lambda ctx, c: run_m1(ctx, c, evidence="shared"),
           "m1_aligned": lambda ctx, c: run_m1(ctx, c, evidence="aligned"),
           "m1_untyped": lambda ctx, c: run_m1(ctx, c, untyped=True),
           "m1_llmagg": run_m1_llmagg,
           "m2": run_m2,
           "m2_random": lambda ctx, c: run_m2(ctx, c, targeting="random"),
           "m2_claimiter": run_m2_claimiter}
