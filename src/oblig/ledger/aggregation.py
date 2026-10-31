"""Deterministic verdict policies over a ledger's current (effective) statuses.

scifact_v1 is the proposal's rule and is fixed. pubhealth_v1 is a *candidate*
four-class rule; it is selected or revised on PUBHEALTH dev before Gate 3.

Rationale order (protocol amendment 2, 2026-10-29). SciFact's abstract-level metric scores only the first
three predicted sentences of an abstract, so the order of a ledger's rationale matters. Sentences are
ordered most decisive first: cited by more of the deciding obligations, then by obligation type
(relation, outcome/atomic, subject, qualifiers), then by the position the Auditor gave them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import CONTRADICTED, PARTIAL, SATISFIED, Ledger, Obligation

TYPE_PRIORITY = {"relation": 0, "outcome": 1, "atomic": 1, "subject": 2}   # qualifiers: 3


@dataclass
class Verdict:
    label: str
    reason: str
    rationale_docs: dict[str, list[int]] = field(default_factory=dict)   # doc_id -> cited sentence ids, ordered


def ordered_sentences(ledger: Ledger, obligations: list[Obligation], doc: str) -> list[int]:
    """One abstract's sentences cited by `obligations`, most decisive first (see module docstring)."""
    stats: dict[int, tuple[set, int, int]] = {}
    for o in obligations:
        pri = TYPE_PRIORITY.get(o.type, 3)
        for e in ledger.evidence(o.obligation_id):
            if e.doc_id != doc:
                continue
            for pos, s in enumerate(e.sentence_ids):
                obs, p, q = stats.get(s, (set(), 9, 99))
                stats[s] = (obs | {o.obligation_id}, min(p, pri), min(q, pos))
    return sorted(stats, key=lambda s: (-len(stats[s][0]), stats[s][1], stats[s][2], s))


def cited_evidence(ledger: Ledger, status: str, core_only: bool = False) -> dict[str, list[int]]:
    obs = [o for o in ledger.obligations
           if (a := ledger.latest(o.obligation_id)) and a.effective_status == status
           and (not core_only or o.criticality == "core")]
    docs = dict.fromkeys(e.doc_id for o in obs for e in ledger.evidence(o.obligation_id))
    return {d: ordered_sentences(ledger, obs, d) for d in docs}


def scifact_v1(ledger: Ledger) -> Verdict:
    """REFUTES if any obligation CONTRADICTED; SUPPORTS if every core obligation is
    SATISFIED by sentences of one abstract and none is CONTRADICTED; else NEI.
    Qualifiers never block SUPPORTS."""
    core = [o for o in ledger.obligations if o.criticality == "core"]
    if not core:
        return Verdict("NEI", "no core obligations")
    if any(ledger.status(o.obligation_id) == CONTRADICTED for o in ledger.obligations):
        return Verdict("REFUTES", "an obligation is contradicted", cited_evidence(ledger, CONTRADICTED))
    if not all(ledger.status(o.obligation_id) == SATISFIED for o in core):
        return Verdict("NEI", "a core obligation is not satisfied")
    docs_per_ob = [{e.doc_id for e in ledger.evidence(o.obligation_id)} for o in core]
    common = set.intersection(*docs_per_ob)
    if not common:
        return Verdict("NEI", "core obligations satisfied only across different abstracts")
    doc = sorted(common)[0]
    return Verdict("SUPPORTS", "all core obligations satisfied in one abstract", {doc: ordered_sentences(ledger, core, doc)})


def _docs(ledger: Ledger, o: Obligation) -> set[str]:
    return {e.doc_id for e in ledger.evidence(o.obligation_id)}


def anchored(ledger: Ledger, o: Obligation, satisfied_subject: bool = False) -> bool:
    """Protocol amendment 5 (2026-10-30): a contradiction is anchored if an abstract it cites is also cited by a
    subject obligation of the same ledger (with satisfied_subject, a SATISFIED one). In a ledger without subject
    obligations (M1-untyped) every contradiction is anchored."""
    subjects = [s for s in ledger.obligations if s.type == "subject"]
    if not subjects:
        return True
    return any(_docs(ledger, o) & _docs(ledger, s) for s in subjects
               if not satisfied_subject or ledger.status(s.obligation_id) == SATISFIED)


def scifact_v1_filtered(ledger: Ledger, counts) -> Verdict:
    """scifact_v1, except that a CONTRADICTED obligation for which counts(ledger, o) is False is treated as
    unresolved: it cannot give REFUTES, and a core one blocks SUPPORTS."""
    core = [o for o in ledger.obligations if o.criticality == "core"]
    if not core:
        return Verdict("NEI", "no core obligations")
    contradicted = [o for o in ledger.obligations if ledger.status(o.obligation_id) == CONTRADICTED]
    counted = [o for o in contradicted if counts(ledger, o)]
    if counted:
        docs = dict.fromkeys(e.doc_id for o in counted for e in ledger.evidence(o.obligation_id))
        return Verdict("REFUTES", "an obligation is contradicted", {d: ordered_sentences(ledger, counted, d) for d in docs})
    if not all(ledger.status(o.obligation_id) == SATISFIED for o in core):
        return Verdict("NEI", "a contradiction does not count under the rule" if contradicted
                       else "a core obligation is not satisfied")
    common = set.intersection(*[_docs(ledger, o) for o in core])
    if not common:
        return Verdict("NEI", "core obligations satisfied only across different abstracts")
    doc = sorted(common)[0]
    return Verdict("SUPPORTS", "all core obligations satisfied in one abstract", {doc: ordered_sentences(ledger, core, doc)})


# Amendment 5: the anchored rule, chosen on prompt_dev (best of the four contradiction rules tried there); the
# other three are reported as ablations of the rule only.
CONTRADICTION_RULES = {
    "scifact_v1_anchored": anchored,
    "scifact_v1_nosubject": lambda led, o: o.type != "subject",
    "scifact_v1_relqual": lambda led, o: o.type not in ("subject", "outcome"),
    "scifact_v1_anchoredsat": lambda led, o: anchored(led, o, satisfied_subject=True),
}
# Run-id suffix of a run re-scored under each rule (scripts/rescore_ledgers.py, scripts/compare_systems.py).
POLICY_TAGS = {"scifact_v1_anchored": "anch", "scifact_v1_nosubject": "nosubject", "scifact_v1_relqual": "relqual",
               "scifact_v1_anchoredsat": "anchsat"}


def pubhealth_v1(ledger: Ledger) -> Verdict:
    """Candidate: mixture if core obligations are both satisfied and contradicted
    (or a core one is only PARTIAL while another is contradicted); false if a core
    obligation is contradicted and none satisfied; true if all core satisfied;
    unproven otherwise."""
    core = [o for o in ledger.obligations if o.criticality == "core"]
    if not core:
        return Verdict("unproven", "no core obligations")
    st = [ledger.status(o.obligation_id) for o in core]
    if CONTRADICTED in st and (SATISFIED in st or PARTIAL in st):
        return Verdict("mixture", "core obligations both established and contradicted",
                       {**cited_evidence(ledger, SATISFIED, True), **cited_evidence(ledger, CONTRADICTED, True)})
    if CONTRADICTED in st:
        return Verdict("false", "a core obligation is contradicted", cited_evidence(ledger, CONTRADICTED, True))
    if all(s == SATISFIED for s in st):
        return Verdict("true", "all core obligations satisfied", cited_evidence(ledger, SATISFIED, True))
    return Verdict("unproven", "core obligations unresolved")


POLICIES = {"scifact_v1": scifact_v1, "pubhealth_v1": pubhealth_v1,
            **{name: (lambda led, rule=rule: scifact_v1_filtered(led, rule)) for name, rule in CONTRADICTION_RULES.items()}}
