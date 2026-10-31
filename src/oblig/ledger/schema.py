"""Evidence-obligation ledger: data model, invariants and serialisation.

The ledger is append-only: every audit is kept with its retrieval round, so the
history of earlier rounds is never overwritten. An obligation's current status
is its latest audit. SATISFIED and CONTRADICTED are terminal; PARTIAL and
UNSUPPORTED may be re-audited in a later round (M2).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import jsonschema

CORE_TYPES = ("subject", "outcome", "relation")
QUALIFIER_TYPES = ("population", "comparator", "timepoint", "scope", "other")
UNTYPED = "atomic"            # M1-untyped control: FActScore-style facts, all treated as core
OBLIGATION_TYPES = CORE_TYPES + QUALIFIER_TYPES + (UNTYPED,)

SATISFIED, CONTRADICTED, PARTIAL, UNSUPPORTED = "SATISFIED", "CONTRADICTED", "PARTIAL", "UNSUPPORTED"
STATUSES = (SATISFIED, CONTRADICTED, PARTIAL, UNSUPPORTED)
TERMINAL = (SATISFIED, CONTRADICTED)
MAX_OBLIGATIONS = 6


class LedgerError(ValueError):
    pass


def criticality(obligation_type: str) -> str:
    """Derived deterministically from the type, never chosen by the model."""
    return "core" if obligation_type in CORE_TYPES or obligation_type == UNTYPED else "qualifier"


@dataclass
class Obligation:
    obligation_id: str
    description: str
    type: str
    expected_condition: str
    query: str

    @property
    def criticality(self) -> str:
        return criticality(self.type)


@dataclass
class EvidenceRef:
    doc_id: str
    sentence_ids: list[int]
    quoted_text: str          # filled from the corpus, never from model output


@dataclass
class Audit:
    obligation_id: str
    round: int
    status: str                       # status as returned by the Auditor
    effective_status: str             # status used for aggregation (see Ledger.add_audit)
    evidence: list[EvidenceRef]
    rationale: str
    needs_more_retrieval: bool
    shown_doc_ids: list[str]          # evidence the Auditor was given in this round
    invalid_citations: list[dict] = field(default_factory=list)
    error: str | None = None
    # "audit": a status judgement. "consolidation" (M2 amendment, 2026-10-27): a re-check of an
    # already SATISFIED core obligation against one newly cited abstract. It never changes the
    # status; when SATISFIED with a valid citation it only adds evidence (see Ledger.evidence).
    kind: str = "audit"


@dataclass
class Ledger:
    claim_id: str
    claim: str
    obligations: list[Obligation] = field(default_factory=list)
    audits: list[Audit] = field(default_factory=list)
    planner_error: str | None = None
    # One entry per re-retrieval round (M2 variants): targets, queries, new docs, stop reason.
    retrieval_rounds: list[dict] = field(default_factory=list)

    # -- construction -------------------------------------------------------------------
    def add_obligation(self, ob: Obligation) -> None:
        if ob.type not in OBLIGATION_TYPES:
            raise LedgerError(f"unknown obligation type {ob.type!r}")
        if len(self.obligations) >= MAX_OBLIGATIONS:
            raise LedgerError("obligation cap reached")
        if any(o.obligation_id == ob.obligation_id for o in self.obligations):
            raise LedgerError(f"duplicate obligation id {ob.obligation_id}")
        self.obligations.append(ob)

    def obligation(self, oid: str) -> Obligation:
        for o in self.obligations:
            if o.obligation_id == oid:
                return o
        raise LedgerError(f"unknown obligation {oid}")

    def add_audit(self, audit: Audit) -> None:
        self.obligation(audit.obligation_id)
        if audit.status not in STATUSES or audit.effective_status not in STATUSES:
            raise LedgerError(f"invalid status {audit.status!r}")
        if audit.kind == "consolidation":
            self._add_consolidation(audit)
            return
        if audit.kind != "audit":
            raise LedgerError(f"unknown audit kind {audit.kind!r}")
        prev = self.latest(audit.obligation_id)
        if prev is not None:
            if prev.effective_status in TERMINAL:
                raise LedgerError(f"{audit.obligation_id} already resolved as {prev.effective_status}")
            if audit.round <= prev.round:
                raise LedgerError("audit rounds must increase")
        # A resolving status without a valid citation cannot resolve the obligation.
        if audit.effective_status in TERMINAL and not audit.evidence:
            raise LedgerError("SATISFIED/CONTRADICTED require at least one valid citation")
        self.audits.append(audit)

    def _add_consolidation(self, audit: Audit) -> None:
        ob, prev = self.obligation(audit.obligation_id), self.latest(audit.obligation_id)
        if ob.criticality != "core" or prev is None or prev.effective_status != SATISFIED:
            raise LedgerError("consolidation applies only to SATISFIED core obligations")
        if audit.round < prev.round:
            raise LedgerError("consolidation cannot precede the audit it extends")
        if len(audit.shown_doc_ids) != 1:
            raise LedgerError("consolidation checks exactly one abstract")
        if audit.effective_status == SATISFIED and not audit.evidence:
            raise LedgerError("SATISFIED consolidation requires a valid citation")
        if any(a.kind == "consolidation" and a.obligation_id == audit.obligation_id
               and a.shown_doc_ids == audit.shown_doc_ids for a in self.audits):
            raise LedgerError("abstract already consolidated for this obligation")
        self.audits.append(audit)

    # -- queries ------------------------------------------------------------------------
    def latest(self, oid: str) -> Audit | None:
        """Latest status judgement (consolidations excluded)."""
        hist = [a for a in self.audits if a.obligation_id == oid and a.kind == "audit"]
        return hist[-1] if hist else None

    def evidence(self, oid: str) -> list[EvidenceRef]:
        """Evidence behind the current status: the latest audit's citations plus, for a
        SATISFIED obligation, citations from SATISFIED consolidations."""
        a = self.latest(oid)
        if a is None:
            return []
        extra = [e for c in self.audits if c.kind == "consolidation" and c.obligation_id == oid
                 and c.effective_status == SATISFIED and a.effective_status == SATISFIED for e in c.evidence]
        return a.evidence + extra

    def status(self, oid: str) -> str | None:
        a = self.latest(oid)
        return a.effective_status if a else None

    def unresolved(self) -> list[Obligation]:
        return [o for o in self.obligations if self.status(o.obligation_id) not in TERMINAL]

    def seen_doc_ids(self, oid: str | None = None) -> set[str]:
        return {d for a in self.audits if oid is None or a.obligation_id == oid for d in a.shown_doc_ids}

    # -- serialisation ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        for o, od in zip(self.obligations, d["obligations"]):
            od["criticality"] = o.criticality
            od["status"] = self.status(o.obligation_id)
        jsonschema.validate(d, LEDGER_SCHEMA)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Ledger":
        jsonschema.validate(d, LEDGER_SCHEMA)
        led = cls(d["claim_id"], d["claim"], planner_error=d.get("planner_error"),
                  retrieval_rounds=d.get("retrieval_rounds", []))
        for od in d["obligations"]:
            led.add_obligation(Obligation(**{k: od[k] for k in
                                             ("obligation_id", "description", "type", "expected_condition", "query")}))
        for ad in d["audits"]:
            led.add_audit(Audit(**{**ad, "evidence": [EvidenceRef(**e) for e in ad["evidence"]]}))
        return led


_EVIDENCE = {"type": "object", "required": ["doc_id", "sentence_ids", "quoted_text"],
             "properties": {"doc_id": {"type": "string"},
                            "sentence_ids": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                            "quoted_text": {"type": "string"}}}
LEDGER_SCHEMA = {
    "type": "object",
    "required": ["claim_id", "claim", "obligations", "audits"],
    "properties": {
        "claim_id": {"type": "string"}, "claim": {"type": "string"},
        "planner_error": {"type": ["string", "null"]},
        "retrieval_rounds": {"type": "array", "items": {
            "type": "object", "required": ["round", "targets", "new_doc_ids", "stop_reason"]}},
        "obligations": {"type": "array", "maxItems": MAX_OBLIGATIONS, "items": {
            "type": "object",
            "required": ["obligation_id", "description", "type", "expected_condition", "query"],
            "properties": {"type": {"enum": list(OBLIGATION_TYPES)},
                           "criticality": {"enum": ["core", "qualifier"]},
                           "status": {"enum": list(STATUSES) + [None]}}}},
        "audits": {"type": "array", "items": {
            "type": "object",
            "required": ["obligation_id", "round", "status", "effective_status", "evidence", "rationale",
                         "needs_more_retrieval", "shown_doc_ids"],
            "properties": {"round": {"type": "integer", "minimum": 0},
                           "kind": {"enum": ["audit", "consolidation"]},
                           "status": {"enum": list(STATUSES)}, "effective_status": {"enum": list(STATUSES)},
                           "evidence": {"type": "array", "items": _EVIDENCE},
                           "needs_more_retrieval": {"type": "boolean"}}}},
    },
}
