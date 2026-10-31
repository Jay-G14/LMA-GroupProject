"""Versioned SciFact prompts and output schemas (B0, B1, B2, Planner, Auditor).

Each system prompt starts with a role tag ("ROLE: ...") used only by the mock
backend to route test outputs. Bump a version when its wording changes.

Prompts that read abstracts put them first (v3, before any real-model run), so every call about the
same evidence shares one prefix and the server's prefix cache reuses it (e.g. all Auditor calls of a
claim, all B1-SC samples). The order is the same for every system.
"""
from __future__ import annotations

from ..data.scifact import LABELS
from ..ledger.schema import CORE_TYPES, MAX_OBLIGATIONS, QUALIFIER_TYPES, STATUSES

# v2 definitions (after the prompt_dev pilot, 2026-10-30): the pilot's errors on NEI claims were mostly
# "contradictions" read into a different finding or into missing information. The same clarification goes
# into the verdict labels (all direct verdicts) and the Auditor statuses, so no system gets extra help.
LABEL_DEFINITIONS = (
    "Labels:\n"
    "- SUPPORTS: the evidence establishes the claim.\n"
    "- REFUTES: the evidence states the opposite of the claim (for example a decrease where the claim says an "
    "increase). A different finding, a different topic or missing information is not a refutation.\n"
    "- NEI: the evidence neither establishes nor contradicts the claim, including when the abstracts do not address it.")

# Same instruction for every system (protocol amendment 2): SciFact's abstract-level metric scores only the
# first three predicted sentences of an abstract, in the order given.
CITE_RULE = "Cite at most 3 sentences per abstract, most important first."

_CITATIONS = {"type": "array", "items": {
    "type": "object", "required": ["doc", "sentences"],
    "properties": {"doc": {"type": "string"},
                   "sentences": {"type": "array", "items": {"type": "integer"}}}}}


def _system(role: str, body: str) -> dict:
    return {"role": "system", "content": f"ROLE: {role}\n{body} Answer with a single JSON object only."}


def format_abstracts(abstracts: list[tuple[str, str, list[str]]]) -> str:
    """abstracts: (alias, title, sentences). Sentences are numbered from 0."""
    blocks = []
    for alias, title, sents in abstracts:
        lines = "\n".join(f"  ({i}) {s}" for i, s in enumerate(sents))
        blocks.append(f"[{alias}] {title}\n{lines}")
    return "\n\n".join(blocks)


# -- B0 ----------------------------------------------------------------------------------
B0_VERSION = "sf-b0-v2"
B0_SCHEMA = {"type": "object", "required": ["verdict", "justification"],
             "properties": {"verdict": {"enum": list(LABELS)}, "justification": {"type": "string"}}}


def b0_messages(claim: str) -> list[dict]:
    return [_system("verdict", "You verify scientific claims."),
            {"role": "user", "content": (
                f"Judge the claim from your own knowledge.\n{LABEL_DEFINITIONS}\n\nClaim: {claim}\n\n"
                'Return JSON: {"verdict": "SUPPORTS|REFUTES|NEI", "justification": "<one or two sentences>"}')}]


# -- B1 ----------------------------------------------------------------------------------
B1_VERSION = "sf-b1-v4"
VERDICT_SCHEMA = {"type": "object", "required": ["verdict", "citations", "justification"],
                  "properties": {"verdict": {"enum": list(LABELS)}, "citations": _CITATIONS,
                                 "justification": {"type": "string"}}}


def b1_messages(claim: str, abstracts) -> list[dict]:
    return [_system("verdict", "You verify scientific claims against research abstracts."),
            {"role": "user", "content": (
                f"Abstracts:\n{format_abstracts(abstracts)}\n\n"
                f"Judge the claim using only the abstracts above.\n{LABEL_DEFINITIONS}\n\nClaim: {claim}\n\n"
                f"Cite the abstract ids and sentence numbers your verdict relies on (none for NEI). {CITE_RULE}\n"
                'Return JSON: {"verdict": "SUPPORTS|REFUTES|NEI", "citations": [{"doc": "A1", "sentences": [0]}], '
                '"justification": "<one or two sentences>"}')}]


# -- B2: ordinary sub-questions, one verdict, no statuses ----------------------------------
B2_DECOMP_VERSION = "sf-b2-decomp-v1"
B2_DECOMP_SCHEMA = {"type": "object", "required": ["questions"],
                    "properties": {"questions": {"type": "array", "minItems": 1, "maxItems": MAX_OBLIGATIONS,
                                                 "items": {"type": "string"}}}}


def b2_decomp_messages(claim: str) -> list[dict]:
    return [_system("subquestions", "You help search the scientific literature."),
            {"role": "user", "content": (
                f"Write up to {MAX_OBLIGATIONS} short search questions whose answers would help verify the claim. "
                f"Use as few as the claim needs.\n\nClaim: {claim}\n\n"
                'Return JSON: {"questions": ["<question>", ...]}')}]


B2_VERDICT_VERSION = "sf-b2-verdict-v4"


def b2_verdict_messages(claim: str, questions: list[str], abstracts) -> list[dict]:
    qs = "\n".join(f"- {q}" for q in questions)
    return [_system("verdict", "You verify scientific claims against research abstracts."),
            {"role": "user", "content": (
                f"Abstracts:\n{format_abstracts(abstracts)}\n\n"
                f"Judge the claim using only the abstracts above, which were retrieved for these questions:\n{qs}\n\n"
                f"{LABEL_DEFINITIONS}\n\nClaim: {claim}\n\n"
                f"Cite the abstract ids and sentence numbers your verdict relies on (none for NEI). {CITE_RULE}\n"
                'Return JSON: {"verdict": "SUPPORTS|REFUTES|NEI", "citations": [{"doc": "A1", "sentences": [0]}], '
                '"justification": "<one or two sentences>"}')}]


# -- M1 Planner ----------------------------------------------------------------------------
PLANNER_VERSION = "sf-planner-v2"
PLANNER_SCHEMA = {"type": "object", "required": ["obligations"], "properties": {"obligations": {
    "type": "array", "minItems": 1, "maxItems": MAX_OBLIGATIONS, "items": {
        "type": "object", "required": ["type", "description", "expected_condition", "query"],
        "properties": {"type": {"enum": list(CORE_TYPES + QUALIFIER_TYPES)},
                       "description": {"type": "string", "minLength": 1},
                       "expected_condition": {"type": "string"}, "query": {"type": "string", "minLength": 1}}}}}}

# One worked example, claim 709 from the leak-free few-shot pool (proposal: few-shot examples from that pool).
PLANNER_EXAMPLE = (
    "Example\n"
    "Claim: Long chain polyunsaturated fatty acids supplementation is associated with lower rates of atopic eczema at 1 year.\n"
    '{"obligations": ['
    '{"type": "subject", "description": "the exposure is long chain polyunsaturated fatty acid (LCPUFA) supplementation", '
    '"expected_condition": "the study gives or measures LCPUFA supplementation", "query": "long chain polyunsaturated fatty acid supplementation"}, '
    '{"type": "outcome", "description": "the outcome is the rate of atopic eczema", '
    '"expected_condition": "the study measures atopic eczema rates", "query": "atopic eczema incidence"}, '
    '{"type": "relation", "description": "supplementation is associated with lower eczema rates", '
    '"expected_condition": "eczema rates are lower with supplementation than without", "query": "LCPUFA supplementation lowers atopic eczema"}, '
    '{"type": "timepoint", "description": "the effect is seen at 1 year of age", '
    '"expected_condition": "eczema is assessed at about 1 year", "query": "atopic eczema at 1 year"}]}')


def planner_messages(claim: str) -> list[dict]:
    return [_system("planner", "You decompose scientific claims into what evidence must establish."),
            {"role": "user", "content": (
                "List the atomic conditions that evidence must establish for the claim to be true. Each condition "
                "is one short statement, in words, about one part of the claim (never just the type name).\n"
                f"Core types (what the claim is about): {', '.join(CORE_TYPES)} "
                "(subject = intervention/exposure/entity; outcome = measured property or effect; "
                "relation = direction or kind of relation between them).\n"
                f"Qualifier types (restrictions): {', '.join(QUALIFIER_TYPES)}.\n"
                f"Use at most {MAX_OBLIGATIONS}. Include a qualifier only if the claim states it. "
                "Do not repeat a condition. Each needs a short search query.\n\n"
                f"{PLANNER_EXAMPLE}\n\nClaim: {claim}\n\n"
                'Return JSON: {"obligations": [{"type": "<type>", "description": "<the condition in words>", '
                '"expected_condition": "<what the evidence must show>", "query": "<search query>"}]}')}]


# -- M1 Auditor ----------------------------------------------------------------------------
AUDITOR_VERSION = "sf-auditor-v4"
AUDITOR_SCHEMA = {"type": "object",
                  "required": ["status", "citations", "rationale", "needs_more_retrieval"],
                  "properties": {"status": {"enum": list(STATUSES)}, "citations": _CITATIONS,
                                 "rationale": {"type": "string"}, "needs_more_retrieval": {"type": "boolean"}}}

STATUS_DEFINITIONS = (
    "Statuses:\n"
    "- SATISFIED: a cited sentence establishes the condition.\n"
    "- CONTRADICTED: a cited sentence states the opposite of the condition for the same subject (for example a "
    "decrease where the condition requires an increase). A different finding, a different topic or missing "
    "information is not a contradiction.\n"
    "- PARTIAL: the evidence addresses the condition but is incomplete or not specific enough.\n"
    "- UNSUPPORTED: the abstracts do not address the condition, or address something else.")


def auditor_messages(claim: str, description: str, expected: str, abstracts) -> list[dict]:
    return [_system("auditor", "You check one condition of a scientific claim against research abstracts."),
            {"role": "user", "content": (
                f"Abstracts:\n{format_abstracts(abstracts)}\n\n"
                f"Claim: {claim}\nCondition to check: {description}\nThe evidence must show: {expected}\n\n"
                f"{STATUS_DEFINITIONS}\n\n"
                "Judge only this condition, using only these abstracts. SATISFIED and CONTRADICTED must cite "
                f"the sentences that show it; if several abstracts show it, cite each of them. {CITE_RULE}\n"
                'Return JSON: {"status": "SATISFIED|CONTRADICTED|PARTIAL|UNSUPPORTED", '
                '"citations": [{"doc": "A1", "sentences": [0]}], "rationale": "<one sentence>", '
                '"needs_more_retrieval": true|false}')}]


# -- M1-untyped: FActScore-style atomic facts ----------------------------------------------
FACTS_VERSION = "sf-facts-v1"
FACTS_SCHEMA = {"type": "object", "required": ["facts"], "properties": {"facts": {
    "type": "array", "minItems": 1, "maxItems": MAX_OBLIGATIONS, "items": {
        "type": "object", "required": ["fact", "query"],
        "properties": {"fact": {"type": "string", "minLength": 1}, "query": {"type": "string", "minLength": 1}}}}}}


def facts_messages(claim: str) -> list[dict]:
    return [_system("facts", "You break scientific claims into atomic facts."),
            {"role": "user", "content": (
                f"Break the claim into at most {MAX_OBLIGATIONS} atomic facts, each a short standalone statement, "
                "with a short search query for each. Use as few as the claim needs.\n\n"
                f"Claim: {claim}\n\n"
                'Return JSON: {"facts": [{"fact": "<statement>", "query": "<search query>"}]}')}]


# -- M2: targeted follow-up query for one unresolved obligation ------------------------------
REQUERY_VERSION = "sf-requery-v1"
REQUERY_SCHEMA = {"type": "object", "required": ["missing", "query"],
                  "properties": {"missing": {"type": "string"}, "query": {"type": "string", "minLength": 1}}}


def requery_messages(claim: str, description: str, expected: str, status: str, rationale: str) -> list[dict]:
    return [_system("requery", "You plan literature searches for scientific fact-checking."),
            {"role": "user", "content": (
                f"Claim: {claim}\nCondition: {description}\nThe evidence must show: {expected}\n"
                f"Current status: {status}\nAuditor's note: {rationale or '(none)'}\n\n"
                "State what information is still missing, and write one new search query for it.\n"
                'Return JSON: {"missing": "<what is missing>", "query": "<search query>"}')}]


# -- M2-claimiter: claim-level follow-up query (FIRE/QACheck-style) --------------------------
CLAIM_REQUERY_VERSION = "sf-claim-requery-v1"


def claim_requery_messages(claim: str, justification: str) -> list[dict]:
    return [_system("claim_requery", "You plan literature searches for scientific fact-checking."),
            {"role": "user", "content": (
                f"Claim: {claim}\nThe abstracts read so far were not enough to decide. "
                f"Note: {justification or '(none)'}\n\n"
                "State what information is still missing, and write one new search query for it.\n"
                'Return JSON: {"missing": "<what is missing>", "query": "<search query>"}')}]


B1_ITER_VERSION = "sf-b1-iter-v4"   # same wording as B1; separate version keeps caches apart


# -- M1-LLMagg: LLM verdict over the ledger --------------------------------------------------
AGGREGATE_VERSION = "sf-aggregate-v2"


def aggregate_messages(claim: str, rows: list[dict]) -> list[dict]:
    lines = []
    for r in rows:
        ev = "; ".join(f'"{e}"' for e in r["evidence"]) or "none"
        lines.append(f"- [{r['criticality']}] {r['description']}: {r['status']}. {r['rationale']} Evidence: {ev}")
    return [_system("aggregate", "You verify scientific claims from an evidence audit."),
            {"role": "user", "content": (
                "Each condition of the claim was checked separately against research abstracts:\n"
                + "\n".join(lines) + f"\n\n{LABEL_DEFINITIONS}\n\nClaim: {claim}\n\n"
                'Decide the verdict. Return JSON: {"verdict": "SUPPORTS|REFUTES|NEI", '
                '"justification": "<one or two sentences>"}')}]
