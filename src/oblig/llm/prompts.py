"""Versioned prompts and output schemas for the PUBHEALTH baselines.

Bump the version string whenever a prompt's wording changes: it is part of every
cache key and prediction record.
"""
from __future__ import annotations

from ..data.pubhealth import LABELS

LABEL_DEFINITIONS = (
    "Labels:\n"
    "- true: the claim is accurate.\n"
    "- false: the claim is inaccurate.\n"
    "- mixture: the claim has significant elements of both truth and falsity.\n"
    "- unproven: there is insufficient evidence to establish whether the claim is true or false."
)

SYSTEM = "You are a careful fact-checker of public-health claims. Answer with a single JSON object only."

B0_VERSION = "b0-v1"
B0_SCHEMA = {
    "type": "object",
    "required": ["verdict", "justification"],
    "properties": {"verdict": {"enum": list(LABELS)}, "justification": {"type": "string"}},
}


def b0_messages(claim: str) -> list[dict]:
    user = (f"Classify the claim using your own knowledge.\n{LABEL_DEFINITIONS}\n\n"
            f"Claim: {claim}\n\n"
            'Return JSON: {"verdict": "true|false|mixture|unproven", "justification": "<one or two sentences>"}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


B1_VERSION = "b1-v2"
B1_SCHEMA = {
    "type": "object",
    "required": ["verdict", "cited_passages", "justification"],
    "properties": {"verdict": {"enum": list(LABELS)},
                   "cited_passages": {"type": "array", "items": {"type": "string"}},
                   "justification": {"type": "string"}},
}


def alias_passages(passages: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Replace corpus ids with per-prompt aliases E1..Ek so no identifier reaches the model."""
    aliased = [(f"E{i + 1}", text) for i, (_, text) in enumerate(passages)]
    return aliased, {f"E{i + 1}": pid for i, (pid, _) in enumerate(passages)}


def format_evidence(passages: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{alias}]\n{text}" for alias, text in passages)


def b1_messages(claim: str, passages: list[tuple[str, str]]) -> list[dict]:
    """passages: (alias, text) pairs from alias_passages."""
    user = (f"Classify the claim using only the evidence passages below.\n{LABEL_DEFINITIONS}\n\n"
            f"Evidence:\n{format_evidence(passages)}\n\nClaim: {claim}\n\n"
            "Cite the ids (in square brackets above) of the passages your verdict relies on.\n"
            'Return JSON: {"verdict": "true|false|mixture|unproven", "cited_passages": ["<id>", ...], '
            '"justification": "<one or two sentences>"}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
