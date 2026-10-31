"""Sentence splitting and verdict masking for PUBHEALTH article text.

Fact-check articles may state their own verdict ("We rate it ...", "Our ruling",
PolitiFact rating definitions). Such sentences are removed from the evidence
corpus; every other sentence is kept so legitimate evidence is preserved.
"""
from __future__ import annotations

import re

from .leakage import VERDICT_PATTERNS
from .pubhealth import normalise_ws

_SENT = re.compile(r"(?<=[.!?])[\"'”’)]?\s+(?=[\"'“‘(]?[A-Z0-9])")

# PolitiFact Truth-O-Meter definitions that survive after the rating word was stripped.
RATING_DEFINITIONS = [
    r"the statement is accurate and there.s nothing significant missing",
    r"the statement is accurate but needs clarification or additional information",
    r"the statement is partially accurate but leaves out important details",
    r"the statement contains an element of truth but ignores critical facts",
    r"the statement is not accurate",
    r"the statement is not accurate and makes a ridiculous claim",
]
_MASK = re.compile("|".join(f"(?:{p})" for p in list(VERDICT_PATTERNS.values()) + RATING_DEFINITIONS),
                   re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    text = normalise_ws(text)
    return [s.strip() for s in _SENT.split(text) if s.strip()] if text else []


def is_verdict_sentence(sentence: str) -> bool:
    return bool(_MASK.search(sentence))


def mask_sentences(sentences: list[str]) -> tuple[list[str], list[int]]:
    """Return kept sentences and the original indices of removed ones."""
    kept, removed = [], []
    for i, s in enumerate(sentences):
        if is_verdict_sentence(s):
            removed.append(i)
        else:
            kept.append(s)
    return kept, removed
