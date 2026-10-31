"""Deterministic role-routed mock for SciFact pipeline tests. Its outputs exercise
every code path (statuses, citations, pooling); its scores mean nothing."""
from __future__ import annotations

import json
import re

_WORD = re.compile(r"[a-z0-9]{4,}")


def _abstracts(text: str) -> list[tuple[str, list[str]]]:
    out = []
    for block in re.split(r"\n\n(?=\[A\d+\])", text.split("Abstracts:\n", 1)[-1]):
        m = re.match(r"\[(A\d+)\]", block)
        if m:
            out.append((m.group(1), re.findall(r"^\s+\(\d+\) (.*)$", block, flags=re.MULTILINE)))
    return out


def _field(text: str, name: str) -> str:
    """Last occurrence: prompts may show a worked example (e.g. the planner's) before the real input."""
    found = re.findall(rf"^{name}: (.*)$", text, flags=re.MULTILINE)
    return found[-1] if found else ""


def mock_scifact(messages: list[dict]) -> str:
    role = messages[0]["content"].split("\n", 1)[0].removeprefix("ROLE: ")
    user = messages[-1]["content"]
    claim = _field(user, "Claim")
    words = claim.rstrip(".").split()
    if role == "planner":
        half = max(1, len(words) // 2)
        return json.dumps({"obligations": [
            {"description": " ".join(words[:half]), "type": "subject", "expected_condition": "addressed",
             "query": " ".join(words[:half])},
            {"description": " ".join(words[half:]) or claim, "type": "outcome", "expected_condition": "measured",
             "query": " ".join(words[half:]) or claim},
            {"description": f"direction: {claim}", "type": "relation", "expected_condition": "as claimed",
             "query": claim}]})
    if role == "facts":
        half = max(1, len(words) // 2)
        return json.dumps({"facts": [{"fact": " ".join(words[:half]), "query": " ".join(words[:half])},
                                     {"fact": claim, "query": claim}]})
    if role in ("requery", "claim_requery"):
        cond = _field(user, "Condition") or claim
        return json.dumps({"missing": "mock", "query": " ".join(cond.split()[-4:])})
    if role == "aggregate":
        statuses = re.findall(r"^- \[(?:core|qualifier)\] .*?: (SATISFIED|CONTRADICTED|PARTIAL|UNSUPPORTED)\.",
                              user, flags=re.MULTILINE)
        verdict = ("REFUTES" if "CONTRADICTED" in statuses else
                   "SUPPORTS" if statuses and all(x == "SATISFIED" for x in statuses) else "NEI")
        return json.dumps({"verdict": verdict, "justification": "mock"})
    if role == "subquestions":
        return json.dumps({"questions": [f"What is known about {' '.join(words[:4])}?", f"Is it true that {claim}"]})
    abstracts = _abstracts(user)
    target = set(_WORD.findall((_field(user, "Condition to check") or claim).lower()))
    best = max(((len(target & set(_WORD.findall(s.lower()))), a, i)
                for a, sents in abstracts for i, s in enumerate(sents)), default=(0, None, None))
    if role == "auditor":
        if best[0] >= 2:
            return json.dumps({"status": "SATISFIED", "citations": [{"doc": best[1], "sentences": [best[2]]}],
                               "rationale": "mock overlap", "needs_more_retrieval": False})
        status = "PARTIAL" if best[0] == 1 else "UNSUPPORTED"
        return json.dumps({"status": status, "citations": [], "rationale": "mock", "needs_more_retrieval": True})
    if best[0] >= 3:
        return json.dumps({"verdict": "SUPPORTS", "citations": [{"doc": best[1], "sentences": [best[2]]}],
                           "justification": "mock overlap"})
    return json.dumps({"verdict": "NEI", "citations": [], "justification": "mock"})
