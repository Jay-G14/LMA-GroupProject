"""SciFact loading, claim-level labels, the SciFact-Open test set and leak-free development splits.

Label space is SUPPORTS / REFUTES / NEI and is never merged with PUBHEALTH's.

Held-out evaluation sets (proposal Section 9):
  * claims_dev.jsonl (300 claims); claims_test.jsonl is unlabelled.
  * SciFact-Open's labels for the hidden SciFact test claims, restricted to the 5,183-abstract corpus
    (278 claims after excluding one with conflicting labels).

Development data: SciFact's splits overlap (dev claims often have a near-copy in train with the opposite
label), and the shipped cross_validation folds mix train and dev claims, so neither is used. Prompt
development uses only train claims that share no abstract with any held-out claim (design choice D9),
frozen in data/splits/scifact_leakfree_v1.json: 199 prompt-development claims and 186 few-shot exemplars.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

LABELS = ("SUPPORTS", "REFUTES", "NEI")
_EVIDENCE_LABEL = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"}
# OBLIG_DATA_DIR lets a cluster job read a copy of data/ staged on node-local storage.
_DATA = Path(os.environ.get("OBLIG_DATA_DIR") or Path(__file__).resolve().parents[3] / "data")
SPLITS_FILE = _DATA / "splits" / "scifact_leakfree_v1.json"
DEV_SPLITS = ("prompt_dev", "fewshot_pool")


def default_root() -> Path:
    return _DATA / "raw" / "scifact" / "data"


def default_open_root() -> Path:
    return _DATA / "raw" / "scifact_open"


@dataclass(frozen=True)
class Abstract:
    doc_id: str
    title: str
    sentences: list[str]
    structured: bool


@dataclass
class Claim:
    id: int
    claim: str
    label: str | None                                    # None for unlabelled test claims
    evidence: dict[str, list[int]] = field(default_factory=dict)          # doc_id -> rationale sentence ids
    evidence_sets: dict[str, list[list[int]]] = field(default_factory=dict)  # doc_id -> rationale sets
    doc_labels: dict[str, str] = field(default_factory=dict)              # doc_id -> SUPPORTS/REFUTES
    cited_doc_ids: list[str] = field(default_factory=list)
    dataset: str = "scifact"
    # Evidence documents whose rationale sentences are gold (expert-annotated); None = all of them.
    # SciFact-Open's pooled evidence has a human label but model-predicted rationale sentences.
    rationale_docs: list[str] | None = None
    # SciFact-Open merges a document's alternative rationale sets into one sentence list.
    merged_rationales: bool = False

    def has_gold_rationale(self, doc: str) -> bool:
        return doc in self.evidence and (self.rationale_docs is None or doc in self.rationale_docs)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_corpus(root: Path | None = None) -> dict[str, Abstract]:
    rows = _read_jsonl((root or default_root()) / "corpus.jsonl")
    return {str(r["doc_id"]): Abstract(str(r["doc_id"]), r["title"].strip(),
                                       [s.strip() for s in r["abstract"]], bool(r["structured"])) for r in rows}


def claim_label(evidence: dict) -> str | None:
    """Claim-level label: NEI without evidence, else the (single) evidence label."""
    labels = {_EVIDENCE_LABEL[e["label"]] for ev in evidence.values() for e in ev}
    if not labels:
        return "NEI"
    if len(labels) > 1:
        raise ValueError("claim has both supporting and refuting evidence")
    return labels.pop()


def load_claims(split: str, root: Path | None = None) -> list[Claim]:
    out = []
    for r in _read_jsonl((root or default_root()) / f"claims_{split}.jsonl"):
        ev = r.get("evidence")
        c = Claim(r["id"], r["claim"].strip(), None if ev is None else claim_label(ev),
                  cited_doc_ids=[str(d) for d in r.get("cited_doc_ids", [])])
        for doc, sets in (ev or {}).items():
            c.evidence_sets[doc] = [sorted(s["sentences"]) for s in sets]
            c.evidence[doc] = sorted({i for s in sets for i in s["sentences"]})
            c.doc_labels[doc] = _EVIDENCE_LABEL[sets[0]["label"]]
        out.append(c)
    return out


def load_scifact_open(corpus_ids, root: Path | None = None) -> tuple[list[Claim], list[int]]:
    """SciFact-Open labels for the hidden SciFact test claims in the 5,183-abstract setting.

    Every human-labelled evidence abstract inside the SciFact corpus counts for the label (citation and
    pooled provenance); a claim without any is NEI. Claims whose in-corpus evidence carries both labels
    are excluded and returned separately. Gold rationale sentences exist only for citation evidence.
    Returns (claims, ids of excluded conflicting claims)."""
    root = root or default_open_root()
    corpus_ids = set(corpus_ids)
    meta = {r["id"]: r for r in _read_jsonl(root / "claims_metadata.jsonl")}
    claims, conflicting = [], []
    for r in _read_jsonl(root / "claims.jsonl"):
        ev = {str(d): e for d, e in r["evidence"].items() if str(d) in corpus_ids}
        labels = {_EVIDENCE_LABEL[e["label"]] for e in ev.values()}
        if len(labels) > 1:
            conflicting.append(r["id"])
            continue
        gold_docs = sorted(d for d, e in ev.items() if e["provenance"] == "citation" and e["sentences"])
        c = Claim(r["id"], r["claim"].strip(), labels.pop() if labels else "NEI",
                  cited_doc_ids=[str(d) for d in meta.get(r["id"], {}).get("cited_doc_ids", [])],
                  dataset="scifact_open", rationale_docs=gold_docs, merged_rationales=True)
        for d, e in ev.items():
            gold = sorted(e["sentences"]) if d in gold_docs else []
            c.doc_labels[d], c.evidence[d], c.evidence_sets[d] = _EVIDENCE_LABEL[e["label"]], gold, [gold] if gold else []
        claims.append(c)
    return claims, conflicting


# -- leak-free development data -------------------------------------------------------------
def held_out_abstracts(dev: list[Claim], open_root: Path | None = None) -> set[str]:
    """Abstracts cited by claims_dev claims, or cited by / evidence for SciFact-Open test claims."""
    root = open_root or default_open_root()
    docs = {d for c in dev for d in c.cited_doc_ids}
    for r in _read_jsonl(root / "claims_metadata.jsonl"):
        docs |= {str(d) for d in r.get("cited_doc_ids", [])}
    for r in _read_jsonl(root / "claims.jsonl"):
        docs |= {str(d) for d in r["evidence"]}
    return docs


def leak_free_train(train: list[Claim], dev: list[Claim], open_root: Path | None = None) -> list[Claim]:
    """Train claims that share no cited abstract with any held-out claim (the 385-claim pool)."""
    held = held_out_abstracts(dev, open_root)
    return [c for c in train if not set(c.cited_doc_ids) & held]


def load_split(name: str, root: Path | None = None) -> list[Claim]:
    """A frozen development split of claims_train: 'prompt_dev' (199) or 'fewshot_pool' (186)."""
    if name not in DEV_SPLITS:
        raise ValueError(f"unknown development split {name!r}; choose from {DEV_SPLITS}")
    ids = set(json.loads(SPLITS_FILE.read_text(encoding="utf-8"))[name])
    return [c for c in load_claims("train", root) if c.id in ids]


def load_eval_set(name: str, corpus_ids=None) -> list[Claim]:
    """Development splits ('prompt_dev', 'fewshot_pool') or held-out sets ('dev', 'scifact_open')."""
    if name in DEV_SPLITS:
        return load_split(name)
    if name == "dev":
        return load_claims("dev")
    if name == "scifact_open":
        return load_scifact_open(corpus_ids if corpus_ids is not None else load_corpus().keys())[0]
    raise ValueError(f"unknown evaluation set {name!r}")


HELD_OUT_SETS = ("dev", "scifact_open")
