import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json  # noqa: E402

import pytest  # noqa: E402

from oblig.data.scifact import (DEV_SPLITS, SPLITS_FILE, claim_label, held_out_abstracts, leak_free_train,  # noqa: E402
                                load_claims, load_corpus, load_eval_set, load_scifact_open, load_split)
from oblig.retrieval.bm25 import BM25  # noqa: E402
from oblig.retrieval.hybrid import HybridRetriever, rrf  # noqa: E402


def test_split_labels():
    assert Counter(c.label for c in load_claims("train")) == {"SUPPORTS": 332, "NEI": 304, "REFUTES": 173}
    assert Counter(c.label for c in load_claims("dev")) == {"SUPPORTS": 124, "NEI": 112, "REFUTES": 64}
    assert all(c.label is None for c in load_claims("test"))


def test_claim_label_rule():
    assert claim_label({}) == "NEI"
    assert claim_label({"1": [{"sentences": [0], "label": "CONTRADICT"}]}) == "REFUTES"


def test_development_splits_are_leak_free_and_frozen():
    """Proposal D9: prompts are developed only on train claims sharing no abstract with a held-out claim."""
    train, dev = load_claims("train"), load_claims("dev")
    pool = {c.id for c in leak_free_train(train, dev)}
    frozen = json.loads(SPLITS_FILE.read_text())
    prompt_dev, fewshot = set(frozen["prompt_dev"]), set(frozen["fewshot_pool"])
    assert len(pool) == 385 and prompt_dev | fewshot == pool and not prompt_dev & fewshot
    assert (len(prompt_dev), len(fewshot)) == (199, 186)
    held = held_out_abstracts(dev)
    for name in DEV_SPLITS:
        claims = load_split(name)
        assert {c.id for c in claims} == set(frozen[name])
        assert not any(set(c.cited_doc_ids) & held for c in claims)      # no shared abstract, so no twins
    assert not prompt_dev & {c.id for c in dev}
    with pytest.raises(ValueError):
        load_split("train_dev")


def test_scifact_open_test_set_in_5k_setting():
    corpus = load_corpus()
    claims, conflicting = load_scifact_open(corpus.keys())
    assert len(claims) == 278 and conflicting == [158]
    assert Counter(c.label for c in claims) == {"SUPPORTS": 97, "REFUTES": 96, "NEI": 85}
    test_text = {c.id: c.claim for c in load_claims("test")}
    assert all(test_text[c.id] == c.claim for c in claims)             # labels for the hidden test claims
    pooled = 0
    for c in claims:
        assert c.dataset == "scifact_open" and c.merged_rationales
        for doc, sents in c.evidence.items():
            assert doc in corpus and all(0 <= i < len(corpus[doc].sentences) for i in sents)
            assert c.has_gold_rationale(doc) == bool(sents)            # pooled evidence: label, no gold sentences
            pooled += not c.has_gold_rationale(doc)
    assert pooled == 24
    assert len(load_eval_set("scifact_open", corpus.keys())) == 278 and len(load_eval_set("prompt_dev")) == 199


def test_gold_rationales_resolve_to_corpus_sentences():
    corpus = load_corpus()
    assert len(corpus) == 5183
    for c in load_claims("dev"):
        for doc, sents in c.evidence.items():
            assert doc in corpus
            assert all(0 <= i < len(corpus[doc].sentences) for i in sents)


def test_rrf_fuses_and_is_deterministic():
    fused = rrf([[("a", 9), ("b", 5)], [("b", 0.9), ("c", 0.8)]], k=60)
    assert fused[0][0] == "b" and {d for d, _ in fused} == {"a", "b", "c"}
    assert rrf([[("x", 1)], [("y", 1)]]) == [("x", 1 / 61), ("y", 1 / 61)]


def test_hybrid_with_bm25_only_matches_bm25_order():
    texts, ids = ["aspirin lowers stroke risk", "statins lower cholesterol", "aspirin and bleeding"], ["1", "2", "3"]
    bm = BM25().fit(texts, ids)
    h = HybridRetriever([bm], depth=3)
    assert [d for d, _ in h.search("aspirin stroke", 2)] == [d for d, _ in bm.search("aspirin stroke", 2)]
