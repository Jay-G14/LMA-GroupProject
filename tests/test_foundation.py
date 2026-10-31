import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.masking import is_verdict_sentence, mask_sentences, split_sentences  # noqa: E402
from oblig.data.pubhealth import LABELS, load_all, model_input  # noqa: E402
from oblig.evaluation.classification import INVALID, classification_report, paired_bootstrap  # noqa: E402
from oblig.llm.client import LLMClient, LLMConfig  # noqa: E402
from oblig.llm.prompts import B1_SCHEMA, alias_passages, b1_messages  # noqa: E402
from oblig.retrieval.bm25 import BM25  # noqa: E402
from oblig.retrieval.corpus import chunk_document  # noqa: E402
from oblig.utils.cache import cache_key  # noqa: E402


@pytest.fixture(scope="module")
def splits():
    return load_all()


def test_split_sizes_and_labels(splits):
    assert {s: len(sp.data) for s, sp in splits.items()} == {"train": 9804, "dev": 1214, "test": 1233}
    assert {s: sp.n_malformed for s, sp in splits.items()} == {"train": 28, "dev": 7, "test": 2}
    for sp in splits.values():
        assert set(sp.data.label) == set(LABELS)
        assert sp.data.example_id.is_unique


def test_example_ids_do_not_leak_claim_id_or_split(splits):
    for name, sp in splits.items():
        for ex, cid in zip(sp.data.example_id.head(200), sp.data.claim_id.head(200)):
            assert cid not in ex and name not in ex


def test_model_input_has_claim_only(splits):
    r = splits["dev"].data.iloc[0]
    assert set(model_input(r)) == {"example_id", "claim"}


def test_verdict_masking():
    sents = split_sentences("Smith said X. We rate the claim Mostly False. The study had 40 people.")
    assert len(sents) == 3
    kept, removed = mask_sentences(sents)
    assert removed == [1] and kept == [sents[0], sents[2]]
    assert is_verdict_sentence("The statement contains an element of truth but ignores critical facts.")
    assert not is_verdict_sentence("Vaccines were tested on 40,000 people.")


def test_chunking_ids_and_coverage():
    ps, st = chunk_document("d1", "dev", "A one. B two. C three. D four. E five.", sents_per_passage=2)
    assert [p.passage_id for p in ps] == ["d1::p0", "d1::p1", "d1::p2"]
    assert sum(len(p.sentence_ids) for p in ps) == st["n_sentences"] == 5


def test_bm25_deterministic_and_relevant():
    texts = ["vitamin d prevents flu", "masks reduce covid spread", "vitamin c and colds"]
    bm = BM25().fit(texts, ["a", "b", "c"])
    r1, r2 = bm.search("does vitamin d prevent flu", 3), bm.search("does vitamin d prevent flu", 3)
    assert r1 == r2 and r1[0][0] == "a"
    assert bm.search("vitamin", 3, exclude={"a"})[0][0] == "c"


def test_citation_aliases_hide_ids():
    ev, alias = alias_passages([("phabc::p3", "text one"), ("phdef::p0", "text two")])
    msg = b1_messages("claim", ev)[-1]["content"]
    assert "phabc" not in msg and "[E1]" in msg and alias == {"E1": "phabc::p3", "E2": "phdef::p0"}


def test_llm_errors_are_recorded_not_raised():
    bad = LLMClient(LLMConfig(backend="mock", extra={"mock_fn": lambda m: "not json"}))
    r = bad.generate_json([{"role": "user", "content": "x"}], B1_SCHEMA, "t")
    assert r.parsed is None and r.error.startswith("parse_error")
    off = LLMClient(LLMConfig(backend="mock", extra={"mock_fn": lambda m: json.dumps({"verdict": "maybe"})}))
    assert off.generate_json([], B1_SCHEMA, "t").error.startswith("schema_error")


def test_cache_key_changes_with_config():
    assert cache_key(model="a", prompt="p") != cache_key(model="b", prompt="p")
    assert cache_key(model="a", prompt="p") == cache_key(prompt="p", model="a")


def test_metrics_and_bootstrap():
    gold = ["true", "false", "mixture", "unproven"] * 5
    rep = classification_report(gold, gold[:-1] + [INVALID])
    assert rep["invalid_predictions"] == 1 and rep["accuracy"] == pytest.approx(0.95)
    bs = paired_bootstrap(gold, ["true"] * 20, gold, n_boot=200)
    assert bs["delta"] > 0 and bs["ci95"][0] > 0
