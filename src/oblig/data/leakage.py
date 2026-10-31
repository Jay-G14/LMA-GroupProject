"""Split-overlap and answer-leakage checks for PUBHEALTH."""
from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .pubhealth import normalise_ws, split_list

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")

# Phrases by which a fact-check article can state its own verdict. Matched
# case-insensitively against main_text; used both to measure and to mask.
VERDICT_PATTERNS = {
    "we_rate": r"\bwe rate (?:this|the|it|that|his|her|their)\b",
    "our_ruling": r"\bour ruling\b",
    "rating_label": r"\brating:\s",
    "truth_o_meter": r"\btruth-o-meter\b",
    "pants_on_fire": r"\bpants on fire\b",
    "half_true": r"\bhalf[- ]true\b",
    "mostly_tf": r"\bmostly (?:true|false)\b",
    "claim_is_tf": r"\b(?:this|the) (?:claim|rumor|story|post|statement) is (?:false|true|untrue|unproven|a mixture)\b",
    "verdict_word": r"\bverdict\b",
}


def norm_claim(text: str) -> str:
    return normalise_ws(_NON_ALNUM.sub(" ", text.lower()))


def cross_split_exact(frames: dict[str, pd.DataFrame], key) -> dict[str, int]:
    """Number of rows in the second split whose key also occurs in the first."""
    out = {}
    keys = {s: set(key(df)) - {""} for s, df in frames.items()}
    for a, b in combinations(frames, 2):
        out[f"{a}&{b}"] = len(keys[a] & keys[b])
    return out


def near_duplicate_pairs(frames: dict[str, pd.DataFrame], threshold: float = 0.9,
                         field: str = "claim", chunk: int = 2000) -> pd.DataFrame:
    """Cross-split pairs with TF-IDF (word 1-2gram) cosine >= threshold."""
    names = list(frames)
    texts = pd.concat([frames[s][field] for s in names], ignore_index=True)
    owner = np.concatenate([[s] * len(frames[s]) for s in names])
    ids = pd.concat([frames[s]["example_id"] for s in names], ignore_index=True).to_numpy()
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(texts)
    rows = []
    for start in range(0, X.shape[0], chunk):
        sims = (X[start:start + chunk] @ X.T).tocoo()
        for i, j, v in zip(sims.row + start, sims.col, sims.data):
            if j > i and v >= threshold and owner[i] != owner[j]:
                rows.append((ids[i], owner[i], ids[j], owner[j], float(v)))
    return pd.DataFrame(rows, columns=["id_a", "split_a", "id_b", "split_b", "cosine"])


def verdict_phrase_rates(df: pd.DataFrame, field: str = "main_text") -> dict[str, float]:
    text = df[field].str.lower()
    rates = {k: float(text.str.contains(p, regex=True).mean()) for k, p in VERDICT_PATTERNS.items()}
    any_pat = "|".join(f"(?:{p})" for p in VERDICT_PATTERNS.values())
    rates["any"] = float(text.str.contains(any_pat, regex=True).mean())
    return rates


def explanation_in_text(df: pd.DataFrame) -> pd.Series:
    """True where the (whitespace-normalised) explanation occurs verbatim in main_text."""
    return pd.Series([len(e) > 30 and normalise_ws(e) in normalise_ws(t)
                      for e, t in zip(df["explanation"], df["main_text"])], index=df.index)


def explanation_is_lede(df: pd.DataFrame, n_chars: int = 120) -> pd.Series:
    """True where main_text starts with the explanation (typical of wire-news records)."""
    out = []
    for e, t in zip(df["explanation"], df["main_text"]):
        e, t = normalise_ws(e), normalise_ws(t)
        out.append(len(e) > 30 and t.startswith(e[:n_chars]))
    return pd.Series(out, index=df.index)


def source_url_overlap(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    return cross_split_exact(frames, lambda d: [u for s in d["sources"] for u in split_list(s)])
