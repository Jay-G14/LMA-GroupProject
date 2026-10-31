"""PUBHEALTH loading, validation and label normalisation.

The raw TSVs are read verbatim and never modified. Rows broken by embedded
newlines/quotes in the original release (mostly TruthOrFiction records) are not
repaired: they are flagged as malformed and excluded from every split, and the
exclusion counts are reported by the dataset audit.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SPLITS = ("train", "dev", "test")
COLUMNS = ["claim_id", "claim", "date_published", "explanation", "fact_checkers",
           "main_text", "sources", "label", "subjects"]

# Canonical four-class label space. The raw files use "mixture", not "mixed".
LABELS = ("true", "false", "mixture", "unproven")
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# Fields a verifier may never see at inference time (gold or gold-derived).
GOLD_FIELDS = ("label", "explanation")
# Metadata that is not evidence; excluded from model inputs by default.
METADATA_FIELDS = ("fact_checkers", "subjects", "date_published", "sources")

_WS = re.compile(r"\s+")


def default_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "PUBHEALTH"


def normalise_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def split_list(field: str) -> list[str]:
    """Split comma-separated fact_checkers/subjects/sources into clean items."""
    return [x.strip() for x in field.split(",") if x.strip()]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def opaque_id(split: str, claim_id: str, row: int) -> str:
    return "ph" + hashlib.sha1(f"{split}|{claim_id}|{row}".encode()).hexdigest()[:10]


def read_raw(split: str, root: Path | None = None) -> pd.DataFrame:
    """Read a split exactly as pandas parses it, all fields as strings."""
    path = (root or default_root()) / f"{split}.tsv"
    df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False)
    # test.tsv carries an extra leading pandas index column.
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    assert list(df.columns) == COLUMNS, f"unexpected schema in {path}: {list(df.columns)}"
    df.insert(0, "row", range(len(df)))
    return df


def validity_mask(df: pd.DataFrame) -> pd.Series:
    """A row is valid if it has a numeric id, a non-empty claim and a known label."""
    return (df["claim_id"].str.fullmatch(r"\d+")
            & (df["claim"].str.strip() != "")
            & df["label"].str.strip().str.lower().isin(LABELS))


@dataclass
class Split:
    name: str
    raw: pd.DataFrame       # every parsed row
    data: pd.DataFrame      # valid rows only, with derived columns

    @property
    def n_malformed(self) -> int:
        return len(self.raw) - len(self.data)


def load_split(split: str, root: Path | None = None) -> Split:
    raw = read_raw(split, root)
    df = raw[validity_mask(raw)].copy()
    df["label"] = df["label"].str.strip().str.lower()
    df["label_id"] = df["label"].map(LABEL2ID)
    df["claim"] = df["claim"].map(normalise_ws)
    # Stable, opaque example id. claim_id ranges encode source site and label (see
    # reports/dataset_audit.md), so neither claim_id nor the split may appear in an id.
    df["example_id"] = [opaque_id(split, cid, r) for cid, r in zip(df["claim_id"], df["row"])]
    df["split"] = split
    return Split(split, raw, df.reset_index(drop=True))


def load_all(root: Path | None = None) -> dict[str, Split]:
    return {s: load_split(s, root) for s in SPLITS}


def model_input(record: dict | pd.Series) -> dict:
    """The only fields a verifier receives by default: the claim to verify."""
    return {"example_id": record["example_id"], "claim": record["claim"]}
