"""Score a filled annotation set (see scripts/export_annotation.py): inter-annotator agreement (Cohen's kappa)
and the model's obligation-status accuracy against the gold labels. Writes annotation/<name>/scores.json.

Usage:
  python scripts/score_annotation.py --name dev_m1_50
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.evaluation.annotation import score  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    d = ROOT / "annotation" / args.name
    key = json.loads((d / "key.json").read_text(encoding="utf-8"))
    result = {"run": key["run"], **score(read_csv(d / "sheet.csv"), read_csv(d / "claims.csv"), key["items"])}
    (d / "scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
