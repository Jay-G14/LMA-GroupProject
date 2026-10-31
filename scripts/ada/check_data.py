"""Integrity check run by the cluster job before anything expensive: fail in seconds on partial data."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {"raw/scifact/data/corpus.jsonl": 5183, "raw/scifact/data/claims_train.jsonl": 809,
            "raw/scifact/data/claims_dev.jsonl": 300, "raw/scifact/data/claims_test.jsonl": 300,
            "raw/scifact_open/claims.jsonl": 279, "raw/scifact_open/claims_metadata.jsonl": 279}
for rel, n in expected.items():
    path = root / rel
    if not path.exists():
        sys.exit(f"ERROR: missing {path}")
    lines = sum(1 for l in open(path, encoding="utf-8") if l.strip())
    if lines != n:
        sys.exit(f"ERROR: {path} has {lines} lines, expected {n}")
splits = json.loads((root / "splits/scifact_leakfree_v1.json").read_text())
if (len(splits["prompt_dev"]), len(splits["fewshot_pool"])) != (199, 186):
    sys.exit("ERROR: leak-free split file has unexpected sizes")
print("data integrity check passed:", ", ".join(f"{Path(k).name}={v}" for k, v in expected.items()))
