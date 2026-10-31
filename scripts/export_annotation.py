"""Export a blind obligation-status annotation set from a ledger run (proposal: 50 SciFact dev claims, M1).

Writes annotation/<name>/: sheet.csv (one row per obligation), claims.csv (coverage), packet.md (claims,
obligations and the abstracts the Auditor saw), key.json (model statuses; do not open before labelling)
and README.md (instructions). Two annotators fill in their columns independently; disagreements are then
adjudicated in adjudicated_status / adjudicated_covers. Score with scripts/score_annotation.py.

Usage:
  python scripts/export_annotation.py --run dev_m1 --n 50 --name dev_m1_50 --results-dir to_transfer/dev
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.scifact import load_corpus  # noqa: E402
from oblig.evaluation.annotation import CLAIM_FIELDS, SHEET_FIELDS, build_items, sample_records  # noqa: E402
from oblig.llm.prompts_scifact import STATUS_DEFINITIONS  # noqa: E402

README = """# Obligation-status annotation ({name})

Source run: `{run}` ({system}, {eval_set}); {n_claims} claims, {n_items} obligations; sample seed {seed}.

Two annotators work **independently** and do not open `key.json` (the model's answers) until scoring.

1. Open `packet.md`. For each claim, read the abstracts listed under it.
2. In `sheet.csv`, fill your column (`annotator_A_status` or `annotator_B_status`) for every obligation with
   its true status, judged **only on the abstracts that obligation was audited on** (column `audited_on`):

{statuses}

3. In `claims.csv`, fill `annotator_A_covers` / `annotator_B_covers` with `yes` if the obligations together
   cover every condition the claim states, else `no`.
4. Afterwards, discuss disagreements and fill `adjudicated_status` / `adjudicated_covers` where you differed.
5. Run `python scripts/score_annotation.py --name {name}`.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="experiment id of a ledger system (M1, M2, ...)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--name", required=True)
    ap.add_argument("--results-dir", default=str(ROOT / "results"), help="e.g. to_transfer/dev for cluster runs")
    args = ap.parse_args()
    pred = Path(args.results_dir) / "predictions" / f"{args.run}.jsonl"
    cfg = json.loads(pred.with_suffix(".config.json").read_text(encoding="utf-8"))
    with open(pred, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    if not any("ledger" in r for r in records):
        sys.exit(f"{args.run} has no ledgers; export from an obligation system")
    chosen = sample_records(records, args.n, args.seed)
    corpus = load_corpus()
    sheet, claim_rows, key, packet = [], [], {}, [f"# Annotation packet: {args.name}", "", STATUS_DEFINITIONS, ""]
    for r in chosen:
        rows, claim_row, k, md = build_items(r, corpus)
        sheet += rows
        claim_rows.append(claim_row)
        key.update(k)
        packet.append(md)
    out = ROOT / "annotation" / args.name
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already exists; choose a new --name (never overwrite labels)")
    out.mkdir(parents=True, exist_ok=True)
    for name, fields, rows in (("sheet.csv", SHEET_FIELDS, sheet), ("claims.csv", CLAIM_FIELDS, claim_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    (out / "packet.md").write_text("\n".join(packet) + "\n", encoding="utf-8")
    (out / "key.json").write_text(json.dumps({"run": args.run, "items": key}, indent=1), encoding="utf-8")
    (out / "README.md").write_text(README.format(
        name=args.name, run=args.run, system=cfg["system"], eval_set=cfg["eval_set"], n_claims=len(chosen),
        n_items=len(sheet), seed=args.seed, statuses="\n".join("   " + l for l in STATUS_DEFINITIONS.splitlines())),
        encoding="utf-8")
    print(f"wrote {out}: {len(chosen)} claims, {len(sheet)} obligations")


if __name__ == "__main__":
    main()
