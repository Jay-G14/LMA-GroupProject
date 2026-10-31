# Verify What Must Be True: Obligation Ledgers for Scientific Claim Verification

LMA Group Project 2026, Team Comfortably Numpy: Jayant Gupta, Manas Agarwal, Sparsh Bhartia, Nishanth R.

We test whether a verifier that writes down what must be true for a claim (typed *obligations*), checks each
obligation against retrieved abstracts and decides with a fixed rule, does better on SciFact than direct
retrieval-augmented generation (RAG). Every system uses the same model (Qwen3-8B, official 4-bit AWQ build,
served with vLLM), the same retriever (BM25 and BGE-base, fused by reciprocal-rank fusion) and the same number
of abstracts.

## Results so far

Claim-level macro-F1, one run per system with the frozen configuration (all systems and metrics:
[to_transfer/summary.md](to_transfer/summary.md)):

| System | SciFact dev (300 claims) | SciFact-Open (278 claims) |
|---|---|---|
| B0, no retrieval | 0.459 | 0.454 |
| B1, direct RAG | **0.733** | **0.812** |
| B2, sub-question RAG | 0.698 | 0.766 |
| M1, obligation ledger | 0.699 | 0.767 |
| M2, M1 + targeted re-retrieval | 0.677 | 0.740 |

Neither hypothesis holds so far. The Auditor, which checks each obligation, often marks an obligation as
contradicted on an abstract that is only related to the claim. [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)
records the frozen settings, every amendment and the run log.

## Layout

| Path | Contents |
|---|---|
| `src/oblig/` | data loaders, retrieval, the obligation ledger and verdict rules, prompts, all systems, metrics |
| `configs/` | model, retrieval and dataset settings; `experiment_matrix.yaml` lists the systems and the comparisons fixed in advance |
| `scripts/` | runs, evaluation, comparisons, re-scoring and analyses; `scripts/ada/` holds the cluster jobs |
| `data/` | SciFact release, SciFact-Open claim files, the frozen leak-free development split (`data/splits/`), PUBHEALTH (audit only) |
| `to_transfer/` | outputs of every reported run: `dev/` (SciFact dev), `open/` (SciFact-Open), `pilot/` (the 199 development claims), `pilot_iter1_promptv3/` (pilot 1) |
| `annotation/` | blind packet for the human annotation of obligation statuses |
| `tests/` | unit tests |
| `results/`, `reports/` | early measurements: PUBHEALTH audit and shortcut baselines, first retrieval checks |

## Data and licences

All data in the repo are public releases, included unchanged so the results can be checked without downloads.

| Data | Folder | Source | Licence |
|---|---|---|---|
| SciFact claims and evidence labels | `data/raw/scifact/` | Wadden et al., EMNLP 2020 ([allenai/scifact](https://github.com/allenai/scifact)) | CC BY 4.0 |
| SciFact abstracts (`corpus.jsonl`, from S2ORC) | `data/raw/scifact/` | as above | ODC-By 1.0 |
| SciFact-Open labels for the SciFact test claims | `data/raw/scifact_open/` | Wadden et al., Findings of EMNLP 2022 ([dwadden/scifact-open](https://github.com/dwadden/scifact-open)) | none stated; files as in the official release |
| PUBHEALTH (dataset audit only) | `data/PUBHEALTH/` | Kotonya and Toni, EMNLP 2020 | MIT |

The run outputs in `to_transfer/` and the annotation packet quote sentences from the SciFact abstracts. Model weights
are not in the repo: `scripts/ada/setup_env.sh` downloads Qwen3-8B-AWQ and BGE-base from Hugging Face.

## Setup

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

## Reproduce

Model runs need a GPU and a vLLM server; [scripts/ada/README.md](scripts/ada/README.md) has the cluster jobs. One
system on the development claims:
```
python scripts/run_scifact.py --system m1 --eval-set prompt_dev --profile qwen3_8b_vllm --exp-id pilot_m1
```
The test sets (`--eval-set dev` or `scifact_open`) also need `--heldout`. Systems: `b0 b1 b1_9 b1_sc b2 m1
m1_shared m1_aligned m1_untyped m1_llmagg m2 m2_random m2_claimiter`; M2-noconsol and M2-claimfollow are `m2` with
`--m2-consolidate off` and `--m2-followup claim`. A CPU-only dry run uses `--profile mock_scifact --retriever
hybrid_bm25_minilm`; its scores mean nothing.

Everything else runs on the stored outputs, without a GPU (shown for SciFact dev; for SciFact-Open use
`--eval-set scifact_open`, prefix `open_` and `to_transfer/open`):
```
python scripts/rescore_ledgers.py --results-dir to_transfer/dev --prefix dev_    # post-hoc amendments 5 and 6
python scripts/compare_systems.py --eval-set dev --prefix dev_ --name dev --results-dir to_transfer/dev
python scripts/veto_analysis.py --results-dir to_transfer/dev --prefix dev_ --name dev
python scripts/error_analysis.py to_transfer/dev dev
python scripts/summarize_runs.py --comparisons to_transfer/dev/comparisons/dev.json \
    to_transfer/open/comparisons/open.json to_transfer/pilot/comparisons/prompt_dev.json --out to_transfer/summary.md
```
`scripts/dev_rescore.py` holds the development-only checks behind the amendments, and
`scripts/inspect_ledgers.py` prints ledgers for reading by hand.

Human annotation: two annotators fill in `annotation/dev_m1_50/sheet.csv` and `claims.csv` as its README says,
then run `python scripts/score_annotation.py --name dev_m1_50`.
