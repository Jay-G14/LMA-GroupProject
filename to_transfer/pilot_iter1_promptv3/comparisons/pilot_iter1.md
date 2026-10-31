# pilot_iter1: prompt_dev, 60 claims

Macro-F1 with 95% bootstrap CI. Unsupported = gold-NEI claims given SUPPORTS/REFUTES (safety). Abstract = SciFact abstract-level F1 (label+rationale, or label-only where rationale sets are merged). Sentence = SciFact sentence-level selection+label F1 (RAG metric). Recall = gold-abstract recall of the abstracts shown.

| system | macro-F1 [95% CI] | acc | F1 S / R / NEI | unsupported | abstract | sentence | recall | calls | tokens |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.290 [0.202, 0.375] | 0.383 | 0.47 / 0.00 / 0.40 | 0.703 | 0.000 | 0.000 | 0.000 | 1.0 | 190 |
| B1 | 0.672 [0.506, 0.808] | 0.750 | 0.75 / 0.46 / 0.81 | 0.270 | 0.526 | 0.404 | 1.000 | 1.0 | 2195 |
| B1-9 | 0.638 [0.492, 0.768] | 0.717 | 0.77 / 0.38 / 0.77 | 0.324 | 0.424 | 0.369 | 1.000 | 1.0 | 3737 |
| M1 | 0.641 [0.509, 0.765] | 0.700 | 0.79 / 0.38 / 0.75 | 0.378 | 0.400 | 0.300 | 1.000 | 4.48 | 8246 |
| M1-LLMagg | 0.658 [0.522, 0.777] | 0.717 | 0.81 / 0.40 / 0.76 | 0.351 | 0.405 | 0.296 | 1.000 | 5.48 | 8892 |
| M2-claimiter | 0.638 [0.486, 0.771] | 0.717 | 0.73 / 0.40 / 0.78 | 0.324 | 0.500 | 0.381 | 1.000 | 2.9 | 5741 |

## Paired comparisons (b − a)

| comparison | RQ | Δ macro-F1 [95% CI] | p (Holm) | McNemar a-only / b-only, p (Holm) | Δ unsupported [95% CI] |
|---|---|---|---|---|---|
| B0 vs B1 | RQ1 | +0.382 [+0.202, +0.542] | 0.000 | 3 / 25, 0.000 | -0.432 [-0.622, -0.243] |
| B1 vs M1 | RQ1 | -0.031 [-0.136, +0.087] | 0.672 | 6 / 3, 1.000 | +0.108 [-0.027, +0.270] |
| M1-LLMagg vs M1 | RQ1 | -0.016 [-0.048, +0.001] | 0.672 | 1 / 0, 1.000 | +0.027 [+0.000, +0.081] |

## Agentic loop (first pass → iterated, within each run)

| system | loop gain Δ macro-F1 [95% CI] | Δ gold-abstract recall | extra calls / claim | Δ macro-F1 per call | fixed / broken verdicts | entering loop |
|---|---|---|---|---|---|---|
| M2-claimiter | -0.035 [-0.086, +0.000] | 0.000 | 1.9 | -0.0184 | 0 / 2 | 0.500 |

## Ledger diagnostics

| system | obligations / claim | planner failures | resolving citations on a gold abstract | cited sentences in gold rationale |
|---|---|---|---|---|
| M1 | 3.48 | 0.000 | 0.439 | 0.534 |
| M1-LLMagg | 3.48 | 0.000 | 0.439 | 0.534 |
