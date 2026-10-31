# pilot_iter2_60: prompt_dev, 60 claims

Macro-F1 with 95% bootstrap CI. Unsupported = gold-NEI claims given SUPPORTS/REFUTES (safety). Abstract = SciFact abstract-level F1 (label+rationale, or label-only where rationale sets are merged). Sentence = SciFact sentence-level selection+label F1 (RAG metric). Recall = gold-abstract recall of the abstracts shown.

| system | macro-F1 [95% CI] | acc | F1 S / R / NEI | unsupported | abstract | sentence | recall | calls | tokens |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.352 [0.260, 0.435] | 0.483 | 0.53 / 0.00 / 0.52 | 0.568 | 0.000 | 0.000 | 0.000 | 1.0 | 224 |
| B1 | 0.672 [0.505, 0.806] | 0.750 | 0.75 / 0.46 / 0.81 | 0.270 | 0.473 | 0.386 | 1.000 | 1.0 | 2228 |
| B1-9 | 0.669 [0.492, 0.806] | 0.733 | 0.73 / 0.50 / 0.78 | 0.297 | 0.491 | 0.378 | 1.000 | 1.0 | 3768 |
| B1-SC | 0.672 [0.505, 0.806] | 0.750 | 0.75 / 0.46 / 0.81 | 0.270 | 0.464 | 0.366 | 1.000 | 5.0 | 11147 |
| B2 | 0.603 [0.430, 0.740] | 0.650 | 0.68 / 0.46 / 0.67 | 0.486 | 0.431 | 0.364 | 1.000 | 2.0 | 2559 |
| M1 | 0.695 [0.547, 0.819] | 0.767 | 0.78 / 0.47 / 0.84 | 0.243 | 0.531 | 0.353 | 1.000 | 4.37 | 8320 |
| M1-LLMagg | 0.695 [0.555, 0.817] | 0.767 | 0.81 / 0.44 / 0.83 | 0.270 | 0.493 | 0.336 | 1.000 | 5.37 | 8927 |
| M2 | 0.631 [0.503, 0.747] | 0.683 | 0.81 / 0.35 / 0.73 | 0.405 | 0.442 | 0.307 | 1.000 | 10.83 | 20479 |
| M2-claimiter | 0.643 [0.484, 0.777] | 0.717 | 0.73 / 0.43 / 0.77 | 0.324 | 0.441 | 0.370 | 1.000 | 2.93 | 5920 |

## Paired comparisons (b − a)

| comparison | RQ | Δ macro-F1 [95% CI] | p (Holm) | McNemar a-only / b-only, p (Holm) | Δ unsupported [95% CI] |
|---|---|---|---|---|---|
| B0 vs B1 | RQ1 | +0.321 [+0.148, +0.466] | 0.000 | 2 / 18, 0.004 | -0.297 [-0.460, -0.135] |
| B1 vs B2 | RQ1 | -0.070 [-0.169, +0.026] | 0.925 | 8 / 2, 0.766 | +0.216 [+0.081, +0.351] |
| B1 vs M1 | RQ1 | +0.022 [-0.085, +0.145] | 1.000 | 4 / 5, 1.000 | -0.027 [-0.162, +0.108] |
| B2 vs M1 | RQ1 | +0.092 [-0.025, +0.230] | 0.925 | 3 / 10, 0.738 | -0.243 [-0.378, -0.108] |
| B1-SC vs M1 | RQ1 | +0.022 [-0.085, +0.145] | 1.000 | 4 / 5, 1.000 | -0.027 [-0.162, +0.108] |
| M1-LLMagg vs M1 | RQ1 | -0.001 [-0.056, +0.054] | 1.000 | 2 / 2, 1.000 | -0.027 [-0.135, +0.054] |
| M1 vs M2 | RQ2 | -0.064 [-0.134, +0.001] | 0.445 | 6 / 1, 0.766 | +0.162 [+0.054, +0.297] |
| B1-9 vs M2 | RQ2 | -0.039 [-0.159, +0.095] | 1.000 | 7 / 4, 1.000 | +0.108 [-0.054, +0.270] |
| M2-claimiter vs M2 | RQ2 | -0.013 [-0.117, +0.107] | 1.000 | 6 / 4, 1.000 | +0.081 [-0.081, +0.243] |

## Agentic loop (first pass → iterated, within each run)

| system | loop gain Δ macro-F1 [95% CI] | Δ gold-abstract recall | extra calls / claim | Δ macro-F1 per call | fixed / broken verdicts | entering loop |
|---|---|---|---|---|---|---|
| M2 | -0.064 [-0.134, +0.001] | 0.000 | 6.467 | -0.0099 | 1 / 6 | 0.600 |
| M2-claimiter | -0.029 [-0.076, +0.000] | 0.000 | 1.933 | -0.0152 | 0 / 2 | 0.500 |

## Ledger diagnostics

| system | obligations / claim | planner failures | resolving citations on a gold abstract | cited sentences in gold rationale |
|---|---|---|---|---|
| M1 | 3.37 | 0.000 | 0.567 | 0.578 |
| M1-LLMagg | 3.37 | 0.000 | 0.567 | 0.578 |
| M2 | 3.37 | 0.000 | 0.482 | 0.578 |
