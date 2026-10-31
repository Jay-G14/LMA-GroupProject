# Experiment protocol

Status: frozen at Gate 3 (2026-10-30, 01:20 IST), before any test-set run; the research questions and controls
were frozen earlier (§9, Gate 2, 2026-10-27). Every change is a dated amendment in §12: amendments 1–4 came before
the freeze, amendments 5 and 6 after it (post hoc, applied once to the stored outputs). Sections 1–11 are kept as
written before the freeze, including the PUBHEALTH plans that the team later set aside.

## 1. Tasks (kept separate; never pooled into one headline number)

| | SciFact (original proposal) | PUBHEALTH (team's new direction) |
|---|---|---|
| labels | SUPPORTS / REFUTES / NEI | true / false / mixture / unproven |
| evidence | 5,183-abstract corpus, sentence-level gold rationales | pooled, verdict-masked fact-check / news articles; no gold sentence evidence |
| held-out eval | dev (300) and SciFact-Open's labels for the hidden test claims (278, 5K setting; amendment 2) | official test (1,233 valid) |
| prompt development | `prompt_dev`: 199 of the 385 train claims that share no abstract with any held-out claim (93 NEI / 75 S / 31 R); the other 186 are the few-shot pool (`data/splits/scifact_leakfree_v1.json`, amendment 2). Not used: the shipped CV folds (55–68 claims_dev claims in every fold) and the earlier random `train_dev` (64 of its 200 claims shared an abstract with a dev claim) | dev (1,214) |
| evidence metrics | SciFact official abstract- and sentence-level metrics, gold-abstract recall (dev); label-only abstract F1 and per-sentence rationale F1 on citation evidence (SciFact-Open) | silver only: own-article recall@k, explanation-sentence recall |

**Consequence for the research questions.** RQ2 asks whether targeted re-retrieval improves *evidence recall*. Only
SciFact has gold evidence to measure that. PUBHEALTH can answer the *verdict* part of RQ1/RQ2 and adds a
generalisation test, but its evidence metrics are silver. Recommendation: SciFact remains the primary testbed for
RQ1/RQ2 as proposed; PUBHEALTH is the second, larger testbed with its own four-class aggregation rule (§4).

## 2. PUBHEALTH inputs

* Model input: claim text only.
* Evidence corpus: main_text of all 12,251 valid records (train+dev+test), split into 4-sentence passages
  (stride 4), verdict sentences masked. The claim's own article is in the pool and must be *retrieved*.
* Forbidden as input: label, explanation, claim_id (encodes label), fact_checkers, subjects, sources,
  date_published, split name. Passage ids are opaque hashes; prompts show passages as `E1..Ek`.
* Few-shot exemplars (if any) come from train only and never from articles shared with test.

## 3. Systems (one model, one retriever, matched budgets)

| id | system | evidence per claim (PUBHEALTH draft) |
|---|---|---|
| B0 | zero-shot LLM | 0 |
| B1 | direct RAG, BM25(claim) | 5 passages |
| B1-9 | direct RAG, matched to M2's max budget | 5 + max extra of M2 (9 for SciFact abstracts) |
| B2 | sub-question RAG (no ledger, no statuses) | 5 passages total, split across sub-questions |
| M1 | obligation RAG, single pass | 5 passages total, split across obligations |
| M2 | iterative obligation RAG | M1 + ≤2 rounds × ≤2 new passages/abstracts per round |
| M3 | debate (optional) | only if compute remains |

SciFact keeps the proposal's settings: top-5 abstracts, ≤6 obligations, ≤2 extra rounds of ≤2 new abstracts.

**SciFact retrieval (measured on train_dev, 125 claims with evidence; `results/metrics/retrieval_scifact_train_dev.json`).**
Recall@5 / @10: BM25 0.887 / 0.913; MiniLM dense 0.865 / 0.905; BM25+MiniLM RRF 0.882 / 0.956. At the k=5 budget the
three are indistinguishable at this sample size; the hybrid's advantage appears at k=10. Re-measured on the leak-free
prompt_dev (106 claims with evidence; `results/metrics/retrieval_scifact_prompt_dev.json`): BM25 0.906 / 0.934; MiniLM
0.893 / 0.934; BM25+MiniLM RRF 0.915 / 0.962. BGE-base, the configured default (amendment 2), is measured on the GPU
server before Gate 3. Abstract-level evidence
recall at k=5 already misses only ~12% of gold abstracts, which is the most any re-retrieval (M2 or B1-9) could recover;
simply extending the claim query to k=10 recovers ~7 points of it. RQ2's evidence-recall effect is therefore bounded
and small on SciFact, and B1-9 is a strong control. The 5,183-abstract corpus is small, so retrieval is
easier than in an open setting (cf. SciFact-Open). Selected: hybrid RRF (proposal design); the dense model is
BGE-base once downloaded, else MiniLM, and the choice is recorded before any held-out run.

## 4. Aggregation

* SciFact: proposal rule — REFUTES if any obligation CONTRADICTED; SUPPORTS if every core obligation SATISFIED from
  one abstract and none CONTRADICTED; else NEI.
* PUBHEALTH (to be fixed on dev before Gate 3, candidates): false if a *core* obligation is CONTRADICTED and none
  SATISFIED; mixture if both SATISFIED and CONTRADICTED core obligations occur; true if all core SATISFIED;
  unproven otherwise. Compared against LLM aggregation over the same ledger. Chosen on dev only.

## 5. Metrics and statistics

Primary: macro-F1 (four-class PUBHEALTH; three-class SciFact claim-level) with a 95% bootstrap CI.

SciFact metric groups (`src/oblig/evaluation/scifact.py`, amendment 2), one capability metric each marked:
* task: macro-F1, accuracy, per-class P/R/F1, confusion matrix;
* safety: unsupported-verdict rate on gold-NEI claims (the proposal's safety metric), polarity errors, verdicts
  without a valid citation, invalid outputs;
* rationale: SciFact's official abstract-level label+rationale F1 (first 3 predicted sentences), plus label-only and
  an uncapped diagnostic; official sentence-level selection+label F1 (**RAG capability metric**) and selection-only;
* evidence: gold-abstract recall of the abstracts shown to the verdict step, all-gold-shown, rationale-sentence coverage;
* ledger: obligations, types, statuses, planner failures, verdict reasons, how often resolving audits cite gold
  abstracts and gold rationale sentences;
* iteration: **loop gain** = change in macro-F1 and in gold-abstract recall from the first pass to the iterated
  result, per extra LLM call (**agentic capability metric**); verdict flips, status transitions, stop reasons;
* strata: gold abstract shown vs missed, number of obligations; robustness: 31 minimal pairs on dev (pair accuracy,
  contrast consistency); cost: calls, tokens, latency.

Reasoning capability metric: obligation-status accuracy against two annotators on 50 dev claims (proposal), with a
blind export (`scripts/export_annotation.py`) and scoring (`scripts/score_annotation.py`: Cohen's kappa, accuracy,
macro-F1, coverage).

Comparisons (`scripts/compare_systems.py`; list in `configs/experiment_matrix.yaml`): paired bootstrap 95% CI of the
macro-F1 difference (10,000 resamples, seed 13), exact McNemar test on per-claim correctness, change in
unsupported-verdict rate; p-values Holm-adjusted across all comparisons. Failed or unparseable outputs count as wrong
(`__invalid__`), never dropped.

Mandatory reference rows for PUBHEALTH: majority, claim_id-bin shortcut, claim-only TF-IDF LR, masked-article
TF-IDF LR, metadata LR (`results/metrics/shortcut_baselines.json`). PUBHEALTH results are also reported separately for
the wire-news block (claim_id < 8936, all `true`) and the fact-check remainder — the split is used for analysis only.

## 6. Generalisation splits (PUBHEALTH)

Held-out source site (wire-news vs PolitiFact vs Snopes/TruthOrFiction vs Health News Review), inferred from
id ranges and metadata and documented. Held-out fact-checker requires a cleaned checker mapping first (credentials
are comma-split in the raw field). Since the LLM systems are not trained, these splits only restrict which data may be
used for prompt development.

## 7. Model and decoding

Qwen3-8B, non-thinking mode, served by vLLM, temperature 0, seed 13, revision pinned on first download. One profile
(`configs/model.yaml: qwen3_8b_vllm`) for all controlled systems.

## 8. Open decisions for the team (block Gate 3)

1. Primary testbed: SciFact (as proposed) + PUBHEALTH (secondary), or PUBHEALTH primary?
2. GPU: the audit machine is CPU-only; which cluster/GPU will serve Qwen3-8B?
3. Whether the PUBHEALTH corpus should exclude each claim's own article (harder, "open" setting) as a second condition.

## 9. Research questions — FROZEN (Gate 2 passed)

Approved by the team on 2026-10-27; frozen from this point. Changes require a dated protocol amendment recorded here.
Gate 3 (full protocol freeze) still needs the PUBHEALTH aggregation rule and the dense-model choice fixed on development data.

Source: `docs/LITERATURE_REVIEW.md` §4–5. Verdict: the mechanism partially overlaps prior work (closest: FactAgent,
Trinh et al. 2025, decompose → per-subclaim retrieval → per-subclaim labels → AND rule, on SciFact-Open; FIRE,
NAACL Findings 2025, per-claim confidence-driven re-search; Vladika et al. NAACL 2025, iterative question-guided
verification on SciFact; Evidence-Ledger Adjudication, arXiv 2607.26512, a claim-level 4-relation "ledger").
No "first" claims. The contribution is framed as a **budget-controlled study** of typed obligation ledgers and
status-targeted re-retrieval with an open 8B model.

* **RQ1 (refined).** With model, retriever and evidence-token budget fixed, does per-obligation verification beat
  direct RAG and sub-question RAG on SciFact, and which component explains any difference: obligation-aligned
  evidence vs shared claim-level evidence, per-obligation statuses, or the deterministic rule vs LLM aggregation?
  H1a gain concentrated in REFUTES/NEI recall; H1b gain shrinks with shared evidence (Akhter et al. 2026);
  H1c risk: decomposition/auditing noise at 8B cancels the gain (FactAgent with a 3B model).
* **RQ2 (refined).** Does status-targeted re-retrieval (PARTIAL/UNSUPPORTED only) beat single-pass and beat
  untargeted extra evidence at equal added evidence, and at what cost? Report gain per added abstract and per LLM
  call, status-transition matrix, share of claims entering M2, harmful vs helpful verdict flips, false-SUPPORTS on NEI.
  Bound: on train_dev, recall@5 already misses only ~12% of gold abstracts (§3), so the evidence-recall effect is small.
* **RQ3 (optional, secondary).** Robustness on PUBHEALTH with its own frozen four-class rule, stratified by genre.

Added controls (all same model, caps and evidence budget):
B1-SC direct RAG with self-consistency (token-matched); M1-shared obligations audited on shared claim evidence;
M1-untyped FActScore-style untyped decomposition; M1-LLMagg LLM aggregation over the same ledger;
M2-random re-retrieval for random obligations; M2-claimiter FIRE/QACheck-style claim-level iteration under M2's caps.

## 10. B2 / M1 implementation decisions (SciFact; `src/oblig/systems/scifact.py`)

Design choices, not measured results. Gate 7 (hand-checked ledgers from the real model) is still open.

* **Evidence budget.** B1, B2, M1 and M1-shared each show the model at most 5 distinct abstracts per claim. B2 and M1
  build that pool by reciprocal-rank fusion (k = 60) of their per-query rankings (proposal D3; amendment 2; originally
  round-robin). So decomposition changes *which* abstracts are seen, not how many. Token cost differs (M1 makes 1 + n_obligations
  calls) and is reported per claim.
* **Queries.** B2 sub-question / M1 obligation query = `"{query} {claim}"` (proposal: combine with the claim).
* **M1 evidence.** Every obligation is audited on the whole pool, as in the proposal (amendment 2). The original design,
  in which each obligation saw only the pool abstracts within its own top-5 (round-robin pool), is kept as the
  ablation **M1-aligned**. **M1-shared** audits every obligation on B1's claim-level top-5.
* **B2 has no statuses.** It makes one decomposition call and one verdict call that sees the sub-questions and the
  pooled abstracts.
* **Planner.** At most 6 obligations. Criticality comes from the type (subject/outcome/relation = core), never from
  the model. Duplicate (type, normalised description) obligations are dropped and recorded. If the Planner fails,
  the ledger is empty and the verdict is NEI, counted as a planner failure.
* **Auditor citations.** These are abstract aliases A1..Ak plus sentence numbers, resolved against the abstracts
  actually shown. Quoted text is taken from the corpus. A SATISFIED/CONTRADICTED answer without a valid citation
  keeps its raw status but has effective status UNSUPPORTED. Aggregation uses effective statuses.
* **Ledger.** Audits are append-only with round numbers. SATISFIED/CONTRADICTED are terminal, and rounds must
  increase. These invariants are enforced in code (`src/oblig/ledger/schema.py`).
* **Verdict.** This is the proposal rule `scifact_v1`, with the reason logged per claim. On SUPPORTS, the rationale is
  the core obligations' cited sentences in the shared abstract, ordered most decisive first (amendment 2).
  Abstract-level scoring keeps the first 3 sentences in that order, as in the SciFact metric.
* **Held-out guard.** `run_scifact.py` refuses claims_dev and the SciFact-Open test set without `--heldout`, and refuses to resume a prediction
  file produced with a different configuration.

## 11. M2 and added controls: implementation (SciFact; `src/oblig/systems/scifact.py`)

These are design choices, not results. All controls use the same model, retriever, caps and prompts as the system
they control.

* **M2.** It starts with M1 (round 0), then runs at most 2 rounds. Each round:
  1. One follow-up-query call per PARTIAL/UNSUPPORTED obligation (the Auditor's rationale is included).
  2. Up to 2 abstracts the claim hasn't seen yet are retrieved, fused by RRF across the follow-up queries (amendment 2)
     (`"{query} {claim}"`).
  3. Every unresolved obligation is re-audited on its previous evidence plus the new abstracts.

  At most 5 + 2×2 = 9 distinct abstracts per claim. Resolved obligations are never re-audited, and history is
  append-only.

  Stop reasons are recorded per claim: all obligations resolved / no new evidence / round cap reached. If a
  follow-up-query call fails, the obligation's own query is used and the error is logged. The record keeps the M1
  verdict, so helpful and harmful flips can be counted. `needs_more_retrieval` is recorded but does not trigger
  anything; the trigger is the status.
* **M2-random.** Identical to M2, except each round's follow-up queries are written for the same *number* of
  obligations drawn uniformly from all obligations (seeded per claim). Re-audit still covers the unresolved
  obligations. This isolates whether targeting by status matters.
* **M2-claimiter.** Claim-level iteration in the style of FIRE/QACheck: a B1 verdict; while it is NEI, one claim-level
  follow-up query and up to 2 new abstracts, at most 2 rounds (9 abstracts). There is no calibrated confidence, so
  NEI is the trigger.
* **B1-9.** B1 with top-9 abstracts (M2's maximum).
* **B1-SC.** B1 evidence, 5 verdicts sampled at temperature 0.7 (seeds 13..17), majority vote. A tie goes to NEI if
  NEI is among the tied labels, otherwise to the label sampled first. The sample count is to be token-matched to M1
  from the GPU pilot and recorded before Gate 3.
* **M1-untyped.** FActScore-style atomic facts (≤6), each a core obligation of type `atomic`; otherwise M1.
* **M1-LLMagg.** The same ledger as M1. The verdict comes from one LLM call over the obligation list (criticality,
  status, rationale, quoted evidence). The rule's verdict is also stored (`rule_label`) for a paired comparison.

### Finding from implementation (needs a team decision before Gate 3)

Under `scifact_v1`, SUPPORTS needs every core obligation satisfied **in one abstract**. Resolved obligations are
terminal in M2. So when M2 finds the missing evidence in a *new* abstract, the obligations satisfied earlier still
cite an *old* abstract. SUPPORTS can then only fire if the newly satisfied obligation cites an old abstract. In
practice, M2 can change NEI → REFUTES far more easily than NEI → SUPPORTS. A unit test reproduces this
(`tests/test_iterative.py::test_m2_targets_unresolved_and_preserves_history`). In the mock run, 11/30 claims entered
iteration and 0 changed verdict (mock outputs only).

Options:
1. Keep the rule as specified and report the asymmetry.
2. Amend M2 with a *consolidation audit*: when a new abstract satisfies a core obligation, re-check the other
   satisfied core obligations against that abstract only. This costs extra calls, stays within the 9-abstract cap,
   and is recorded as a new round.

Either choice has to be fixed before any held-out run. **Resolved: option 2 adopted, see §12.**

## 12. Protocol amendments

### Amendment 1 — 2026-10-27, 16:39 UTC / 22:09 IST: M2 consolidation audit

**Decided by:** the team (approval in the project session), before any held-out or real-model run.
**Reason:** §11 finding. Under `scifact_v1`, resolved obligations are terminal and cite their round-0 abstracts. So
evidence that M2 finds in a new abstract could almost never produce SUPPORTS.

**Change (M2 and M2-random only; M1 and every other system unchanged):**
1. After the re-audits of a round, take every abstract cited by a core obligation that became SATISFIED in that round.
2. For each such abstract, re-check every other SATISFIED core obligation that does not already cite it, using the
   standard Auditor prompt with **that abstract alone** (call role `consolidation`).
3. A consolidation is stored as an audit of `kind: "consolidation"` in the same round. It **never changes the
   obligation's status**. Only a SATISFIED consolidation with a valid citation adds evidence (`Ledger.evidence`).
   Any other outcome, including CONTRADICTED, is recorded and has no effect on the verdict.
4. No new retrieval: the 9-abstract cap is unchanged. Each (obligation, abstract) pair is checked at most once, and
   qualifiers are never consolidated.

**Effects to report:** consolidation calls per claim (counted in cost) and verdict flips with and without the step.
`Context.m2_consolidate = False` reproduces the original design for the ablation. RQ2's question is unchanged; M2
now has one more call type, and the comparisons with B1-9, M2-random and M2-claimiter still hold the evidence caps
fixed. M2-random gets the same consolidation, so it differs from M2 only in targeting.

**Tests:** `tests/test_ledger.py::test_consolidation_invariants`, `::test_scifact_rule_uses_consolidated_evidence`;
`tests/test_iterative.py::test_m2_targets_unresolved_and_preserves_history`,
`::test_m2_without_consolidation_keeps_the_asymmetry`,
`::test_failed_consolidation_adds_no_evidence_and_never_changes_status`.

### Amendment 2 — 2026-10-29: proposal fidelity, leak-free development data, second test set, full metrics

**Decided by:** the team (Jayant), before any held-out or real-model run.

**Changes:**
1. **Development data (leakage).** The random 200-claim `train_dev` is replaced by `prompt_dev`: 199 of the 385 train
   claims that share no cited abstract with any held-out claim (proposal D9; frozen ids in
   `data/splits/scifact_leakfree_v1.json`; the other 186 are the few-shot pool). Measured on the old split: 64 of its
   200 claims shared an abstract with a claims_dev claim, 33 were near-identical to a dev claim (string similarity
   ≥ 0.75), and 18 of those twin pairs had opposite labels, so prompt tuning on it would tune on near-copies of test
   claims. `load_split` / `leak_free_train` recompute and check the pool (tests).
2. **Second held-out test set.** SciFact-Open's labels for the hidden SciFact test claims, 5,183-abstract setting:
   278 claims (97 S / 96 R / 85 NEI); claim 158 excluded (conflicting labels). Pooled evidence counts for labels but
   has no gold rationale sentences; rationale sets are merged, so label+rationale F1 is computed on dev only.
3. **M1 as in the proposal.** Per-obligation rankings are fused by RRF into one pool of 5 and every obligation is
   audited on the whole pool. B2 and M2's new-abstract selection use the same fusion. The original design is kept
   as the ablation M1-aligned.
4. **Rationale order (metric bias).** SciFact's abstract-level metric scores only the first three predicted
   sentences. Citations were previously sorted by sentence number, so M1 (which pools sentences from several
   obligations) systematically kept early, often background, sentences. Now: citations keep the model's order; every
   system's prompt asks for at most 3 sentences per abstract, most important first (B1/B2/Auditor prompts v2); a
   ledger's rationale is ordered by the number of deciding obligations citing a sentence, then obligation type
   (relation, outcome, subject, qualifiers), then citation position. An uncapped label+rationale F1 is reported as a
   diagnostic. The Auditor is also asked to cite every abstract that shows a condition, so the one-abstract SUPPORTS
   rule is not failed by citing only one of two qualifying abstracts.
5. **Metrics.** Full set in §5 (safety, official sentence level, evidence recall, loop gain, ledger grounding, strata,
   minimal pairs), the comparison script with Holm-adjusted tests, and the annotation tooling. `paired_bootstrap`
   previously defaulted to PUBHEALTH's labels (all SciFact macro-F1 would have read 0); labels are now explicit.
6. **Dense model.** BGE-base (proposal D3) is the configured default; MiniLM is for local CPU smoke runs only.

**Not changed:** research questions, systems, budgets, caps, verdict rule, Amendment 1.

**Tests:** `tests/test_scifact.py` (leak-free splits, SciFact-Open), `tests/test_ledger.py` (pool, evidence modes,
citation order, rationale order), `tests/test_iterative.py` (first-pass record), `tests/test_evaluation.py` (every
metric group, statistics, annotation). 57 tests passing. Mock smoke runs of all 13 systems on prompt_dev:
`results/predictions/smoke2_*`, `results/comparisons/smoke2_prompt_dev.md` (mock scores mean nothing).

**Expectation to check in the pilot:** because each obligation query includes the claim, M1's fused pool equalled
the claim-level top-5 for 19 of 30 smoke claims (4.6 of 5 abstracts shared). M1 vs M1-shared is therefore expected
to be small; M1's difference from B1 comes mainly from per-obligation auditing and the rule.

### Amendment 3 — 2026-10-29 (evening): serving setup, follow-up search, prompt order

**Decided by:** the team (Jayant), before any real-model run.

1. **M2 follow-up search.** Each follow-up query is searched alone and with the claim appended, and the two
   rankings are RRF-fused (`Context.m2_followup = "fused"`; `"claim"` reproduces the amendment-2 behaviour).
   Reason: with the claim appended, 22 of the 42 abstracts M2 added in the mock smoke runs were exactly the
   claim's own ranks 6-9, i.e. what B1-9 sees, so "targeted" re-retrieval was barely distinguishable from more
   abstracts (RQ2). The proposal fixes "obligation + claim" for the obligation queries (D3), not for the
   Auditor's follow-up query. The same search is used by M2-random and M2-claimiter.
2. **Prompt order.** Prompts that read abstracts (B1, B2 verdict, Auditor; v3) put the abstracts first, so calls
   about the same evidence share a prefix the server caches (all Auditor calls of a claim, all B1-SC samples).
   Same order for every system.
3. **Structured output.** The server constrains each output to the call's JSON schema (vLLM guided decoding,
   `structured_output: true` in `configs/model.yaml`), for every system; the proposal specifies a fixed schema.
4. **Serving.** Qwen3-8B in non-thinking mode, greedy, via vLLM 0.9.2, using Qwen's official AWQ 4-bit release
   (`Qwen/Qwen3-8B-AWQ`, revision 4da05a8) on one RTX 2080 Ti per job (V0 engine, fp16 activations, prefix
   caching). Reasons: the plafnet2 RTX 3090s are unavailable (the account's GPU-minute cap is reached); the
   research account's 2080 Ti cards (11 GB) cannot hold the 16 GB fp16 model; 4-GPU requests fell back to
   GTX 1080 Ti nodes, which vLLM does not support. Every system uses the same model build. (The proposal's
   compute plan originally named AWQ replicas, one per GPU.)
5. **Runs.** `scripts/ada/` (setup and run jobs based on the team's Ada template). Claims run in parallel worker
   threads; transport errors are retried before a call counts as failed. The obligation systems of one eval
   set run in one job sharing one response cache, so M1-LLMagg, M2, M2-random and the M1 ablations reuse M1's
   exact planner outputs (and M2 its round-0 audits); the baselines run in a second job.
6. **Gate 3** is passed after the pilot on prompt_dev (all 13 systems): B1-SC's sample count is set from the
   pilot's token ratio to M1, and nothing else changes before the held-out runs.

### Amendment 4 — 2026-10-30, 01:00 IST: pilot iteration 1 (Gate 7 hand-check) and fixes

**Decided by:** the team (Jayant), on prompt_dev evidence only, before any held-out run.

**Pilot iteration 1** (prompts v3, first 60 prompt_dev claims, Qwen3-8B-AWQ; results kept in
`to_transfer/pilot_iter1_promptv3/`): B0 0.290, B1 0.672, B1-9 0.638, M1 0.641, M1-LLMagg 0.658, M2-claimiter
0.638 macro-F1. Hand-checking M1 ledgers found two faults:
1. **Collapsed decomposition.** 183 of 209 obligations (56 of 60 claims) had the type name as their description
   ("Subject", "Outcome", "Relation"), with the whole claim restated as the expected condition, so M1 was not
   testing decomposition at all.
2. **Contradiction read into non-evidence.** M1 gave REFUTES on 9 of 37 gold-NEI claims (B1: 3), each through an
   obligation marked CONTRADICTED on a different finding or on missing information (e.g. ATM neurodegeneration
   findings "contradicting" a claim about acute tissue loss).

**Fixes (all systems where the text is shared):**
* Planner v2: schema field order `type` before `description`; instruction that each condition is a short
  statement in words, never the type name; one worked example, claim 709 from the leak-free few-shot pool
  (the proposal's few-shot mitigation for the planner).
* Status definitions (Auditor) and label definitions (every direct verdict: B0, B1, B1-9, B1-SC, B2,
  M2-claimiter, M1-LLMagg) v2: a contradiction/refutation is a sentence stating the opposite for the same
  subject; a different finding, a different topic or missing information is not.
* Serving (no effect on outputs' meaning): no prefix caching (its Triton kernel does not compile on Turing);
  GPU memory 0.85, 8,192-token batches and context, 32 sequences (0.92 / 12,288 ran out of memory); the job
  restarts a dead server and the runner never records a failed server call as a model answer.

Pilot iteration 2 (same 60 claims) checks the fixes; Gate 3 follows it.

### Gate 3 — protocol frozen, 2026-10-30, 01:20 IST

Pilot iteration 2 (prompts v4, first 60 prompt_dev claims): B0 0.352, B1 0.673, B1-9 0.669, B1-SC (5 samples)
0.673, B2 0.603, M2-claimiter 0.643, M1 0.695, M1-LLMagg 0.695, M2 0.631 macro-F1. Obligations are now real
conditions (0 of the obligations in the first 28 claims were bare type names; 3.3 per claim).

**M2 finding, reported, not tuned away.** M2 changed 7 of 60 verdicts: 1 fixed (gold SUPPORTS) and 6 broken (gold
NEI turned REFUTES), each through a re-audit that read a contradiction into a newly retrieved, tangential abstract.
Re-retrieval on claims without evidence only adds distractors, and the 8B Auditor still over-reads contradiction.
Further prompt tuning on 60 development claims would risk overfitting, so M2 is frozen as specified; the
claim-only follow-up and no-consolidation runs are the planned ablations for the final report.

**Frozen for the held-out runs (dev, scifact_open):** prompt versions (b0 sf-b0-v2, b1 sf-b1-v4, b2_decomp sf-b2-decomp-v1, b2_verdict sf-b2-verdict-v4, planner sf-planner-v2, auditor sf-auditor-v4, facts sf-facts-v1, requery sf-requery-v1, claim_requery sf-claim-requery-v1, b1_iter sf-b1-iter-v4, aggregate sf-aggregate-v2); B1-SC with 4 samples (token-matched
to M1 on the pilot: 8,320 vs 2,228 tokens per claim, ratio 3.7); retriever BM25 + BGE-base fused by RRF (k = 60),
K = 5, R = 2 rounds of up to 2 new abstracts, fused follow-up search, consolidation on, verdict rule scifact_v1;
Qwen3-8B (Qwen/Qwen3-8B-AWQ, revision 4da05a8) in non-thinking mode, greedy, vLLM 0.9.2 with structured output.
The baseline jobs were launched at 01:18 (their prompts could no longer change); the obligation-system jobs
after this entry.

**Run log, 2026-10-30, 01:50 IST (no held-out result of the runs below had been seen).**
* M2-claimiter failed on both held-out sets at ~01:28: every call that needed vLLM's outlines backend returned
  HTTP 500 because its SQLite cache, shared on NFS home by four jobs, was corrupted. The runner aborts a claim on
  a server error instead of recording it, so the 70 (dev) and 69 (SciFact-Open) completed claims are unaffected
  (no call errors); the remaining claims are run by resuming the same experiment ids with a node-local cache.
  No other system was affected: B0 and B1 had finished, and the group-A servers (opened the cache at 01:20)
  logged no HTTP 500s and their records no call errors (checked at 01:47).
* The two amendment ablations planned at Gate 3 (M2 without consolidation; M2 with claim-appended follow-up
  search only) are compared with M2 as a separate Holm family (`ablation_comparisons` in
  `configs/experiment_matrix.yaml`), so the 13 pre-registered comparisons keep their frozen adjustment.

**Run log, 2026-10-30, 02:55 IST — an omission.** §3 planned to measure BGE-base retrieval on prompt_dev before
Gate 3; the pilot jobs did not run the retrieval evaluation, so the frozen configuration used the proposal's hybrid
BM25 + BGE-base (D3) without that measurement. It runs now in `oblig_pilot_B`, after the test runs. On dev (test
set 1, `oblig_dev_B`) recall@5 is BM25 0.865, BGE-base 0.943, hybrid 0.919. The test runs stand as frozen; any
retriever change for the final submission would be an amendment reported next to these results, and would be
informed by this test-set measurement.
* 04:26 IST: the prompt_dev measurement (106 claims with evidence, `to_transfer/pilot/retrieval_scifact_prompt_dev.json`)
  gives recall@5 BM25 0.906, BGE-base 0.943, hybrid 0.953 (recall@10 0.934 / 0.981 / 0.962). On development data the
  hybrid would have been kept at K = 5, so the frozen choice stands.


**Run log, 2026-10-30, 13:17 IST — development-only exploration for the final submission.** No dev or
SciFact-Open data was used. `scripts/dev_rescore.py` re-scores the stored prompt_dev ledgers (199 claims) under
alternative contradiction rules without new LLM calls. Nothing is adopted: a rule taken forward becomes a
numbered amendment, frozen before it is applied once to the stored dev and SciFact-Open ledgers.
* Rules tried (macro-F1; frozen M1 0.736, B1 0.729):
  * subject obligations cannot be contradicted: 0.736, no verdict changes;
  * only relation or qualifier contradictions count: 0.729;
  * a contradiction counts only if an abstract it cites is also cited for a subject obligation ("anchored"):
    0.752, unsupported 0.312 → 0.258; on M2's final ledgers 0.711 → 0.742;
  * anchored to a SATISFIED subject obligation: 0.621, which breaks 20 gold-REFUTES claims, because the Auditor
    also marks the subject contradicted when only the direction is.

  Four rules were tried and the anchored rule is the best of them, so its development gain is optimistic.
* M1 vs B1: 0.695 vs 0.673 on the 60 claims used to revise prompts, and 0.735 vs 0.730 on the other 139.
* Fallback analysis for the final report: B1's verdict where M1 agrees, NEI otherwise. Macro-F1 is 0.735 and the
  unsupported-verdict rate 0.247 (B1 0.355); the two systems agree on 165 of 199 claims.

### Amendment 5 — 2026-10-30, 17:50 IST (12:20 UTC): anchored contradictions and the ledger-as-check fallback

**Decided by:** the team (Jayant), on prompt_dev evidence only (run log 13:17 and the checks below). Frozen here,
before the rule is applied to any dev or SciFact-Open ledger.

**Status: post hoc.** The idea came from the error analysis of the frozen test runs (the Auditor marks core
obligations CONTRADICTED on abstracts about another subject), and the rule is the best of four tried on prompt_dev.
Its test-set results are reported next to the frozen results, never instead of them; the frozen results remain the
test of H1 and H2.

**What changes:** only the verdict rule, applied to the stored ledgers. No new LLM calls; statuses, retrieval and
the M2 loop stay as run.
* `scifact_v1_anchored` (`src/oblig/ledger/aggregation.py`): as scifact_v1, except that a CONTRADICTED obligation
  counts only if an abstract it cites is also cited by a subject obligation of the same ledger, whatever that
  obligation's status. A contradiction that does not count is treated as unresolved: it cannot give REFUTES, and on
  a core obligation it blocks SUPPORTS. In a ledger without subject obligations (M1-untyped) every contradiction
  counts, so M1-untyped cannot change. A REFUTES verdict cites only the contradictions that count.
* Applied to every system whose verdict is the rule over a ledger: M1, M1-shared, M1-aligned, M1-untyped, M2,
  M2-random, M2-noconsol, M2-claimfollow (run ids `<set>_<system>_anch`, `scripts/rescore_ledgers.py`). For the M2
  family the first-pass (round-0) verdict is re-scored too, so loop gain compares like with like; an obligation
  whose contradiction no longer counts is not re-audited, since the loop is not re-run. Not applicable to B0, B1,
  B1-9, B1-SC, B2, M2-claimiter and M1-LLMagg, whose verdicts come from the LLM.
* Rule ablations: the three rules not chosen (subject obligations cannot be contradicted; only relation or
  qualifier contradictions count; anchored to a SATISFIED subject obligation) are applied to M1's ledgers only and
  reported descriptively, without tests.
* Fallback "B1-checked": B1's verdict where M1 gives the same verdict, NEI otherwise; with the frozen M1
  (`b1_checked`) and with M1-anch (`b1_checked_anch`). It costs both systems' calls.

**Comparisons** (`amendment_comparisons` in `configs/experiment_matrix.yaml`), a third Holm family, so the 13
pre-registered comparisons and the 2 amendment ablations keep their adjustment: M1 → M1-anch, B1 → M1-anch,
B2 → M1-anch, M2 → M2-anch, M1-anch → M2-anch, B1 → B1-checked, B1 → B1-checked-anch; statistics as in §5.

**prompt_dev (199 claims) through the reporting pipeline.** The frozen rule reproduces every stored verdict and
every stored M2 first-pass verdict. Macro-F1, frozen rule → anchored rule: M1 0.736 → 0.752 (unsupported 0.312 →
0.258), M1-shared 0.739 → 0.752, M1-aligned 0.714 → 0.743, M1-untyped 0.717 → 0.717, M2 0.711 → 0.742, M2-random
0.708 → 0.717, M2-noconsol 0.711 → 0.742, M2-claimfollow 0.703 → 0.728. B1-checked 0.735 (unsupported 0.247),
B1-checked-anch 0.743 (0.237), against B1 0.729 (0.355). Rule ablations on M1: 0.736, 0.729, 0.621. No comparison
in the amendment family is significant on prompt_dev (smallest Holm-adjusted p 0.489, B2 → M1-anch).

**Applied once** to the stored dev and SciFact-Open ledgers after this entry; the results go to the run log below
and to the report, next to the frozen results. Any further change to the rule is a new amendment.

**Not part of this amendment:** a fix at the source, an Auditor prompt that first asks whether the abstract studies
the obligation's subject. That needs a GPU re-run and would be designed on prompt_dev as a separate amendment.

**Run log, 2026-10-30, 18:00 IST — amendment 5 applied once to dev and SciFact-Open** (`scripts/rescore_ledgers.py`;
the frozen rule reproduced every stored verdict and M2 first pass on both sets). Macro-F1 [95% CI], unsupported rate:
* M1 → M1-anch: dev 0.699 → 0.716 (Δ +0.018 [−0.011, +0.046], Holm p 1.000), unsupported 0.348 → 0.241
  (Δ −0.107 [−0.170, −0.054]); SciFact-Open 0.767 → 0.739 (Δ −0.028 [−0.058, +0.002], p 0.249), unsupported
  0.271 → 0.200 (Δ −0.071 [−0.129, −0.024]).
* B1 → M1-anch: dev −0.016 (p 1.000); SciFact-Open −0.073 [−0.122, −0.028] (p 0.010). H1 stays unsupported.
* M2 → M2-anch: dev 0.677 → 0.723 (Δ +0.045 [+0.016, +0.076], p 0.022); SciFact-Open 0.740 → 0.737 (p 1.000).
  M1-anch → M2-anch: +0.006 (dev), −0.002 (SciFact-Open), both p 1.000.
* B1 → B1-checked: dev 0.720 (Δ −0.013, p 1.000), unsupported 0.393 → 0.250; SciFact-Open 0.776 (Δ −0.037,
  p 0.162), unsupported 0.224 → 0.153. B1 → B1-checked-anch: dev 0.730 (p 1.000), unsupported 0.205; SciFact-Open
  0.740 (Δ −0.073, p 0.006), unsupported 0.129.
* Rule ablations on M1 (dev / SciFact-Open): no-subject 0.702 / 0.763, relation-or-qualifier 0.706 / 0.760,
  anchored 0.716 / 0.739, anchored to a satisfied subject 0.506 / 0.520 (frozen 0.699 / 0.767).
* Why the rule does not transfer to SciFact-Open: it can only turn REFUTES into NEI. On dev it removed 23 REFUTES
  verdicts (12 gold NEI, 5 gold SUPPORTS, 6 gold REFUTES); on SciFact-Open 25 (6 NEI, 4 SUPPORTS, 15 REFUTES).
  SciFact-Open is 35% REFUTES (dev 21%, prompt_dev 16%), and in 14 of its 15 lost REFUTES the subject obligation
  was left UNSUPPORTED, so the contradicting abstract had no subject citation to anchor to.
* Decision: the frozen results remain the test of H1 and H2. Amendment 5 is reported next to them as a post-hoc
  analysis: it lowers unsupported verdicts on all three claim sets and removes M2's loss on dev, but it does not
  transfer as a macro-F1 gain. No further re-scoring rule will be tried on these ledgers; the next step is the
  source fix above, designed on prompt_dev.

### Amendment 6 — 2026-10-30, 18:12 IST (12:42 UTC): is the ledger a better check than any second opinion?

**Question (post hoc, no new LLM calls).** B1-checked lowers the unsupported-verdict rate, but so does any check
that makes B1 answer NEI more often. Does the ledger pick out B1's errors better than a random veto of the same
size, and better than a second opinion from a system without a ledger?

**Added.**
* B1-checked-by-B2 and B1-checked-by-B1-9 (`scripts/rescore_ledgers.py`): B1's verdict where B2 (or B1-9) gives the
  same verdict, NEI otherwise. B1-SC is not used as a check: its four sampled verdicts equal B1's on every
  prompt_dev claim that B1 answers, so it would veto nothing.
* `scripts/veto_analysis.py`: for each check, its vetoes (B1 SUPPORTS/REFUTES turned NEI), how many were B1 errors
  (veto precision), the exact hypergeometric p of that precision against a random veto of the same size, and
  macro-F1 and unsupported rate over 10,000 random vetoes of the same size (seed 13).
* Comparisons B1-checked-by-B2 → B1-checked and B1-checked-by-B1-9 → B1-checked (`check_comparisons` in
  `configs/experiment_matrix.yaml`), a fourth Holm family.

**prompt_dev (199 claims).** B1 answers 127 claims with SUPPORTS or REFUTES, 40 of them wrong (0.315). Vetoes, of
which wrong (veto precision, p): M1 24, 14 (0.583, p 0.002); M1-anch 26, 16 (0.615, p < 0.001); B2 14, 10 (0.714,
p 0.001); B1-9 8, 4 (0.500, p 0.216). Macro-F1 (unsupported): B1-checked 0.735 (0.247), against 0.673 (0.288) for a
random veto of 24; B1-checked-by-B2 0.733 (0.301); B1-checked-by-B1-9 0.711 (0.344). B1-checked-by-B2 → B1-checked:
macro-F1 +0.002 (Holm p 0.951), unsupported −0.054 [−0.129, +0.021]; B1-checked-by-B1-9 → B1-checked: +0.024
(p 0.590), unsupported −0.097 [−0.161, −0.032].

**Reading fixed before the test sets.** The ledger counts as a better check than a second opinion in general only
if, on both test sets, (1) its veto precision beats a random veto of the same size (p < 0.05), and (2) against
B1-checked-by-B2 its macro-F1 is not significantly lower while its unsupported rate is lower with an interval
below zero. Otherwise the safety gain is credited to a second opinion, not to the ledger.

**Applied once** to the stored dev and SciFact-Open runs after this entry.

**Run log, 2026-10-30, 18:20 IST — amendment 6 applied once to dev and SciFact-Open.**
* B1's SUPPORTS/REFUTES answers: dev 216, 61 wrong (0.282); SciFact-Open 199, 39 wrong (0.196). B1-SC unanimity
  would veto 5 and 8 of them.
* Vetoes, of which wrong (veto precision, p against a random veto of the same size), dev / SciFact-Open:
  M1 42, 20 (0.476, p 0.002) / 29, 13 (0.448, p < 0.001); M1-anch 51, 27 (0.529, p < 0.001) / 45, 16 (0.356,
  p 0.003); B2 28, 17 (0.607, p < 0.001) / 24, 15 (0.625, p < 0.001); B1-9 16, 10 (0.625, p 0.003) / 15, 8 (0.533,
  p 0.003).
* Macro-F1 (unsupported), dev / SciFact-Open: B1-checked 0.720 (0.250) / 0.776 (0.153), against 0.666 (0.316) /
  0.739 (0.191) for random vetoes of the same size; B1-checked-by-B2 0.725 (0.312) / 0.792 (0.188);
  B1-checked-by-B1-9 0.735 (0.339) / 0.794 (0.200).
* B1-checked-by-B2 → B1-checked: macro-F1 −0.005 (Holm p 0.854) / −0.016 (p 0.553); unsupported −0.062
  [−0.143, +0.018] / −0.035 [−0.094, +0.024]. B1-checked-by-B1-9 → B1-checked: −0.015 / −0.018 (p 0.854 / 0.553);
  unsupported −0.089 [−0.152, −0.027] / −0.047 [−0.094, −0.012].
* Reading as fixed above: criterion 1 holds (the ledger's vetoes find B1's errors far better than chance on both
  sets); criterion 2 fails (against B2's check the unsupported-rate intervals include zero on both sets). The
  safety gain is credited to a second opinion, not to the ledger: B2 vetoes fewer answers but more precisely, at
  under half the ledger check's tokens.
