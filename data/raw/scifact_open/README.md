# SciFact-Open claim files (second held-out test set)

Source: the official SciFact-Open release, `https://scifact.s3.us-west-2.amazonaws.com/scifact-open/latest/scifact_open.tar.gz`
(Wadden et al., Findings of EMNLP 2022), files `data/claims.jsonl` and `data/claims_metadata.jsonl`, unmodified.

* 279 claims; every id is a SciFact *test* claim id with identical claim text, so these files publish labels
  for the hidden SciFact test claims.
* Each evidence entry has `provenance`: `citation` (rationale sentences annotated by experts) or `pooling`
  (label annotated by humans, rationale sentences predicted by a model; see the release's `doc/data.md`).
* The 500,000-abstract SciFact-Open corpus is **not** included (889 MB). The main setting uses only evidence
  whose abstract is in SciFact's 5,183-abstract corpus; there, the sentence lists are identical to SciFact's
  (checked for all 196 such abstracts), so rationale sentence numbers carry over.
* Loader: `oblig.data.scifact.load_scifact_open`. Held-out: scored only with `--heldout`.
