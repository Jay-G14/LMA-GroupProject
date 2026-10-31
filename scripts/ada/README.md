# Running on IIIT Ada

1. `scripts/ada/sync_to_ada.sh` copies code and SciFact data to `~/LMA-Project` (no git history or PUBHEALTH).
2. Once: `sbatch scripts/ada/setup_env.sh` (Python 3.11 env in `.venv`, Qwen3-8B-AWQ and BGE-base weights in
   `.hf/` (home; `/share1` is not mounted on compute nodes), BGE-base corpus embeddings).
3. Per eval set, two one-GPU jobs (obligation systems share one cache; baselines in the other):
   ```
   sbatch --job-name=oblig_dev_A --export=ALL,EVAL_SET=dev,TAG=dev,GROUP=A scripts/ada/run_scifact_vllm.sh
   sbatch --job-name=oblig_dev_B --export=ALL,EVAL_SET=dev,TAG=dev,GROUP=B,SC_SAMPLES=5,RETRIEVAL_EVAL=1 scripts/ada/run_scifact_vllm.sh
   ```
   Held-out sets (`dev`, `scifact_open`) are run only after the pilot on `prompt_dev` (Gate 3).
4. Results land in `~/LMA-Project/to_transfer/<TAG>/` (predictions, metrics, model revision). Copy them back
   with `rsync -az jayant.g@10.4.24.30:~/LMA-Project/to_transfer/ to_transfer/` and compare locally:
   `python scripts/compare_systems.py --eval-set dev --prefix dev_ --name dev --results-dir to_transfer/dev`.
