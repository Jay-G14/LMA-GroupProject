#!/bin/bash
# One-time setup for the GPU jobs (CPU only, so no GPU waits on installs):
#   Python 3.11 env in the experiment folder (.venv; vLLM 0.9.2 needs Python <= 3.12, the login shell's conda
#   base is 3.14), model weights in the experiment folder (.hf; /share1 is not mounted on compute nodes),
#   BGE-base corpus embeddings (cached in home). Wheels are staged on node-local /scratch and removed after.
# Submit from the repo root:  sbatch scripts/ada/setup_env.sh
#SBATCH --job-name=oblig_setup
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --mail-user=jayant.g@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x.log
#SBATCH --error=logs/%x.err
#SBATCH --open-mode=truncate
set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Setup started on $(hostname) at $(date)"
# Single connections to PyPI run at ~0.1 MB/s here, so pip only resolves (metadata); fetch_wheels.py downloads
# the wheels with parallel byte ranges (sha256-checked) and pip installs offline from them.
if [ ! -f .venv/.complete ]; then
    if [ "$(.venv/bin/python -c 'import sys; print(sys.version_info[:2] == (3, 11))' 2>/dev/null)" != "True" ]; then
        rm -rf .venv
        ~/miniconda3/bin/conda create -y -p "$PWD/.venv" -c conda-forge --override-channels python=3.11 pip
    fi
    WHEELS=/scratch/$USER/oblig_wheels_${SLURM_JOB_ID:-manual}
    mkdir -p "$WHEELS"; trap 'rm -rf "$WHEELS" "$WHEELS.report.json"' EXIT
    .venv/bin/pip install --dry-run --ignore-installed --quiet --report "$WHEELS.report.json" -r scripts/ada/requirements-gpu.txt
    .venv/bin/python scripts/ada/fetch_wheels.py "$WHEELS.report.json" "$WHEELS"
    .venv/bin/pip install --no-index --find-links "$WHEELS" -r scripts/ada/requirements-gpu.txt
    touch .venv/.complete
fi
# Later runs only reconcile pins (small downloads; pip alone is fine for those).
[ -f .venv/.complete ] && .venv/bin/pip install -r scripts/ada/requirements-gpu.txt
export PATH="$PWD/.venv/bin:$PATH"
python --version; python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
python -c "import vllm.transformers_utils.config" || { echo "ERROR: vLLM cannot import its configs (transformers version clash?)"; exit 1; }
export HF_HOME="$PWD/.hf" HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" to_transfer
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-8B-AWQ", "BAAI/bge-base-en-v1.5"):
    print(repo, snapshot_download(repo, max_workers=16))
PY
export HF_HUB_OFFLINE=1
python scripts/ada/check_data.py data
OBLIG_DENSE_DEVICE=cpu python -c "import sys; sys.path.insert(0, 'src'); from oblig.data.scifact import load_corpus; from oblig.retrieval.factory import build_retriever; build_retriever('hybrid_bm25_bge_base', load_corpus()); print('BGE-base corpus embeddings cached')"
echo "Setup complete at $(date)"
