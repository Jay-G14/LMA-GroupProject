#!/bin/bash
# SciFact obligation-ledger runs on IIIT Ada: Qwen3-8B (official AWQ 4-bit release) served by vLLM on one GPU,
# then a group of systems on one eval set. Needs scripts/ada/setup_env.sh to have run once.
# Based on ~/.claude/templates/slurm_ada_template.sh (the user's standing Ada rules). Adaptations:
#  - 1 GPU per job: Qwen3-8B-AWQ fits one 11 GB RTX 2080 Ti (the full fp16 model does not), and single-GPU jobs
#    schedule on the many partly used 2080 Ti nodes; 4-GPU requests fell back to GTX 1080 Ti nodes.
#  - vLLM needs compute capability >= 7.0: keeps --prefer=2080ti and, exactly like the GPU-count check,
#    resubmits itself excluding a node whose GPU is older (GTX 1080 class, compute 6.1).
# Submit from the repo root, e.g.:
#   sbatch --job-name=oblig_pilot_A --export=ALL,EVAL_SET=prompt_dev,TAG=pilot,GROUP=A scripts/ada/run_scifact_vllm.sh
# Env: EVAL_SET (prompt_dev|dev|scifact_open), TAG (folder in to_transfer/), GROUP (A = obligation systems sharing
#      one cache, B = baselines), SYSTEMS (overrides GROUP; "m2:claimfollow m2:noconsol" = the M2 ablations),
#      WORKERS (default 24), N, SC_SAMPLES, RETRIEVAL_EVAL=1.
#SBATCH --job-name=oblig
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --prefer=2080ti
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=jayant.g@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x.log
#SBATCH --error=logs/%x.err
# Fresh .log/.err on every submission; only an automatic resubmission appends (its sbatch passes --open-mode=append).
#SBATCH --open-mode=truncate

set -e
EVAL_SET=${EVAL_SET:-prompt_dev}; TAG=${TAG:-$EVAL_SET}; GROUP=${GROUP:-A}; WORKERS=${WORKERS:-24}
if [ -z "$SYSTEMS" ]; then
    if [ "$GROUP" = "A" ]; then SYSTEMS="m1 m1_llmagg m2 m2_random m1_shared m1_aligned m1_untyped"
    else SYSTEMS="b0 b1 m2_claimiter b1_9 b2 b1_sc"; fi
fi
SCRIPT=scripts/ada/run_scifact_vllm.sh
echo "Job started on $(hostname) at $(date): EVAL_SET=$EVAL_SET TAG=$TAG GROUP=$GROUP SYSTEMS=$SYSTEMS"
BASE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$BASE_DIR" || { echo "Error: Could not cd into $BASE_DIR"; exit 1; }
GPU_NAMES=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)
echo "GPUs on $(hostname): $(echo "$GPU_NAMES" | sort | uniq -c | tr -s ' \n' ' ')"

resubmit() {  # $1 = reason. Template mechanism: exclude this node plus earlier exclusions, at most 3 times.
    ATTEMPT=${RESUBMIT_ATTEMPT:-0}; MAX_RESUBMITS=3
    PREV_EXC=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | grep -oP 'ExcNodeList=\K\S+' | grep -v '(null)' || true)
    EXC="${PREV_EXC:+$PREV_EXC,}$(hostname -s)"
    echo "ERROR: $1"
    if [ -n "$SLURM_JOB_ID" ] && [ "$ATTEMPT" -lt "$MAX_RESUBMITS" ]; then
        NEW_JOB=$(RESUBMIT_ATTEMPT=$((ATTEMPT + 1)) sbatch --parsable --open-mode=append --exclude="$EXC" \
                  --job-name="$SLURM_JOB_NAME" "$SCRIPT") || NEW_JOB=""
        if [ -n "$NEW_JOB" ]; then echo ">>> Resubmitted as job $NEW_JOB (attempt $((ATTEMPT + 1)) of $MAX_RESUBMITS), excluding: $EXC"
        else echo ">>> Automatic resubmission failed. Resubmit by hand: sbatch --exclude=$EXC $SCRIPT"; fi
    else echo ">>> Not resubmitting (attempt $ATTEMPT of $MAX_RESUBMITS). Resubmit by hand: sbatch --exclude=$EXC $SCRIPT"; fi
    exit 1
}

# Environment built by setup_env.sh (Python 3.11 in the experiment folder).
[ -f .venv/.complete ] || { echo "ERROR: .venv missing; run: sbatch scripts/ada/setup_env.sh"; exit 1; }
export PATH="$BASE_DIR/.venv/bin:$PATH"
mkdir -p "to_transfer/$TAG" logs

NUM_GPUS=$(grep -oP '(?<=--gres=gpu:)[0-9]+' "${BASH_SOURCE[0]}" | head -1)
read ACTUAL_GPUS MIN_CC < <(python -c "import torch; n=torch.cuda.device_count(); print(n, min([10*a+b for a,b in (torch.cuda.get_device_capability(i) for i in range(n))] or [0]))")
echo "NUM_GPUS wanted = $NUM_GPUS, torch sees = $ACTUAL_GPUS, min compute capability = $MIN_CC"
if [ "$ACTUAL_GPUS" -lt "$NUM_GPUS" ]; then resubmit "wanted $NUM_GPUS GPUs but $(hostname -s) shows $ACTUAL_GPUS"; fi
if [ "$MIN_CC" -lt 70 ]; then resubmit "vLLM needs compute capability >= 7.0; $(hostname -s) has $MIN_CC (GTX 1080 class)"; fi

# Weights were downloaded by setup_env.sh; resolve them offline.
export HF_HOME="$BASE_DIR/.hf" HF_HUB_OFFLINE=1     # /share1 is not mounted on compute nodes
MODEL_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B-AWQ', local_files_only=True))")
echo "model: $MODEL_PATH" | tee "to_transfer/$TAG/model_revision_${GROUP}.txt"

# Data integrity check, then stage data on node-local /scratch (removed on exit, with the server).
python scripts/ada/check_data.py data
SCRATCH_DATA="/scratch/$USER/oblig_data_${SLURM_JOB_ID:-manual}"
VLLM_PID=""
cleanup() { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null; rm -rf "$SCRATCH_DATA" "$OUTLINES_CACHE_DIR"; }
trap cleanup EXIT
if mkdir -p "$SCRATCH_DATA/raw" 2>/dev/null; then
    rsync -a data/raw/scifact data/raw/scifact_open "$SCRATCH_DATA/raw/"; rsync -a data/splits "$SCRATCH_DATA/"
    export OBLIG_DATA_DIR="$SCRATCH_DATA"; echo "data staged to $SCRATCH_DATA ($(du -sh "$SCRATCH_DATA" | cut -f1))"
else echo "WARNING: /scratch/$USER not writable on $(hostname) — reading data from home"; fi
# vLLM compiles JSON schemas with item or length limits through outlines, whose SQLite cache defaults to
# ~/.cache/outlines on NFS home; jobs on different nodes sharing it corrupted it. One node-local cache per job.
export OUTLINES_CACHE_DIR="/tmp/oblig_outlines_${SLURM_JOB_ID:-manual}"
[ -d "$SCRATCH_DATA" ] && OUTLINES_CACHE_DIR="$SCRATCH_DATA/outlines"
mkdir -p "$OUTLINES_CACHE_DIR"

# BGE-base corpus embeddings: normally cached by setup_env.sh; if missing, compute them on the GPU before the
# server takes it (under a lock, so concurrent jobs do not race on the cache file).
exec 7>"$BASE_DIR/.embeddings.lock"; flock 7
OBLIG_DENSE_DEVICE=cuda python -c "import sys; sys.path.insert(0, 'src'); from oblig.data.scifact import load_corpus; from oblig.retrieval.factory import build_retriever; build_retriever('hybrid_bm25_bge_base', load_corpus())"
flock -u 7

# vLLM server on a free port (several 1-GPU jobs can share a node). V0 engine on Turing (V1 needs compute >= 8.0).
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
if [ "$MIN_CC" -lt 80 ]; then export VLLM_USE_V1=0; fi
export OTEL_SDK_DISABLED=true   # silence a stray OpenTelemetry exporter (nothing listens on :4318)
VLOG="logs/vllm_${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-manual}.log"
# No --enable-prefix-caching: its Triton prefix-prefill kernel fails to compile on Turing ("PassManager::run
# failed" in vllm/attention/ops/prefix_prefill.py). Memory: 0.85 utilisation and 8,192-token batches leave room
# for AWQ's dequantisation buffers on a large prefill (0.92 / 12,288 ran out of memory in the second pilot).
start_vllm() {
    python -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" --served-model-name Qwen/Qwen3-8B \
        --quantization awq --dtype half --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 32 \
        --gpu-memory-utilization 0.85 --seed 13 --port "$PORT" --disable-log-requests >> "$VLOG" 2>&1 &
    VLLM_PID=$!
    for i in $(seq 1 90); do
        if curl -sf "http://localhost:$PORT/v1/models" > /dev/null; then echo ">>> vLLM ready on port $PORT after $((i * 10))s"; return 0; fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "ERROR: vLLM exited"; tail -40 "$VLOG"; return 1; fi
        sleep 10
    done
    echo "ERROR: vLLM not ready after 15 min"; return 1
}
ensure_vllm() {   # restart the server if it died (e.g. an out-of-memory crash); runs resume claim by claim
    if ! curl -sf "http://localhost:$PORT/health" > /dev/null; then
        echo ">>> $(date +%H:%M:%S) vLLM is down; restarting"
        [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null; sleep 5
        start_vllm || exit 1
    fi
}
start_vllm || exit 1

# Systems in an order that reuses cached calls (B1 -> M2-claimiter; M1 -> M1-LLMagg, M2, M2-random and the M1
# ablations' planner calls). One cache per job, so concurrent jobs never share an SQLite file.
HELDOUT=""; case "$EVAL_SET" in dev|scifact_open) HELDOUT="--heldout";; esac
EXTRA=""; [ -n "$N" ] && EXTRA="$EXTRA --n $N"; [ -n "$SC_SAMPLES" ] && EXTRA="$EXTRA --sc-samples $SC_SAMPLES"
[ -n "$M2_FOLLOWUP" ] && EXTRA="$EXTRA --m2-followup $M2_FOLLOWUP"; [ -n "$M2_CONSOLIDATE" ] && EXTRA="$EXTRA --m2-consolidate $M2_CONSOLIDATE"
SUFFIX=${SUFFIX:-}   # appended to experiment ids, e.g. _noconsol for an ablation run
set +e
# A SYSTEMS entry may name an M2 ablation (protocol amendments 1 and 3), run with the group's cache:
#   m2:claimfollow = follow-up search with the claim appended only; m2:noconsol = no consolidation audit.
for ENTRY in $SYSTEMS; do
    S=${ENTRY%%:*}; VARIANT=""; SFX=$SUFFIX
    case "$ENTRY" in
        *:claimfollow) VARIANT="--m2-followup claim"; SFX="${SUFFIX}_claimfollow" ;;
        *:noconsol)    VARIANT="--m2-consolidate off"; SFX="${SUFFIX}_noconsol" ;;
        *:*)           echo "WARNING: unknown variant in $ENTRY; skipped"; continue ;;
    esac
    for ATTEMPT in 1 2 3; do       # an aborted run (server failure) resumes after a server restart
        ensure_vllm
        echo ">>> $(date +%H:%M:%S) system $ENTRY (attempt $ATTEMPT)"
        PYTHONUNBUFFERED=1 python scripts/run_scifact.py --system "$S" --eval-set "$EVAL_SET" $HELDOUT $EXTRA $VARIANT \
            --profile qwen3_8b_vllm --base-url "http://localhost:$PORT/v1" --workers "$WORKERS" \
            --results-dir "to_transfer/$TAG" --cache "${CACHE:-to_transfer/$TAG/cache/llm_${GROUP}.sqlite}" \
            --exp-id "${TAG}_$S$SFX" && break
        echo "WARNING: system $ENTRY attempt $ATTEMPT failed"
    done
done
if [ -n "$RETRIEVAL_EVAL" ]; then
    python scripts/eval_retrieval_scifact.py --eval-set "$EVAL_SET" $HELDOUT --configs bm25 dense_bge_base hybrid_bm25_bge_base \
        > "to_transfer/$TAG/retrieval_stdout.txt" 2>&1 && cp "results/metrics/retrieval_scifact_${EVAL_SET}.json" "to_transfer/$TAG/" \
        || echo "WARNING: retrieval evaluation failed"
fi
set -e
echo ">>> Pipeline complete at $(date). Results in to_transfer/$TAG"
