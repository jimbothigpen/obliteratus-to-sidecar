#!/bin/bash
# Drive the v1 validation matrix: 3 architectures x 3 methods each = 9 runs.
# Each run produces a sidecar GGUF on ceph + logs the OBLITERATUS pipeline.
# Modified safetensors are kept (ceph has space) for A/B comparison later.
#
# Run sequentially because each invocation owns the ai00 GPU.
#
# Skips combos whose model isn't yet present in HF cache (e.g. when Qwopus
# download is still in progress).

set -uo pipefail

CEPH=/mnt/cephfs/0/Container/systems/ai00/users/builduser
RUN_DIR=$CEPH/validation-runs
SIDECAR_DIR=$CEPH/sidecars
HF_CACHE=$CEPH/hf-cache
# Offload to LOCAL nvme on ai00 — ceph is 7-10x slower for offload reads,
# and pointing offload at ceph triggered Accelerate to pre-stage buffer pages
# in CPU RAM that pushed Qwopus 9B into OOM territory on ai00's 32GiB system.
OFFLOAD_DIR=/home/builduser/offload-ai00
VENV=/usr/src/llama-forks/obliteratus-to-sidecar/.venv

mkdir -p "$RUN_DIR" "$SIDECAR_DIR" "$OFFLOAD_DIR"

slug() {
    echo "$1" | tr '/' '_'
}

# Reclaim kernel page-cache between combos. ai00 has only 32 GiB system RAM
# and the previous combo's mmapped weights / staged offload buffers stay in
# buff/cache for a while. Dropping caches reduces baseline RSS pressure
# before the next model gets staged. sudo -n is passwordless on these hosts.
drop_caches() {
    sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
        && echo "[$(date -Iseconds)] dropped page-cache" >> "$RUN_DIR/_overall.log" \
        || echo "[$(date -Iseconds)] drop_caches failed (continuing)" >> "$RUN_DIR/_overall.log"
}

is_cached() {
    local m=$1
    [[ -d "$HF_CACHE/models--$(echo "$m" | tr '/' '-')" ]] || \
    [[ -d "$HF_CACHE/models--$(echo "$m" | tr '/' '-' | sed 's/-/--/')" ]] || \
    find "$HF_CACHE" -maxdepth 1 -type d -name "models--$(echo "$m" | sed 's|/|--|g')" 2>/dev/null | grep -q .
}

run_one() {
    local model=$1 method=$2 per_expert=$3
    local s=$(slug "$model")
    local outdir=$SIDECAR_DIR/$s
    local sidecar=$outdir/${method}.abl.gguf
    local logfile=$RUN_DIR/${s}__${method}.log
    local statusfile=$RUN_DIR/${s}__${method}.status

    if ! is_cached "$model"; then
        echo "[$(date -Iseconds)] SKIP (not cached): $model" | tee -a "$RUN_DIR/_overall.log"
        echo "skipped: not cached" > "$statusfile"
        return 0
    fi

    if [[ -f "$sidecar" ]]; then
        echo "[$(date -Iseconds)] SKIP (already done): $sidecar" | tee -a "$RUN_DIR/_overall.log"
        echo "skipped: existing" > "$statusfile"
        return 0
    fi

    mkdir -p "$outdir"
    drop_caches
    echo "[$(date -Iseconds)] START: $model / $method (per_expert=$per_expert)" | tee -a "$RUN_DIR/_overall.log"

    local args=(--hf-model "$model" --method "$method" --output "$sidecar"
                --trust-remote-code --skip-rebirth
                --offload-folder "$OFFLOAD_DIR/$s.$method" -v)
    [[ "$per_expert" == "1" ]] && args+=(--per-expert)

    HF_HUB_CACHE="$HF_CACHE" \
    TMPDIR=/home/builduser/.pip-tmp \
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    OBLITERATUS_MAX_MEMORY_GPU=80GiB \
    OBLITERATUS_MAX_MEMORY_CPU=12GiB \
        "$VENV/bin/obliteratus-to-sidecar" "${args[@]}" >"$logfile" 2>&1
    local rc=$?

    echo "rc=$rc" > "$statusfile"
    echo "[$(date -Iseconds)] END: $model / $method (rc=$rc)" | tee -a "$RUN_DIR/_overall.log"
    return $rc
}

echo "[$(date -Iseconds)] === validation matrix start ===" >> "$RUN_DIR/_overall.log"

# --- Gemma-4-E2B-it (DENSE+REASONING, ~5B effective) ---
run_one google/gemma-4-E2B-it       optimized        0
run_one google/gemma-4-E2B-it       gabliteration    0
run_one google/gemma-4-E2B-it       spectral_cascade 0

# --- Qwopus3.5-9B (DENSE+REASONING, hybrid linear/softmax 3:1) ---
run_one Jackrong/Qwopus3.5-9B-v3    spectral_cascade 0
run_one Jackrong/Qwopus3.5-9B-v3    aggressive       0
run_one Jackrong/Qwopus3.5-9B-v3    optimized        0

# --- gpt-oss-20b (SMALL_MOE+REASONING, 20B/A3.6B, Top-4 of 32 experts) ---
# Pivot from Nemotron-3 — Mamba-2 hybrids hard-import mamba_ssm at module
# load (CUDA-only, no CPU fallback) and are unviable on ai01. gpt-oss is
# a standard transformer + MoE arch, supported in frankenturbo2 as
# LLM_ARCH_OPENAI_MOE; no kernel gap. Smaller (~13 GB MXFP4 vs GLM's 59 GB
# BF16) so ordered before GLM to land a working MoE+per-expert sidecar
# faster.
run_one openai/gpt-oss-20b          surgical         1
run_one openai/gpt-oss-20b          aggressive       1
run_one openai/gpt-oss-20b          nuclear          1

# --- GLM-4.7-Flash (SMALL_MOE+REASONING, 30B-A3B, 64 experts) ---
run_one zai-org/GLM-4.7-Flash       surgical         1
run_one zai-org/GLM-4.7-Flash       aggressive       1
run_one zai-org/GLM-4.7-Flash       nuclear          1

echo "[$(date -Iseconds)] === validation matrix end ===" >> "$RUN_DIR/_overall.log"
