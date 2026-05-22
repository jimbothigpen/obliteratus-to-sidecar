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
VENV=/mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/obliteratus-to-sidecar/src/jimbothigpen/obliteratus-to-sidecar/.venv

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

# Focused matrix v2 — drove the queue from the OBLITERATUS community
# telemetry dataset (pliny-the-prompter/OBLITERATUS-TELEMETRY) instead of
# guessing methods. For each model, run the top-2 methods by community
# composite score in the model's bucket (model-specific where the dataset
# has direct records, else bucket aggregate).
#
# Skipped:
#  - GLM-4.7-Flash (per user, deferred to keep matrix wall time bounded)
#  - Jackrong/Qwopus3.5-9B-v3 (zero community telemetry; swapped for the
#    closest vanilla model with strong community data — Qwen/Qwen3.5-9B
#    has 3,788 telemetry records, n=2,999 for `advanced` alone)

# --- google/gemma-4-E2B-it (Dense Standard Tiny bucket, n=8) ---
# bucket top: advanced (composite 0.7023), runner-up: basic (0.4092).
run_one google/gemma-4-E2B-it       advanced         0
# gemma/basic — DISABLED 2026-05-08: upstream OBLITERATUS bug. _distill()
# crashes with `cannot convert float NaN to integer` at abliterate.py:1732
# in the per-layer refusal-strength bar-print loop. Reproducible. Likely
# triggered by NaN-valued activations from Gemma-4's SSM hybrid layers.
# Re-enable after upstream patch.
# run_one google/gemma-4-E2B-it       basic            0

# --- Qwen/Qwen3.5-9B (model-specific data, n=3,788) ---
# top: advanced (2,999 runs), runner-up: basic (548 runs).
run_one Qwen/Qwen3.5-9B             advanced         0
run_one Qwen/Qwen3.5-9B             basic            0

# --- openai/gpt-oss-20b — DISABLED 2026-05-08 ---
# Both `surgical` and `optimized` with per_expert=1 hit upstream OBLITERATUS
# shape bug in _compute_expert_granular_directions() at abliterate.py:2675:
#   RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x32 and 512x2880)
# The EGA code expects (n_experts, hidden) but gets a wrong-shaped weight tensor
# for gpt-oss-20b's 32 expert × 2880 hidden combination. Both methods route
# through this code path under per_expert=1. Re-enable after upstream patch.
# run_one openai/gpt-oss-20b          surgical         1
# run_one openai/gpt-oss-20b          optimized        1

echo "[$(date -Iseconds)] === validation matrix end ===" >> "$RUN_DIR/_overall.log"
