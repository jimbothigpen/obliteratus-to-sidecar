# obliteratus-to-sidecar

Run [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) as a refusal-direction
extraction backend and emit a runtime sidecar GGUF compatible with the
[frankenturbo2](https://github.com/jimbothigpen/frankenturbo2) llama.cpp fork's
sidecar adapter (`--sidecar-vectors path.abl.gguf`).

## What it does

1. Loads an HF safetensors model.
2. Runs OBLITERATUS's `AbliterationPipeline` end-to-end (probe → distill →
   excise → verify → rebirth). OBLITERATUS produces a weight-modified safetensors
   and updates `pipeline.refusal_directions` / `pipeline.refusal_subspaces` /
   per-expert direction tables.
3. Captures the per-layer (and optional per-expert) directions from the
   pipeline object after completion.
4. Writes them into the sidecar GGUF schema (`abl.layer_indices`,
   `abl.directions`, plus optional `abl.expert_layer_indices` /
   `abl.expert_directions` for MoE).
5. Optionally retains or deletes OBLITERATUS's modified safetensors. Keeping
   it lets you A/B test the "weight-baked" reference against the
   "base + sidecar" path on the same prompts.

## Why bother

Our in-house Arditi-style direction extraction did not flip refusal on Qwopus3.5
thinking models (memory: `abliteration_thinking_model_failure`). OBLITERATUS
implements ~10 method presets including LEACE, whitened SVD, Wasserstein-optimal,
and Bayesian-optimized direction extraction with CoT-aware refinement. This tool
delegates direction discovery to OBLITERATUS while keeping our deployable
runtime artifact (sidecar GGUF + base GGUF, no weight modification).

## Install

```bash
cd /usr/src/llama-forks/obliteratus-to-sidecar
pip install -e .
```

OBLITERATUS itself is the heavy dependency; install it in the same venv (it
pulls in torch/transformers/bitsandbytes).

## Usage

Basic dense extraction (works for any architecture):

```bash
obliteratus-to-sidecar \
    --hf-model google/gemma-4-E2B-it \
    --method optimized \
    --output /mnt/cephfs/0/Container/.../gemma-4-E2B-it.abl.gguf
```

MoE per-expert extraction:

```bash
obliteratus-to-sidecar \
    --hf-model zai-org/GLM-4.7-Flash \
    --method aggressive \
    --per-expert \
    --output /mnt/cephfs/0/Container/.../GLM-4.7-Flash.abl.gguf
```

The tool logs the OBLITERATUS pipeline progress in real-time and prints the
final sidecar location.

## Output schema

See `frankenturbo2/tools/abliterate/write_sidecar.py` docstring for the full
GGUF schema. Per-expert keys are populated only when `--per-expert` is set.
