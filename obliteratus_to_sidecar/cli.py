"""CLI entry point for obliteratus-to-sidecar."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hf-model", required=True,
                   help="HuggingFace model id (or local path) to extract directions from.")
    p.add_argument("--method", required=True,
                   help="OBLITERATUS method preset (basic, optimized, advanced, cascade, "
                        "aggressive, sota, inverted, moe_aware, sparse_surgery, "
                        "huihui_compat, etc.). See `obliteratus presets` for the full list.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output sidecar GGUF path (e.g. /mnt/cephfs/.../model.abl.gguf)")
    p.add_argument("--arch", default=None,
                   help="Override the auto-detected llama.cpp arch string baked into the "
                        "sidecar's abl.arch KV (must match the GGUF's general.architecture). "
                        "Auto-detect is config-architecture-class based; see arch_map.py.")
    p.add_argument("--per-expert", action="store_true",
                   help="Enable per-expert direction extraction (MoE only). Falls back "
                        "gracefully on dense models — OBLITERATUS just won't emit any "
                        "per-expert dirs.")
    p.add_argument("--obliteratus-output-dir", type=Path, default=None,
                   help="Where OBLITERATUS writes its modified safetensors. Defaults to "
                        "<output-dir>/<slug>.obliteratus_out.")
    p.add_argument("--no-keep-modified", action="store_true",
                   help="Delete OBLITERATUS's modified safetensors after capturing the "
                        "directions. Saves disk but means we lose the weight-baked baseline "
                        "for A/B comparison.")
    p.add_argument("--skip-rebirth", action="store_true",
                   help="Skip OBLITERATUS's _rebirth stage entirely (no modified safetensors "
                        "ever written). Faster + sidesteps the crashy multi-shard save under "
                        "memory pressure. Use when sidecar-only deployment is enough.")
    p.add_argument("--device", default="auto", help="torch device (default: auto)")
    p.add_argument("--dtype", default="float16",
                   help="torch dtype (default: float16). Choices: float32, float16, bfloat16.")
    p.add_argument("--quantization", default=None,
                   help="OBLITERATUS quantization mode for loading (4bit, 8bit, none).")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Pass trust_remote_code=True to OBLITERATUS / transformers loaders. "
                        "Required for some 2026 archs (e.g. Gemma4, GLM4-MoE-Lite).")
    p.add_argument("--n-directions", type=int, default=None,
                   help="Override n_directions (default: from method preset)")
    p.add_argument("--direction-method", default=None,
                   help="Override direction extraction method (diff_means, svd, "
                        "whitened_svd, leace, wasserstein_optimal)")
    p.add_argument("--refinement-passes", type=int, default=None,
                   help="Override refinement passes (default: from method preset)")
    p.add_argument("--regularization", type=float, default=None,
                   help="Override regularization (default: from method preset)")
    p.add_argument("--large-model-mode", action="store_true",
                   help="Enable OBLITERATUS large-model memory mode (CPU offload, etc.)")
    p.add_argument("--verify-sample-size", type=int, default=None,
                   help="Override verify-stage sample size (default: from method preset)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose logging")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="obliteratus-to-sidecar",
        description=(
            "Run OBLITERATUS as a refusal-direction extraction backend and emit a "
            "runtime sidecar GGUF for the frankenturbo2 llama.cpp fork."
        ),
    )
    _add_args(p)
    args = p.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # Echo OBLITERATUS log lines to stderr in real-time.
    def _on_log(msg: str) -> None:
        print(f"[obliteratus] {msg}", file=sys.stderr, flush=True)

    from obliteratus_to_sidecar.adapter import run_extraction

    try:
        result = run_extraction(
            hf_model=args.hf_model,
            method=args.method,
            output_path=args.output,
            arch=args.arch,
            obliteratus_output_dir=args.obliteratus_output_dir,
            per_expert=args.per_expert,
            device=args.device,
            dtype=args.dtype,
            quantization=args.quantization,
            n_directions=args.n_directions,
            direction_method=args.direction_method,
            refinement_passes=args.refinement_passes,
            regularization=args.regularization,
            large_model_mode=args.large_model_mode,
            verify_sample_size=args.verify_sample_size,
            keep_modified_safetensors=not args.no_keep_modified,
            trust_remote_code=args.trust_remote_code,
            skip_rebirth=args.skip_rebirth,
            on_log=_on_log,
        )
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            raise
        return 1

    print()
    print(f"  Sidecar GGUF (full):     {result.sidecar_path}")
    if result.sidecar_selected_path is not None:
        print(f"  Sidecar GGUF (selected): {result.sidecar_selected_path}")
    print(f"  OBLITERATUS output dir:  {result.obliteratus_output_dir}")
    print(f"  Arch:                    {result.arch}")
    print(f"  n_embd:                  {result.n_embd}")
    print(f"  Dense layers (captured): {result.n_dense_layers}")
    if result.sidecar_selected_path is not None:
        print(f"  Dense layers (selected): {result.n_dense_layers_selected}")
    print(f"  Expert layers:           {result.n_expert_layers}"
          f" (with {result.n_experts_per_layer} experts each)")
    print(f"  Method:                  {result.method}")
    print()
    print("Try with frankenturbo2 llama-cli:")
    print(f"  llama-cli -m <base.gguf> --sidecar-vectors {result.sidecar_path} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
