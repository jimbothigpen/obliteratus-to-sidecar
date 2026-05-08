"""Core adapter: run OBLITERATUS pipeline, capture directions, emit sidecar GGUF.

The capture point is post-`pipeline.run()`. By then OBLITERATUS has finalized
`refusal_directions` (per-layer dense), `refusal_subspaces` (per-layer
multi-direction), and `_expert_directions` (per-(layer, expert) for MoE) on
the pipeline object. We snapshot those and emit the GGUF.

OBLITERATUS will also have written its weight-modified safetensors to the
output directory; we leave it in place by default so the user can convert it
to GGUF and A/B compare against `base + sidecar` on the same prompts.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    sidecar_path: Path
    sidecar_selected_path: Path | None
    obliteratus_output_dir: Path
    n_dense_layers: int
    n_dense_layers_selected: int
    n_expert_layers: int
    n_experts_per_layer: int
    arch: str
    n_embd: int
    method: str


def _stack_dense(refusal_directions: dict[int, "torch.Tensor"]) -> tuple[list[int], np.ndarray]:
    """Convert OBLITERATUS pipeline.refusal_directions to (chosen_layers, dense_dirs).

    Returns:
        chosen_layers: sorted list of layer indices that have a direction
        dense_dirs:    numpy float32 [n_chosen_layers, n_embd]
    """
    chosen_layers = sorted(refusal_directions.keys())
    if not chosen_layers:
        raise RuntimeError("OBLITERATUS produced no refusal directions — pipeline failed?")
    arr = np.stack(
        [refusal_directions[L].detach().cpu().float().numpy() for L in chosen_layers],
        axis=0,
    )
    return chosen_layers, arr.astype(np.float32, copy=False)


def _stack_expert(
    expert_directions: dict[int, dict[int, "torch.Tensor"]],
    n_embd: int,
    n_experts_total: int,
) -> tuple[list[int], np.ndarray]:
    """Convert OBLITERATUS pipeline._expert_directions to (moe_layers, expert_dirs).

    OBLITERATUS produces directions only for experts that received enough
    harmful-prompt routing during probing. Other experts get zero (a no-op
    projection in the engine).

    Returns:
        moe_layers:    sorted list of MoE layer indices with at least one expert direction
        expert_dirs:   numpy float32 [n_moe_layers, n_experts_total, n_embd]
    """
    moe_layers = sorted(expert_directions.keys())
    if not moe_layers:
        return [], np.zeros((0, n_experts_total, n_embd), dtype=np.float32)

    out = np.zeros((len(moe_layers), n_experts_total, n_embd), dtype=np.float32)
    for j, L in enumerate(moe_layers):
        for ei, dir_t in expert_directions[L].items():
            if 0 <= ei < n_experts_total:
                out[j, ei, :] = dir_t.detach().cpu().float().numpy()
    return moe_layers, out


def _detect_n_experts(pipeline: Any) -> int:
    """Best-effort discovery of the routed-expert count for an MoE model.

    OBLITERATUS doesn't publish `n_experts` directly, but the pipeline holds a
    handle to the HF model whose config typically has `num_local_experts` or
    `num_experts`. Returns 0 for non-MoE models.
    """
    handle = getattr(pipeline, "handle", None)
    if handle is None:
        return 0
    cfg = getattr(handle.model, "config", None) if hasattr(handle, "model") else None
    if cfg is None:
        return 0
    for attr in ("num_local_experts", "num_experts", "n_routed_experts", "num_routed_experts"):
        n = getattr(cfg, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    return 0


def _emit_sidecar_gguf(
    output_path: Path,
    *,
    arch: str,
    n_embd: int,
    chosen_layers: list[int],
    dense_directions: np.ndarray,
    expert_layer_indices: list[int],
    expert_directions: np.ndarray | None,
    scale: float = 1.0,
    threshold: float = 0.0,
) -> None:
    """Write the sidecar GGUF in the schema the frankenturbo2 engine expects.

    The dense `abl.directions` ggml tensor has shape [n_embd, k]. ggml
    interprets a numpy array of shape (k, n_embd) as ggml [n_embd, k] (the
    last numpy dim is ggml dim 0). Same for per-expert: numpy shape
    (k_e, n_experts, n_embd) ↔ ggml [n_embd, n_experts, k_e].
    """
    import gguf  # type: ignore

    output_path.parent.mkdir(parents=True, exist_ok=True)

    w = gguf.GGUFWriter(str(output_path), "abl")
    w.add_string("abl.arch", arch)
    w.add_uint32("abl.n_embd", n_embd)
    w.add_array("abl.layer_indices", chosen_layers)
    w.add_float32("abl.gate_threshold", float(threshold))
    w.add_float32("abl.scale", float(scale))
    w.add_string("abl.directions_dtype", "f32")

    # numpy (k, n_embd) → ggml [n_embd, k]
    w.add_tensor("abl.directions", np.ascontiguousarray(dense_directions))

    if expert_directions is not None and expert_directions.size > 0:
        n_experts = expert_directions.shape[1]
        w.add_array("abl.expert_layer_indices", expert_layer_indices)
        w.add_uint32("abl.n_experts", n_experts)
        # numpy (k_e, n_experts, n_embd) → ggml [n_embd, n_experts, k_e]
        w.add_tensor("abl.expert_directions", np.ascontiguousarray(expert_directions))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def run_extraction(
    *,
    hf_model: str,
    method: str,
    output_path: Path,
    arch: str | None = None,
    obliteratus_output_dir: Path | None = None,
    per_expert: bool | None = None,
    device: str = "auto",
    dtype: str = "auto",
    quantization: str | None = None,
    n_directions: int | None = None,
    direction_method: str | None = None,
    refinement_passes: int | None = None,
    regularization: float | None = None,
    large_model_mode: bool = False,
    verify_sample_size: int | None = None,
    keep_modified_safetensors: bool = True,
    trust_remote_code: bool = False,
    skip_rebirth: bool = False,
    offload_folder: Path | None = None,
    on_log: Any = None,
) -> ExtractionResult:
    """Run OBLITERATUS end-to-end, then emit a sidecar GGUF from the
    captured per-layer (and optional per-expert) directions.

    `arch` overrides the auto-detected llama.cpp arch string. If None, we try
    HF_TO_GGUF_ARCH and abort if unknown.
    """
    from obliteratus.abliterate import AbliterationPipeline, METHODS  # type: ignore

    if method not in METHODS:
        raise ValueError(f"Unknown OBLITERATUS method '{method}'. "
                         f"Available: {sorted(METHODS.keys())}")

    if obliteratus_output_dir is None:
        # Default: stash modified safetensors under output_path's parent
        # so they share the ceph storage location.
        slug = hf_model.replace("/", "_")
        obliteratus_output_dir = output_path.parent / f"{slug}.obliteratus_out"
    obliteratus_output_dir = Path(obliteratus_output_dir)
    obliteratus_output_dir.mkdir(parents=True, exist_ok=True)

    if arch is None:
        from obliteratus_to_sidecar.arch_map import hf_to_gguf_arch
        from transformers import AutoConfig  # type: ignore

        cfg = AutoConfig.from_pretrained(hf_model, trust_remote_code=trust_remote_code)
        archs = getattr(cfg, "architectures", None) or []
        for hf_arch_name in archs:
            arch_guess = hf_to_gguf_arch(hf_arch_name)
            if arch_guess:
                arch = arch_guess
                logger.info("Auto-detected arch '%s' from HF '%s'", arch, hf_arch_name)
                break
        if arch is None:
            raise RuntimeError(
                f"Could not auto-detect llama.cpp arch for HF architectures {archs}; "
                f"pass --arch explicitly. Known mappings live in "
                f"obliteratus_to_sidecar/arch_map.py."
            )

    # Build constructor kwargs, only setting what the user explicitly asked
    # to override; otherwise let the method preset's defaults apply.
    kwargs: dict[str, Any] = dict(
        model_name=hf_model,
        output_dir=str(obliteratus_output_dir),
        method=method,
        device=device,
        dtype=dtype,
        large_model_mode=large_model_mode,
        trust_remote_code=trust_remote_code,
    )
    if quantization is not None:
        kwargs["quantization"] = quantization
    if offload_folder is not None:
        offload_folder = Path(offload_folder)
        offload_folder.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(offload_folder)
    if n_directions is not None:
        kwargs["n_directions"] = n_directions
    if direction_method is not None:
        kwargs["direction_method"] = direction_method
    if refinement_passes is not None:
        kwargs["refinement_passes"] = refinement_passes
    if regularization is not None:
        kwargs["regularization"] = regularization
    if per_expert is not None:
        kwargs["per_expert_directions"] = per_expert
    if verify_sample_size is not None:
        kwargs["verify_sample_size"] = verify_sample_size
    if on_log is not None:
        kwargs["on_log"] = on_log

    logger.info("Constructing AbliterationPipeline with method=%s, per_expert=%s, skip_rebirth=%s",
                method, per_expert, skip_rebirth)

    if skip_rebirth:
        # Subclass to skip the safetensors save. We only need pipeline state for
        # sidecar emission; saving 18-60GB of modified weights to disk is the
        # crashy step on memory-constrained hosts and is unnecessary for our flow.
        class _DirectionOnlyPipeline(AbliterationPipeline):  # type: ignore[misc]
            def _rebirth(self) -> "Path":
                self._emit("rebirth", "running",
                           "Skipping safetensors save (--skip-rebirth)")
                p = Path(self.output_dir) / ".skipped_rebirth"
                self._emit("rebirth", "done", "Skipped (no safetensors written)")
                return p
        pipeline = _DirectionOnlyPipeline(**kwargs)
    else:
        pipeline = AbliterationPipeline(**kwargs)

    logger.info("Running OBLITERATUS pipeline (this may take a long time)...")
    result_path = pipeline.run()
    if skip_rebirth:
        logger.info("OBLITERATUS pipeline run complete (rebirth skipped)")
    else:
        logger.info("OBLITERATUS complete; modified safetensors at %s", result_path)

    # ── Capture directions from pipeline state ────────────────────────────
    refusal_directions = getattr(pipeline, "refusal_directions", {})
    chosen_layers, dense = _stack_dense(refusal_directions)

    handle = pipeline.handle
    n_embd = int(handle.hidden_size)
    if dense.shape[1] != n_embd:
        raise RuntimeError(
            f"Direction width ({dense.shape[1]}) != model n_embd ({n_embd}). "
            f"OBLITERATUS may have used a different residual size; investigate."
        )

    expert_layer_indices: list[int] = []
    expert_directions = None
    n_experts_total = 0

    expert_dir_attr = getattr(pipeline, "_expert_directions", None)
    if expert_dir_attr:
        n_experts_total = _detect_n_experts(pipeline)
        if n_experts_total > 0:
            expert_layer_indices, expert_directions = _stack_expert(
                expert_dir_attr, n_embd, n_experts_total
            )
            logger.info(
                "Captured per-expert directions: %d MoE layers x %d experts (zeros for "
                "experts without enough harmful routing)",
                len(expert_layer_indices), n_experts_total,
            )

    # ── Emit "full" sidecar GGUF (all distilled layers) ──────────────────
    _emit_sidecar_gguf(
        output_path=output_path,
        arch=arch,
        n_embd=n_embd,
        chosen_layers=chosen_layers,
        dense_directions=dense,
        expert_layer_indices=expert_layer_indices,
        expert_directions=expert_directions,
        scale=1.0,
        threshold=0.0,
    )
    logger.info("Wrote full sidecar GGUF to %s", output_path)

    # ── Emit "selected" sidecar GGUF (OBLITERATUS-chosen layer subset) ──
    # OBLITERATUS's _distill() populates self._strong_layers with the layer
    # indices its method-specific layer-selection algorithm picked for the
    # weight-baked excise stage. Filtering our captured directions to just
    # these gives an apples-to-apples reference for engine A/B comparison
    # against OBLITERATUS-modified-as-GGUF baseline.
    sidecar_selected_path: Path | None = None
    n_selected = 0
    strong = list(getattr(pipeline, "_strong_layers", []) or [])
    if strong:
        # Intersect with what we actually captured (some _strong_layers entries
        # may have been skipped by the SVD-empty patch).
        strong_set = set(strong)
        selected_layers = [L for L in chosen_layers if L in strong_set]
        if selected_layers:
            # Build the selected dense sub-tensor, preserving row order to match
            # selected_layers.
            layer_to_row = {L: i for i, L in enumerate(chosen_layers)}
            row_idx = np.array([layer_to_row[L] for L in selected_layers], dtype=np.int64)
            dense_selected = dense[row_idx, :]
            n_selected = len(selected_layers)

            # Per-expert directions are MoE-only; if present, filter rows whose
            # layer index is in the selected set, otherwise pass through unchanged.
            expert_layer_indices_sel = expert_layer_indices
            expert_directions_sel = expert_directions
            if expert_directions is not None and expert_layer_indices:
                e_keep = [(j, L) for j, L in enumerate(expert_layer_indices) if L in strong_set]
                if e_keep:
                    e_rows = np.array([j for j, _ in e_keep], dtype=np.int64)
                    expert_layer_indices_sel = [L for _, L in e_keep]
                    expert_directions_sel = expert_directions[e_rows, :, :]
                else:
                    expert_layer_indices_sel = []
                    expert_directions_sel = None

            sel_name = output_path.name
            if sel_name.endswith(".abl.gguf"):
                sel_name = sel_name[: -len(".abl.gguf")] + ".selected.abl.gguf"
            else:
                sel_name = output_path.stem + ".selected" + output_path.suffix
            sidecar_selected_path = output_path.with_name(sel_name)

            _emit_sidecar_gguf(
                output_path=sidecar_selected_path,
                arch=arch,
                n_embd=n_embd,
                chosen_layers=selected_layers,
                dense_directions=dense_selected,
                expert_layer_indices=expert_layer_indices_sel,
                expert_directions=expert_directions_sel,
                scale=1.0,
                threshold=0.0,
            )
            logger.info(
                "Wrote selected sidecar GGUF (%d of %d layers) to %s",
                n_selected, len(chosen_layers), sidecar_selected_path,
            )
        else:
            logger.warning(
                "OBLITERATUS _strong_layers selection (%d entries) had no overlap "
                "with captured directions (%d layers); skipping selected sidecar.",
                len(strong), len(chosen_layers),
            )
    else:
        logger.info(
            "OBLITERATUS did not populate _strong_layers; skipping selected sidecar "
            "(method may not have run a layer-selection stage)."
        )

    # ── Optional cleanup of modified safetensors ─────────────────────────
    if not keep_modified_safetensors:
        logger.info("Removing OBLITERATUS modified-weights output at %s",
                    obliteratus_output_dir)
        shutil.rmtree(obliteratus_output_dir, ignore_errors=True)

    # Save a small metadata sidecar JSON next to the sidecar GGUF for traceability.
    meta = {
        "hf_model": hf_model,
        "method": method,
        "method_label": METHODS[method].get("label", method),
        "per_expert": bool(per_expert),
        "arch": arch,
        "n_embd": n_embd,
        "n_dense_layers": len(chosen_layers),
        "n_dense_layers_selected": n_selected,
        "captured_layers": chosen_layers,
        "selected_layers": list(strong),
        "n_expert_layers": len(expert_layer_indices),
        "n_experts_per_layer": int(n_experts_total),
        "obliteratus_output_dir": str(obliteratus_output_dir),
        "obliteratus_result_path": str(result_path),
        "sidecar_path": str(output_path),
        "sidecar_selected_path": (str(sidecar_selected_path)
                                   if sidecar_selected_path is not None else None),
        "kept_modified_safetensors": keep_modified_safetensors,
    }
    output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    return ExtractionResult(
        sidecar_path=output_path,
        sidecar_selected_path=sidecar_selected_path,
        obliteratus_output_dir=obliteratus_output_dir,
        n_dense_layers=len(chosen_layers),
        n_dense_layers_selected=n_selected,
        n_expert_layers=len(expert_layer_indices),
        n_experts_per_layer=int(n_experts_total),
        arch=arch,
        n_embd=n_embd,
        method=method,
    )
