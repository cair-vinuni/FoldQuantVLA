# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Export a live module as a FoldQuant plugin-node ONNX graph.

One entry point per module kind — :func:`export_llm`, :func:`export_dit`,
:func:`export_expert`, :func:`export_action_head` — plus :func:`export_module`,
which dispatches on the module name. Each takes the live PyTorch module, a
destination path, a scheme key from :mod:`.schemes` and, for every folded
scheme, a ``forward_loop(module)`` that replays calibration observations
through the host model. Nothing is traced: the graph is emitted node by node
from the module's own weights, and the fold is applied to those weights on the
way out.

Calibration order is fixed and shared by every module kind:

1. SmoothQuant scales — per-channel amax of the activation in the frame the
   fold lands on (raw channel for fold-before, rotated for fold-after).
2. GPTQ Hessians (``_g`` schemes only) — a SECOND replay after the scales
   exist, of the transformed activation ``rot(x / s)``: GPTQ compensates the
   rounding error of the weight the engine stores, so its Hessian must be of
   the activation that weight actually multiplies.
3. Emit: :func:`foldquant.foldq.fold_site` folds scale, rotation and rounding
   into each site's weights, in one place, for both bit widths.

``params`` carries the per-module knobs a scheme exposes: ``sq_alpha``,
``sq_fold_order`` (action modules), and the LLM overrides validated by
:func:`foldquant.llm_rotation_sq.resolve_llm_plugin_params` (``sq_alpha``,
``act_clip_ratio``, ``site_bits``, ``rot_block_size``, ``learned_calib``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch.nn as nn

from . import calibrate, schemes
from .llm_gptq import gptq_prepare

logger = logging.getLogger(__name__)

#: Upper bound the LLM plugin graph bakes its RoPE / causal-mask tables at. The
#: engine handles any seq_len in ``[1, bound]`` by runtime slicing. Deliberately
#: a fixed constant rather than ``config.max_position_embeddings`` (32768 for
#: Qwen3): baking at that size allocates a multi-gigabyte mask initializer for a
#: bound a VLA prefix never approaches.
LLM_BAKE_MAX_SEQ_LEN = 4096

ForwardLoop = Callable[[nn.Module], None]


@dataclass
class ExportResult:
    """What an export produced and what running it needs."""

    module: str
    scheme: str
    onnx_path: Path
    #: Plugin libraries the engine build and the runtime must load.
    plugin_libs: List[str]
    #: Folded-LLM only: everything :func:`install_llm_emulation` needs to
    #: replay the deployed LLM numerics in PyTorch (cascade calibration of the
    #: modules downstream of the LLM). ``None`` otherwise.
    emulation: Optional[Dict[str, Any]] = field(default=None, repr=False)


def _prepare_gptq(hessians: Dict[str, Any]) -> Dict[str, Any]:
    """Factorize each site's Hessian once, for :func:`foldquant.foldq.fold_site`."""
    return {k: gptq_prepare(h) for k, h in hessians.items()}


def _require_loop(scheme: str, forward_loop: Optional[ForwardLoop]) -> ForwardLoop:
    if forward_loop is None:
        raise ValueError(
            f"{scheme!r} needs calibration data: its SmoothQuant scales are the per-channel amax of "
            "the (rotated) activation measured from real inputs, and GPTQ schemes additionally "
            "measure per-site Hessians there. Pass forward_loop."
        )
    return forward_loop


def _act_fold_knobs(scheme: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    """``{"fwht", "fold_order", "alpha"?}`` for an action-module fold.

    Butterfly arms default to fold-BEFORE (SmoothRot: scale the raw channel, then
    rotate). It is the cheaper side of the kernel — the divide fuses into the
    load loop where fold-after needs a second shared-memory pass — and on the
    Gemma expert the two orders differ by 0.0006 of action cosine, far below what
    800 LIBERO episodes resolve. "after" stays selectable as the ablation arm.

    ``alpha`` only has meaning in the raw frame: 1.0 normalizes the activation
    completely and hands every outlier to per-ROW weight quantization, where one
    cold channel then sets the row scale for all of them; 0.5 splits the burden.
    """
    fwht = schemes.uses_fwht(scheme)
    fold_order = str(params.get("sq_fold_order", "before" if fwht else "after"))
    if fold_order not in ("before", "after"):
        raise ValueError(f"sq_fold_order must be 'before' or 'after', got {fold_order!r}")
    knobs: Dict[str, Any] = {"fwht": fwht, "fold_order": fold_order}
    if fold_order == "before":
        knobs["alpha"] = float(params.get("sq_alpha", 0.5))
    return knobs


# --------------------------------------------------------------------------- LLM


def export_llm(
    module: nn.Module,
    onnx_path: Path,
    *,
    scheme: str = schemes.W8A8,
    forward_loop: Optional[ForwardLoop] = None,
    params: Optional[Mapping[str, Any]] = None,
    max_seq_len: int = LLM_BAKE_MAX_SEQ_LEN,
    final_norm: Optional[bool] = None,
) -> ExportResult:
    """Emit the LLM decoder as a plugin-node graph.

    The architecture is read off ``module.config``: Qwen2 / Qwen3 / Qwen3-VL
    (GR00T, Evo-1) get the dynamic-S decoder graph; SmolLM2 (``smolvla_mode``)
    and Gemma (``gemma_mode``) get the KV-stack prefix graph the expert
    cross-attends. A Qwen3-VL LLM needs a ``forward_loop`` even for ``w8a8``:
    its deepstack count and visual-token width come from a captured call.
    ``final_norm`` states whether the graph ends with the tower's final RMSNorm
    (``None`` reads it off the module; see :func:`foldquant.llm.build_llm_plugin_onnx`).
    """
    from .llm import build_llm_plugin_onnx, resolve_qwen3_decoder
    from .llm_rotation_sq import compute_sq_scales_llm, resolve_llm_plugin_params

    schemes.validate("llm", scheme)
    params = dict(params or {})
    onnx_path = Path(onnx_path)
    folded = scheme in schemes.LLM_FOLDED_SCHEMES
    loop = _require_loop(scheme, forward_loop) if folded else forward_loop
    decoder = resolve_qwen3_decoder(module)
    prefix_graph = calibrate.is_llama(module) or calibrate.is_gemma(module)

    mode_kwargs: Dict[str, Any] = {}
    snapshots: list = []
    if prefix_graph:
        mode_kwargs = {"gemma_mode": True} if calibrate.is_gemma(module) else {"smolvla_mode": True}
        if calibrate.is_gemma(module):
            # The Pi processor pads text to a fixed length and the camera count is
            # fixed, so the runtime prefix is constant; pin it from a captured
            # call. Hook layer 0's q_proj LEAF: the Pi prefix replay drives the
            # projection submodules directly and never runs the decoder's forward.
            if loop is None:
                raise ValueError(
                    "The Gemma plugin graph pins its prefix length from a captured forward call — "
                    "pass forward_loop even for the dynamic per-row scheme."
                )
            q0 = decoder.layers[0].self_attn.q_proj
            mode_kwargs["pin_seq_len"] = calibrate.captured_prefix_len(calibrate.capture_llm_snapshots(q0, loop))
    else:
        mode_kwargs["padded_query_mask"] = calibrate.is_qwen2(module)
        mode_kwargs["final_norm"] = final_norm
        if folded or calibrate.is_qwen3_vl(module):
            if loop is None:
                raise ValueError(
                    "This LLM is Qwen3-VL (rope_scaling.mrope_section present). Its plugin graph injects "
                    "M-RoPE position_ids and deepstack residuals as explicit inputs whose count and width "
                    "are only knowable from a real forward call — pass forward_loop even for w8a8."
                )
            snapshots = calibrate.capture_llm_snapshots(decoder, loop)
            mode_kwargs.update(calibrate.qwen3_vl_graph_params(module, snapshots) or {})

    if not folded:
        build_llm_plugin_onnx(module, onnx_path, max_seq_len=max_seq_len, **mode_kwargs)
        logger.info("Emitted %s LLM graph at %s", scheme, onnx_path)
        return ExportResult("llm", scheme, onnx_path, schemes.plugin_libs(scheme, params=params))

    bits = schemes.bits_of(scheme)
    fold = resolve_llm_plugin_params(scheme, bits=bits, overrides=params or None)
    if prefix_graph:
        # The prefix replay drives the layer submodules directly, so the per-site
        # leaf hooks fire under it; there is no decoder-level snapshot to replay.
        # One snapshot therefore stands for the WHOLE forward loop — every
        # calibration observation runs inside it, which is what the "over 1
        # replay snapshot(s)" log lines below count.
        replay_snaps: list = [None]

        def replay(_snap: Any) -> None:
            loop(module)
    else:
        replay_snaps = snapshots

        def replay(snap: Any) -> None:
            decoder(*snap[0], **snap[1])

    learned = calibrate.load_learned_calib(
        fold.get("learned_calib"), num_layers=len(decoder.layers), device=next(decoder.parameters()).device
    )
    if learned is not None:
        sq_scales = learned["sq_scales"]
    else:
        sq_scales = compute_sq_scales_llm(decoder, replay_snaps, alpha=fold["sq_alpha"], forward_fn=replay)
    gptq_hessians = None
    if bits == 4:
        from .llm_gptq import compute_gptq_hessians_llm

        gptq_hessians = compute_gptq_hessians_llm(
            decoder, replay_snaps, sq_scales=sq_scales, rot_bs=int(fold["rot_block_size"]), forward_fn=replay
        )
    act_clip: Any = learned["act_clip"] if learned is not None else float(fold.get("act_clip_ratio", 1.0))
    weight_clip = learned["weight_clip"] if learned is not None else None
    fold_kwargs: Dict[str, Any] = {
        "sq_scales": sq_scales,
        "rot_bs": int(fold["rot_block_size"]),
        "bits": bits,
        "gptq_hessians": gptq_hessians,
        "act_clip_ratio": act_clip,
        "site_bits": fold.get("site_bits"),
        "act_bits": fold.get("act_bits"),
        "weight_clip": weight_clip,
    }
    build_llm_plugin_onnx(module, onnx_path, max_seq_len=max_seq_len, **fold_kwargs, **mode_kwargs)
    if calibrate.is_gemma(module):
        family = "gemma"
    elif calibrate.is_llama(module):
        family = "smollm_llama"
    else:
        family = "qwen"
    logger.info("Emitted %s LLM graph (%s, %d-bit) at %s", scheme, family, bits, onnx_path)
    return ExportResult(
        "llm",
        scheme,
        onnx_path,
        schemes.plugin_libs(scheme, params=params),
        emulation={"family": family, **fold_kwargs},
    )


def install_llm_emulation(module: nn.Module, result: ExportResult) -> Any:
    """Install a kernel-matched fake-quant of an exported LLM, for cascade calibration.

    Modules downstream of the LLM (the DiT / expert) should be calibrated on the
    activations the *quantized* LLM produces, not the float one. Returns an
    ``LlmEmulationHandle``; call ``.remove()`` after the downstream capture.

    Raises:
        ValueError: *result* carries no emulation data (not a folded LLM scheme).
        NotImplementedError: SmolLM2 — its convention is not covered by the
            Qwen / Gemma emulation, and a wrong emulation is worse than none.
    """
    data = result.emulation
    if data is None:
        raise ValueError(f"{result.scheme!r} on {result.module!r} has no LLM emulation to install.")
    if data["family"] not in ("qwen", "gemma", "smollm_llama"):
        raise NotImplementedError(
            f"cascade calibration: the PyTorch emulation covers Qwen2/Qwen3/Qwen3-VL, Gemma and "
            f"SmolLM2/Llama; {data['family']!r} is not validated there."
        )
    if data["family"] == "smollm_llama":
        # Llama layout, plain RMSNorm (y = rms(x)*w): the Qwen path applies unchanged. Validated
        # against the W4A4 kv-stack engine on SmolVLA (see models/smolvla README, cascade).
        logger.info("cascade emulation on a SmolLM2/Llama decoder: plain-RMSNorm (Qwen) fold path")
    from .llm_fake_quant import install_llm_per_row_emulation

    return install_llm_per_row_emulation(
        module,
        sq_scales=data["sq_scales"],
        gptq_hessians=data["gptq_hessians"],
        rot_bs=data["rot_bs"],
        bits=data["bits"],
        act_clip_ratio=data.get("act_clip_ratio", 1.0),
        site_bits=data.get("site_bits"),
        act_bits=data.get("act_bits"),
        weight_clip=data.get("weight_clip"),
    )


# --------------------------------------------------------------------------- DiT


def export_dit(
    module: nn.Module,
    onnx_path: Path,
    *,
    scheme: str = schemes.W8A8,
    forward_loop: Optional[ForwardLoop] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> ExportResult:
    """Emit the GR00T action-head DiT as a plugin-node graph."""
    schemes.validate("dit", scheme)
    params = dict(params or {})
    onnx_path = Path(onnx_path)
    libs = schemes.plugin_libs(scheme, params=params)

    if scheme == schemes.W8A8:
        from .dit_int8 import build_dit_plugin_onnx

        build_dit_plugin_onnx(module, onnx_path)
        return ExportResult("dit", scheme, onnx_path, libs)

    from .dit_int4 import compute_dit_sq_scales

    loop = _require_loop(scheme, forward_loop)
    knobs = _act_fold_knobs(scheme, params)
    if knobs["fwht"] and knobs["fold_order"] != "before":
        raise ValueError(
            f"{scheme} on the DiT carries only the pre-rotation scale (act_scale_pre_*); "
            "sq_fold_order='after' would need a post-rotation vector the macro plugins do not declare."
        )
    # The scales are measured in-process from the same replay, so the fold
    # matches the weights (and rotation) this very graph bakes.
    calib_inputs = calibrate.capture_dit_inputs(module, loop)
    alpha = knobs.get("alpha", 1.0)
    sq = compute_dit_sq_scales(module, calib_inputs, alpha=alpha, fwht=knobs["fwht"], fold_order=knobs["fold_order"])

    if scheme == schemes.W8A8_SH:
        from .dit_int8 import build_dit_plugin_onnx

        build_dit_plugin_onnx(module, onnx_path, sq_scales=sq, sq_fold_order=knobs["fold_order"], fwht=True)
        return ExportResult("dit", scheme, onnx_path, libs)

    from .dit_int4 import build_dit_plugin_onnx_int4

    gptq = None
    if scheme == schemes.W4A4_SHG:
        # Second replay, after the scales exist: Hessians of rot(x / s).
        hess = compute_dit_sq_scales(
            module, calib_inputs, alpha=alpha, fwht=knobs["fwht"], fold_order=knobs["fold_order"], gptq_scales=sq
        )
        gptq = _prepare_gptq(hess)
    build_dit_plugin_onnx_int4(
        module, onnx_path, sq_scales=sq, sq_fold_order=knobs["fold_order"], fwht=knobs["fwht"], gptq=gptq
    )
    return ExportResult("dit", scheme, onnx_path, libs)


# ------------------------------------------------------------------------ expert


def export_expert(
    module: nn.Module,
    onnx_path: Path,
    *,
    scheme: str = schemes.W8A8,
    forward_loop: Optional[ForwardLoop] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> ExportResult:
    """Emit the SmolVLA (SmolLM2 dual-stream) or Pi (Gemma-300M) action expert.

    Which emitter runs is decided structurally: a Pi expert exposes
    ``expert_model``. Both emitters take the SAME kwargs, from one mapping —
    a knob dropped on one side leaves the capture measuring in one frame while
    the emitter folds in another, with no error anywhere.
    """
    schemes.validate("expert", scheme)
    params = dict(params or {})
    onnx_path = Path(onnx_path)
    is_gemma_expert = hasattr(module, "expert_model")
    if is_gemma_expert:
        from .gemma_expert import build_gemma_expert_plugin_onnx as build
        from .gemma_expert import compute_gemma_expert_sq_scales as compute
    else:
        from .smolvla_expert import build_smolvla_expert_plugin_onnx as build
        from .smolvla_expert import compute_smolvla_expert_sq_scales as compute

    kwargs: Dict[str, Any] = {}
    if scheme != schemes.W8A8:
        loop = _require_loop(scheme, forward_loop)
        knobs = _act_fold_knobs(scheme, params)
        scale_kwargs: Dict[str, Any] = {"fold_order": knobs["fold_order"]}
        if knobs["fwht"]:
            scale_kwargs["fwht"] = True
        if "alpha" in knobs:
            scale_kwargs["alpha"] = knobs["alpha"]
        sq = compute(module, loop, **scale_kwargs)
        kwargs = {"int4": scheme in schemes.ACT_W4A4_SCHEMES, "sq_scales": sq, "fold_order": knobs["fold_order"]}
        if scheme == schemes.W4A4_SHG:
            kwargs["gptq"] = _prepare_gptq(compute(module, loop, **scale_kwargs, gptq_scales=sq))
        if knobs["fwht"]:
            kwargs["fwht"] = True
    build(module, onnx_path, **kwargs)
    return ExportResult("expert", scheme, onnx_path, schemes.plugin_libs(scheme, params=params))


# ------------------------------------------------------------------- action head


def export_action_head(
    module: nn.Module,
    onnx_path: Path,
    *,
    scheme: str = schemes.W8A8,
    forward_loop: Optional[ForwardLoop] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> ExportResult:
    """Emit Evo-1's action head (one denoising step) as a plugin-node graph."""
    schemes.validate("action_head", scheme)
    params = dict(params or {})
    onnx_path = Path(onnx_path)
    libs = schemes.plugin_libs(scheme, params=params)

    if scheme == schemes.W8A8:
        from .evo1_head_int8 import build_evo1_head_plugin_onnx

        build_evo1_head_plugin_onnx(module, onnx_path)
        return ExportResult("action_head", scheme, onnx_path, libs)

    from .evo1_head_int4 import compute_evo1_head_sq_scales

    loop = _require_loop(scheme, forward_loop)
    knobs = _act_fold_knobs(scheme, params)
    sq = compute_evo1_head_sq_scales(module, loop, **knobs)

    if scheme == schemes.W8A8_SH:
        from .evo1_head_int8 import build_evo1_head_plugin_onnx

        build_evo1_head_plugin_onnx(module, onnx_path, sq_scales=sq, fold_order=knobs["fold_order"], fwht=True)
        return ExportResult("action_head", scheme, onnx_path, libs)

    from .evo1_head_int4 import build_evo1_head_plugin_onnx_int4

    gptq = None
    if scheme == schemes.W4A4_SHG:
        from .evo1_head_int4 import compute_evo1_head_gptq_hessians

        hess = compute_evo1_head_gptq_hessians(
            module, loop, sq, block_size=64, fold_order=knobs["fold_order"], fwht=knobs["fwht"]
        )
        gptq = _prepare_gptq(hess)
    build_evo1_head_plugin_onnx_int4(
        module, onnx_path, sq_scales=sq, fwht=knobs["fwht"], fold_order=knobs["fold_order"], gptq=gptq
    )
    return ExportResult("action_head", scheme, onnx_path, libs)


# ---------------------------------------------------------------------- dispatch

_EXPORTERS: Dict[str, Callable[..., ExportResult]] = {
    "llm": export_llm,
    "dit": export_dit,
    "expert": export_expert,
    "action_head": export_action_head,
}


def export_module(
    name: str,
    module: nn.Module,
    onnx_path: Path,
    *,
    scheme: str,
    forward_loop: Optional[ForwardLoop] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> ExportResult:
    """Export *module* under *scheme*, dispatching on the module kind *name*."""
    schemes.validate(name, scheme)
    return _EXPORTERS[name](module, onnx_path, scheme=scheme, forward_loop=forward_loop, params=params)


__all__ = [
    "LLM_BAKE_MAX_SEQ_LEN",
    "ExportResult",
    "export_action_head",
    "export_dit",
    "export_expert",
    "export_llm",
    "export_module",
    "install_llm_emulation",
]
