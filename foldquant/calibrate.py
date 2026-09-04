# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Calibration capture: recover a module's real inputs from a replay callable.

Every FoldQuant export takes a ``forward_loop(module)`` — a callable that runs
the host model over calibration observations so *module* is invoked the way the
deployment invokes it. The fold needs those inputs (SmoothQuant scales are the
per-channel amax of the rotated activation; GPTQ Hessians are of the same
tensor), so the helpers here run the loop under forward-pre-hooks and hand back
the captured calls in the shape each capture function wants.

Also here: structural detection of the LLM architecture (from the module's own
config, never a model-family name), the Qwen3-VL graph parameters that are only
knowable from a captured call, and the loader for learned SmoothQuant scales.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

#: The DiT inputs :func:`foldquant.dit_int4.compute_dit_sq_scales` unpacks, in its unpack order.
DIT_SQ_INPUT_NAMES: Tuple[str, ...] = (
    "hidden_states",
    "encoder_hidden_states",
    "timestep",
    "image_mask",
    "backbone_attention_mask",
)


def capture_dit_inputs(
    module: nn.Module, forward_loop: Any, *, input_names: Tuple[str, ...] = DIT_SQ_INPUT_NAMES
) -> list:
    """Replay *forward_loop* and return the DiT's inputs as ``input_names``-ordered tuples.

    The hook takes kwargs and binds against the signature: GR00T's DiT is invoked
    entirely by keyword, so a positional-only capture would record an empty tuple
    per call. Binding normalizes either calling convention into one fixed order.
    """
    captured: list = []
    signature = inspect.signature(module.forward)

    def _hook(_m: nn.Module, args: tuple, kwargs: dict) -> None:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        captured.append(tuple(bound.arguments[name] for name in input_names))

    handle = module.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        forward_loop(module)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            "calibration captured no DiT calls — the replay never reached the module, so "
            "SmoothQuant scales cannot be measured."
        )
    absent = [name for name, value in zip(input_names, captured[0]) if value is None]
    if absent:
        raise RuntimeError(
            f"calibration captured DiT calls missing {absent}; every one of {list(input_names)} "
            "must be present in the replayed call."
        )
    return captured


def capture_llm_snapshots(decoder: nn.Module, forward_loop: Any) -> list:
    """Replay *forward_loop* and return each raw ``(args, kwargs)`` call to *decoder*.

    Hook the DECODER (the module exposing ``.layers``), not a CausalLM-style
    wrapper around it: an export wrapper that calls ``language_model.model``
    directly never fires a hook on ``language_model``. Callers pass
    :func:`foldquant.llm.resolve_qwen3_decoder` output.
    """
    captured: list = []

    def _hook(_m: nn.Module, args: tuple, kwargs: dict) -> None:
        captured.append((args, dict(kwargs)))

    handle = decoder.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        forward_loop(decoder)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            "calibration captured no LLM calls — the replay never reached the decoder, so "
            f"SmoothQuant scales cannot be measured. (Hook target: {type(decoder).__name__}.)"
        )
    return captured


def captured_prefix_len(snapshots: list) -> int:
    """Sequence length of the first captured call's hidden states ([B, S, K] → S)."""
    for args, kwargs in snapshots:
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, torch.Tensor) and value.dim() == 3 and value.is_floating_point():
                return int(value.shape[1])
    raise ValueError("No captured decoder call carries a 3-D hidden-states tensor to pin the prefix length from.")


# --- LLM architecture, decided from the module's config ----------------------


def _model_type(module: nn.Module) -> str:
    cfg = getattr(module, "config", None)
    return str(getattr(cfg, "model_type", "")).lower()


def rope_scaling(module: nn.Module) -> Dict[str, Any]:
    """The module's ``config.rope_scaling`` mapping, or an empty dict."""
    config: Any = getattr(module, "config", None)
    scaling: Any = getattr(config, "rope_scaling", None)
    return dict(scaling) if isinstance(scaling, dict) else {}


def is_gemma(module: nn.Module) -> bool:
    """Pi0 / Pi0.5's PaliGemma decoder reports ``model_type == "gemma"``."""
    return _model_type(module) == "gemma"


def is_llama(module: nn.Module) -> bool:
    """SmolVLA's SmolLM2 text stack reports ``model_type == "llama"``."""
    return _model_type(module) == "llama"


def is_qwen2(module: nn.Module) -> bool:
    """Evo-1's InternVL3 LLM. Implies q/k/v biases (read structurally) and the flash-parity padded-query mask."""
    return _model_type(module) == "qwen2"


def is_qwen3_vl(module: nn.Module) -> bool:
    """GR00T N1.7's Qwen3-VL text tower — decided by ``rope_scaling.mrope_section``, not by name."""
    return bool(rope_scaling(module).get("mrope_section"))


def qwen3_vl_graph_params(module: nn.Module, snapshots: list) -> Optional[Dict[str, Any]]:
    """Emitter kwargs for a Qwen3-VL LLM, or ``None`` for a plain Qwen decoder.

    Qwen3 (N1.5 / N1.6) bakes a 1-D RoPE table; Qwen3-VL (N1.7) computes
    interleaved M-RoPE in-graph from ``position_ids`` and injects deepstack
    residuals after its first layers. Every value comes off the captured
    calibration call the fold itself uses, so graph and scales describe one pass.
    """
    scaling = rope_scaling(module)
    mrope_section = scaling.get("mrope_section")
    if not mrope_section:
        return None
    deepstack: list = []
    batch = 1
    for args, kwargs in snapshots:
        deepstack = list(kwargs.get("deepstack_visual_embeds") or []) or deepstack
        embeds = args[0] if args else kwargs.get("inputs_embeds")
        if embeds is not None and getattr(embeds, "ndim", 0) == 3:
            batch = int(embeds.shape[0])
    if not deepstack:
        raise ValueError(
            "This LLM declares rope_scaling.mrope_section (Qwen3-VL) but the calibration replay "
            "captured no 'deepstack_visual_embeds'; the plugin graph injects those residuals as "
            "explicit inputs and cannot be emitted without knowing how many there are."
        )
    return {
        "n1d7_mode": True,
        "mrope_section": list(mrope_section),
        "attention_scaling": float(scaling.get("attention_factor", 1.0)),
        "num_deepstack": len(deepstack),
        "num_vis_tokens": int(deepstack[0].shape[0]),
        "batch": batch,
    }


def load_learned_calib(path: Optional[str], *, num_layers: int, device: Any) -> Optional[Dict[str, Any]]:
    """Load learned SmoothQuant scales / clips and check they match the decoder.

    Returns ``{"sq_scales", "act_clip", "weight_clip"}`` with tensors on *device*,
    or ``None`` when *path* is empty. The layer count must match: a file learned
    on a different LLM would fold silently into the wrong sites.
    """
    if not path:
        return None
    data = torch.load(path, map_location="cpu")
    sq = {k: torch.as_tensor(v).float().to(device) for k, v in data["sq_scales"].items()}
    layers = {int(k.split("_")[0][1:]) for k in sq}
    if layers != set(range(num_layers)):
        raise ValueError(f"learned_calib {path}: scales cover layers {sorted(layers)}, decoder has {num_layers}")
    act_clip = {k: float(v) for k, v in (data.get("act_clip") or {}).items()}
    wc = data.get("weight_clip") or None
    weight_clip = {k: torch.as_tensor(v).float() for k, v in wc.items()} if wc else None
    return {"sq_scales": sq, "act_clip": act_clip, "weight_clip": weight_clip}


__all__: List[str] = [
    "DIT_SQ_INPUT_NAMES",
    "capture_dit_inputs",
    "capture_llm_snapshots",
    "captured_prefix_len",
    "is_gemma",
    "is_llama",
    "is_qwen2",
    "is_qwen3_vl",
    "load_learned_calib",
    "qwen3_vl_graph_params",
    "rope_scaling",
]
