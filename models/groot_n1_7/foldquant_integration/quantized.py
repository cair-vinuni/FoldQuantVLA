# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""GR00T N1.7 adapter for the FoldQuant fake-quant pipeline (:mod:`foldquant.quantized`).

The pipeline itself lives in ``foldquant``: fake-quant state -> real-quant
plugin ONNX -> TensorRT engines. This module tells it what is GR00T N1.7
specific: how to load the policy, where its quantized modules sit, and how the
engine directory is assembled (the FoldQuant graphs plus upstream's float
components). Run the pipeline from ``models/groot_n1_7``::

    python -m foldquant_integration.quantize ... --output-dir exports/w4a4/quantized
    python -m foldquant.quantized push --quantized-model exports/w4a4/quantized --repo-id <org>/<name>
    python -m foldquant_integration.export --model-path <dir> --output-dir exports/w4a4/onnx
    python -m foldquant_integration.build_engines --onnx-dir exports/w4a4/onnx --engine-dir exports/w4a4/engines

``export`` writes the seven graphs of the ``n17_full_pipeline``: the two
FoldQuant graphs from the recorded codes, and the five float components (ViT,
VL self-attention, state and action encoders, action decoder) with upstream's
own exporters (:func:`complete_onnx`), so no separate float export is needed.
``python -m foldquant_integration.quantized`` is the same command line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from foldquant import quantized as _pipeline

from ._upstream import EXPORT_METADATA_NAME

FAMILY = "groot_n1_7"


def _tag_value(embodiment_tag: Optional[str]) -> Optional[str]:
    """``EmbodimentTag.LIBERO_PANDA`` (an enum's ``str``) -> its value; anything else unchanged."""
    if embodiment_tag and embodiment_tag.startswith("EmbodimentTag."):
        from gr00t.data.embodiment_tags import EmbodimentTag

        return EmbodimentTag[embodiment_tag.split(".", 1)[1]].value
    return embodiment_tag


def load_policy(model_path: str, embodiment_tag: Optional[str], device: Any = "cuda") -> Any:
    from . import calibration

    return calibration.load_policy(model_path, _tag_value(embodiment_tag), device)


def module_paths(policy: Any) -> Dict[str, Any]:
    """The quantizable modules, at the paths ``quantize`` reads them from."""
    return {
        "llm": policy.model.backbone.model.model.language_model,
        "dit": policy.model.action_head.model,
    }


def build_engines(
    onnx_dir: Path,
    engine_dir: Path,
    *,
    max_batch: int = 1,
    workspace_mb: int = 8192,
) -> Dict[str, str]:
    """A complete ``n17_full_pipeline`` engine directory from the seven graphs in *onnx_dir*."""
    from .build_engines import BuildConfig, build

    return build(BuildConfig(onnx_dir=str(onnx_dir), engine_dir=str(engine_dir), max_batch=max_batch, workspace_mb=workspace_mb))


#: The shape keys ``quantize`` records for the float components' exporters.
FLOAT_SHAPE_KEYS = ("vit_pixel_values_shape", "vit_grid_thw", "vl_seq_len")


def export_float_components(policy: Any, onnx_dir: Union[str, Path], shapes: Dict[str, Any]) -> None:
    """Emit the five components FoldQuant never replaces, with upstream's ``export_onnx_n1d7``:
    ViT (FP32, as upstream exports it), VL self-attention, the state and action encoders and
    the action decoder, traced from *shapes* (the ViT's patch tensor shape and image grid, the
    VL sequence length, recorded by ``quantize``). They land in *onnx_dir* next to the two
    tower graphs, so ``build_engines --onnx-dir`` finds the whole set in one directory."""
    from types import SimpleNamespace

    import torch

    from ._upstream import ensure_deployment_on_path

    ensure_deployment_on_path()
    import export_onnx_n1d7 as upstream

    out = str(onnx_dir)
    bs = int(shapes.get("batch_size", 1))
    captured_vit = SimpleNamespace(
        pixel_values_shape=tuple(int(d) for d in shapes["vit_pixel_values_shape"]),
        grid_thw=torch.tensor(shapes["vit_grid_thw"], dtype=torch.long),
    )
    upstream.export_vit_to_onnx(policy, out, captured_vit, use_bf16=False, batch_size=bs)
    upstream.export_vl_self_attention_to_onnx(policy, out, vl_seq_len=int(shapes["vl_seq_len"]), use_bf16=True, batch_size=bs)
    upstream.export_state_encoder_to_onnx(policy, out, use_bf16=True, batch_size=bs)
    upstream.export_action_encoder_to_onnx(policy, out, use_bf16=True, batch_size=bs)
    upstream.export_action_decoder_to_onnx(policy, out, use_bf16=True, batch_size=bs)


# A ModelOpt tower's graph: upstream's own export contract, traced from the call ``quantize``
# recorded in the quantized checkpoint (the ``trace`` tensor group).

#: What upstream's ``LLMInputCapture`` / ``DiTInputCapture`` hold, under the trace's names.
_LLM_TRACE = ("inputs_embeds", "position_ids", "attention_mask", "visual_pos_masks")
_DIT_TRACE = {"sa_embs": "hidden_states", "vl_embs": "encoder_hidden_states", "timestep": "timestep",
              "image_mask": "image_mask", "backbone_attention_mask": "backbone_attention_mask"}


def tower_trace(tower: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The tensors of one call into *tower* (its forward kwargs) under the trace's names:
    the LLM's deepstack list as ``deepstack_<i>``, the DiT's kwargs as upstream binds them."""
    if tower == "llm":
        trace = {k: kwargs[k] for k in _LLM_TRACE if kwargs.get(k) is not None}
        for i, d in enumerate(kwargs.get("deepstack_visual_embeds") or []):
            trace[f"deepstack_{i}"] = d
        return trace
    if tower == "dit":
        return {name: kwargs[k] for name, k in _DIT_TRACE.items() if kwargs.get(k) is not None}
    raise ValueError(f"no tower {tower!r}")


def export_modelopt_graph(tower: str, policy: Any, module: Any, onnx_path: Path, ms: Any) -> Any:
    """The ModelOpt Q/DQ graph of *tower*, with the quantizers restored on the live module,
    traced from the recorded call. Returns ``(ExportResult, export record)``."""
    from foldquant import modelopt_int8
    from foldquant.export import ExportResult
    from foldquant.trace_export import trace_kwargs

    from . import modelopt_export

    onnx_path = Path(onnx_path)
    trace = trace_kwargs(ms, next(module.parameters()).device)
    opset = int(ms.config.get("opset", modelopt_int8.DEFAULT_OPSET))
    if tower == "llm":
        record = modelopt_export.export_llm(module, trace, onnx_path, num_layers=int(ms.config["num_layers"]),
                                            algorithm=ms.scheme, opset=opset)
    else:
        record = modelopt_export.export_dit(module, trace, onnx_path, algorithm=ms.scheme, opset=opset)
    libs = []
    if modelopt_int8.is_weight_only(ms.scheme):
        from foldquant.kernels.locator import INT4_GROUPWISE_LIB

        libs = [INT4_GROUPWISE_LIB]
    return ExportResult(tower, ms.scheme, onnx_path, libs), record


def complete_onnx(policy: Any, onnx_dir: Union[str, Path], state: Any) -> None:
    """Emit the five float components beside the FoldQuant graphs ``export`` just wrote,
    from the shapes ``quantize`` recorded in ``export_metadata.json``."""
    import json

    onnx_dir = Path(onnx_dir)
    meta = json.loads((onnx_dir / EXPORT_METADATA_NAME).read_text())
    missing = [k for k in FLOAT_SHAPE_KEYS if k not in meta]
    if missing:
        raise ValueError(
            f"{EXPORT_METADATA_NAME} lacks {missing}: this checkpoint was quantized before the float "
            "components were exported from it. Quantize it again."
        )
    export_float_components(policy, onnx_dir, meta)


def checkpoint_root(policy: Any) -> Any:
    """The module whose parameter names are the checkpoint's tensor keys."""
    return policy.model


def save_arm_state(
    directory: Path,
    *,
    policy: Any,
    model_path: str,
    results: List[Any],
    export_metadata: Dict[str, Any],
    export_manifest: Dict[str, Any],
    base_model_id: Optional[str] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    """Called by ``quantize``."""
    return _pipeline.save_arm_state(
        directory,
        family=FAMILY,
        model_path=model_path,
        results=results,
        export_manifest=export_manifest,
        extra_files={EXPORT_METADATA_NAME: export_metadata},
        base_model_id=base_model_id,
        load_kwargs=load_kwargs,
        modules=module_paths(policy),
        checkpoint_root=checkpoint_root(policy),
    )


def install(policy: Any, quantized_model: Union[str, Path], model_path: Optional[str] = None, *, check_checkpoint: bool = True) -> Any:
    """Install a fake-quant arm on a loaded ``Gr00tPolicy``; returns ``(handle, state)``."""
    return _pipeline.install_on_policy(policy, quantized_model, model_path, check_checkpoint=check_checkpoint)


def load_state_for(quantized_model: Union[str, Path], model_path: Optional[str] = None, *, check_checkpoint: bool = True) -> Any:
    """Read an N1.7 quant state and check it belongs to *model_path*."""
    state = _pipeline.load_quantized_model(quantized_model)
    if state.manifest.get("family") != FAMILY:
        raise ValueError(f"{quantized_model} is a {state.manifest.get('family')!r} quant state, not {FAMILY!r}")
    if check_checkpoint:
        _pipeline.verify_base_checkpoint(state, _pipeline.resolve_model_path(state, quantized_model, model_path))
    return state


if __name__ == "__main__":
    _pipeline.main()
