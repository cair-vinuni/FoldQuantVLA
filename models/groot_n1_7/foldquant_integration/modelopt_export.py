# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The ModelOpt INT8 SmoothQuant baseline arm for GR00T N1.7 (see :mod:`foldquant.modelopt_int8`).

Both towers are quantized **in place on the live policy**, calibrated by
replaying the calls captured on them, and exported with upstream's
full-pipeline contract. The graphs therefore replace ``llm_bf16.onnx`` and
``dit_bf16.onnx`` exactly as the FoldQuant graphs do, with the same names,
dtypes and dynamic dims:

* LLM: ``inputs_embeds, attention_mask, position_ids, visual_pos_masks,
  deepstack_0..2 -> embeddings``. :class:`LLMQDQExport` is upstream's
  ``LLMForExport`` (eager attention, simple causal mask, deepstack via
  ``masked_scatter``, no final norm), but it calls the live decoder layers
  instead of a copy: a copied ``state_dict`` would drop the quantizers' amax
  and export a float graph.
* DiT: ``sa_embs, vl_embs, timestep, image_mask, backbone_attention_mask -> output``,
  upstream's ``DiTWrapper`` around the live DiT.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from foldquant import modelopt_int8
import torch
from torch import nn

from ._upstream import ensure_deployment_on_path


logger = logging.getLogger("foldquant.groot_n1_7.modelopt")


class LLMQDQExport(nn.Module):
    """Upstream ``LLMForExport`` over the live (quantized) Qwen3-VL text tower."""

    def __init__(self, text_model: nn.Module, num_layers: int, num_deepstack: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(text_model.layers)[:num_layers])
        self.rotary_emb = text_model.rotary_emb
        self.n_deepstack = num_deepstack

    @staticmethod
    def _simple_causal_mask(dtype, device, batch_size, seq_len, attention_mask):
        mask_value = torch.finfo(dtype).min * 0.5
        causal = torch.triu(
            torch.full((seq_len, seq_len), mask_value, device=device, dtype=dtype), diagonal=1
        )
        causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        if attention_mask is not None and attention_mask.dim() == 2:
            causal = causal + (1.0 - attention_mask[:, None, None, :].to(dtype)) * mask_value
        return causal

    @staticmethod
    def _deepstack_add(hidden_states, visual_pos_masks, visual_embeds):
        mask = visual_pos_masks.unsqueeze(-1)
        delta = torch.zeros_like(hidden_states).masked_scatter(
            mask.expand_as(hidden_states), visual_embeds
        )
        return hidden_states + delta

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        visual_pos_masks=None,
        deepstack_0=None,
        deepstack_1=None,
        deepstack_2=None,
    ):
        batch_size, seq_len = inputs_embeds.shape[:2]
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        attn_mask = self._simple_causal_mask(dtype, device, batch_size, seq_len, attention_mask)
        text_position_ids = position_ids[0]
        cache_position = torch.arange(seq_len, device=device)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        deepstack = [d for d in (deepstack_0, deepstack_1, deepstack_2) if d is not None]
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if visual_pos_masks is not None and layer_idx < len(deepstack):
                hidden_states = self._deepstack_add(
                    hidden_states, visual_pos_masks, deepstack[layer_idx]
                )
        # Pre-norm: the backbone reads hidden_states[-1] (see upstream export_llm_to_onnx).
        return hidden_states


class DiTQDQExport(nn.Module):
    """Upstream ``DiTWrapper``: positional inputs to the DiT's keyword call."""

    def __init__(self, dit: nn.Module) -> None:
        super().__init__()
        self.dit = dit

    def forward(self, sa_embs, vl_embs, timestep, image_mask, backbone_attention_mask):
        return self.dit(
            sa_embs,
            vl_embs,
            timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )


class _EagerAttention:
    """Switch every config reachable from *module* to eager attention for the trace, then restore."""

    def __init__(self, module: nn.Module) -> None:
        self._configs: List[Tuple[Any, Any]] = []
        seen = set()
        for m in module.modules():
            cfg = getattr(m, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation") and id(cfg) not in seen:
                seen.add(id(cfg))
                self._configs.append((cfg, cfg._attn_implementation))

    def __enter__(self) -> "_EagerAttention":
        for cfg, _ in self._configs:
            cfg._attn_implementation = "eager"
        return self

    def __exit__(self, *exc: Any) -> None:
        for cfg, impl in self._configs:
            cfg._attn_implementation = impl


def _finish_graph(
    onnx_path: Path, name: str, *, strip_scatternd_reduction: bool, algorithm: str
) -> Dict[str, Any]:
    ensure_deployment_on_path()
    from export_onnx_n1d7 import _consolidate_external_data, _strip_default_scatternd_reduction
    import onnx

    repairs = modelopt_int8.repair_onnx_dtypes(onnx_path, name)
    # Upstream's engines hand bf16 between components (trt_torch asserts it).
    output_casts = modelopt_int8.cast_graph_outputs(onnx_path, name, onnx.TensorProto.BFLOAT16)
    _consolidate_external_data(str(onnx_path))
    if strip_scatternd_reduction:
        # TensorRT 10.3's parser refuses the (default) attribute; see upstream.
        _strip_default_scatternd_reduction(str(onnx_path))
    # Counted BEFORE the INT4 surgery: it consumes the weight DQ nodes it rewrites,
    # so afterwards their absence is the expected state, not a lost bake.
    qdq = modelopt_int8.require_qdq(onnx_path, name)
    record: Dict[str, Any] = {
        "dtype_repairs": repairs,
        "output_casts_to_bf16": output_casts,
        "qdq_nodes": qdq,
    }
    if modelopt_int8.is_weight_only(algorithm):
        record["int4_groupwise_surgery"] = _int4_surgery(onnx_path, name)
    return record


def _int4_surgery(onnx_path: Path, name: str) -> Dict[str, int]:
    """Rewrite the INT4 weight-only DQ chains to ``Int4GroupwiseGemmPlugin`` nodes, in place.

    TensorRT 10.3 has no INT4 weight-only kernel, so the graph ModelOpt exports
    parses but runs dequantized. The surgery is what makes this arm an INT4
    engine; the plugin library it needs is declared in the export manifest.
    """
    from foldquant.int4_groupwise import apply_int4_modelopt_surgery

    staged = onnx_path.with_suffix(".int4.onnx")
    replaced, materialized = apply_int4_modelopt_surgery(onnx_path, staged)
    if replaced == 0:
        raise RuntimeError(
            f"{name}: INT4 surgery rewrote no weight; the engine would run dequantized"
        )
    for old in (onnx_path, Path(str(onnx_path) + ".data")):
        if old.exists():
            old.unlink()
    staged.replace(onnx_path)
    staged_data = Path(str(staged) + ".data")
    if staged_data.exists():
        staged_data.replace(Path(str(onnx_path) + ".data"))
    logger.info(
        "%s: INT4 groupwise surgery replaced %d weights (%d constants materialized)",
        name,
        replaced,
        materialized,
    )
    return {"replaced": replaced, "materialized": materialized}


def export_llm(
    text_model: nn.Module,
    calls: Sequence[Tuple[tuple, dict]],
    onnx_path: Path,
    *,
    num_layers: int,
    algorithm: str = modelopt_int8.MODELOPT_W8A8_SMOOTHQUANT,
    opset: int = modelopt_int8.DEFAULT_OPSET,
) -> Dict[str, Any]:
    """Quantize the live text tower on *calls* (its captured forwards) and export the Q/DQ graph."""
    record = modelopt_int8.quantize_module(
        text_model, modelopt_int8.replay_loop(calls), algorithm=algorithm
    )

    args, kwargs = calls[0]
    embeds = kwargs.get("inputs_embeds", args[0] if args else None)
    deepstack = list(kwargs.get("deepstack_visual_embeds") or [])
    vis_mask = kwargs.get("visual_pos_masks")
    if embeds is None or kwargs.get("position_ids") is None:
        raise RuntimeError("captured LLM call carries no inputs_embeds / position_ids")
    attention_mask = kwargs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones(embeds.shape[:2], dtype=torch.int64, device=embeds.device)
    if deepstack and vis_mask is None:
        raise RuntimeError("captured LLM call has deepstack features but no visual_pos_masks")

    wrapper = LLMQDQExport(text_model, num_layers, len(deepstack)).eval()
    export_args: List[torch.Tensor] = [
        embeds.clone(),
        attention_mask.to(torch.int64).clone(),
        kwargs["position_ids"].clone(),
    ]
    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
        "position_ids": {2: "seq_len"},
        "embeddings": {1: "seq_len"},
    }
    if deepstack:
        export_args.append(vis_mask.clone())
        input_names.append("visual_pos_masks")
        dynamic_axes["visual_pos_masks"] = {1: "seq_len"}
        for i, ds in enumerate(deepstack):
            export_args.append(ds.clone())
            input_names.append(f"deepstack_{i}")

    logger.info("exporting ModelOpt LLM -> %s (opset %d)", onnx_path, opset)
    with _EagerAttention(text_model), torch.no_grad():
        modelopt_int8.onnx_export(
            wrapper,
            tuple(export_args),
            onnx_path,
            input_names=input_names,
            output_names=["embeddings"],
            dynamic_axes=dynamic_axes,
            opset=opset,
        )
    record.update(
        _finish_graph(Path(onnx_path), "llm", strip_scatternd_reduction=True, algorithm=algorithm)
    )
    record["opset"] = opset
    record["calibration_calls"] = len(calls)
    return record


def export_dit(
    dit: nn.Module,
    calls: Sequence[Tuple[tuple, dict]],
    onnx_path: Path,
    *,
    algorithm: str = modelopt_int8.MODELOPT_W8A8_SMOOTHQUANT,
    opset: int = modelopt_int8.DEFAULT_OPSET,
) -> Dict[str, Any]:
    """Quantize the live DiT on *calls* (every denoising step it saw) and export the Q/DQ graph."""
    record = modelopt_int8.quantize_module(
        dit, modelopt_int8.replay_loop(calls), algorithm=algorithm
    )

    _, kwargs = calls[0]
    missing = [
        k
        for k in (
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "image_mask",
            "backbone_attention_mask",
        )
        if kwargs.get(k) is None
    ]
    if missing:
        raise RuntimeError(f"captured DiT call lacks {missing}; is use_alternate_vl_dit off?")
    export_args = (
        kwargs["hidden_states"].clone(),
        kwargs["encoder_hidden_states"].clone(),
        kwargs["timestep"].clone(),
        kwargs["image_mask"].clone(),
        kwargs["backbone_attention_mask"].clone(),
    )
    wrapper = DiTQDQExport(dit).eval()
    logger.info("exporting ModelOpt DiT -> %s (opset %d)", onnx_path, opset)
    with torch.no_grad():
        modelopt_int8.onnx_export(
            wrapper,
            export_args,
            onnx_path,
            input_names=["sa_embs", "vl_embs", "timestep", "image_mask", "backbone_attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "vl_embs": {1: "vl_seq_len"},
                "image_mask": {1: "vl_seq_len"},
                "backbone_attention_mask": {1: "vl_seq_len"},
            },
            opset=opset,
        )
    record.update(
        _finish_graph(Path(onnx_path), "dit", strip_scatternd_reduction=True, algorithm=algorithm)
    )
    record["opset"] = opset
    record["calibration_calls"] = len(calls)
    return record


def capture(modules: Dict[str, nn.Module], forward_loop) -> Dict[str, list]:
    """One float replay of the calibration set, recording every call to each module in *modules*."""
    store = modelopt_int8.capture_many(modules, forward_loop)
    for name, calls in store.items():
        logger.info("captured %d %s calls", len(calls), name)
    return store
