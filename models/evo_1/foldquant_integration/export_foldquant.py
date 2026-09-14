# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Emit the FoldQuant plugin graphs for an Evo-1 checkpoint.

Writes, under ``--output-dir``::

    onnx/llm_bf16.onnx          FoldQuant InternVL3 (Qwen2) tower graph (unless --llm-scheme none)
    onnx/action_head_bf16.onnx  FoldQuant denoise-step graph (unless --head-scheme none)
    onnx/export_metadata.json   captured shapes, for the engine builder
    onnx/foldquant_export.json  what was exported, from which samples, needing which plugins

The tower graph is the whole fused-token pass: ``inputs_embeds`` (vision tiles
and prompt embeddings, produced in PyTorch) and the padding ``attention_mask``
in, ``hidden_states`` out. Upstream pads the prompt to a fixed length and sends
a fixed three-image set, so that sequence is a constant of the checkpoint and
the engine is static; the flash-attention treatment of the padded keys is what
the emitter's ``padded_query_mask`` reproduces.

The head graph is one denoise step (``action_seq``, ``context_tokens``,
``time_emb`` -> ``velocity``); upstream's 50-step Euler loop stays in PyTorch.

Example::

    python -m foldquant_integration.export_foldquant \\
        --checkpoint-dir <Evo1_LIBERO checkpoint> \\
        --dataset-path <LeRobot LIBERO dataset> \\
        --output-dir exports/evo1_w8a8_w4a4 \\
        --llm-scheme w8a8_sr --head-scheme w4a4_shg
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import tyro

from foldquant import schemes
from foldquant.export import export_action_head, export_llm, install_llm_emulation
from foldquant.float_export import FLOAT, Binding, export_module_float, export_with_example
from foldquant.provenance import public_path

from . import calibration
from ._upstream import (
    EXPORT_METADATA_NAME,
    LIBERO_ARM_KEY,
    LIBERO_CHECKPOINT,
    LIBERO_DATASET_KEY,
    MANIFEST_NAME,
)
from .runtime import ContextCapture, head_module, llm_module, model_of

logger = logging.getLogger("foldquant.evo_1.export")

_NONE = ("", "none")


@dataclass
class ExportConfig:
    output_dir: str
    """Destination; the graphs land in ``<output_dir>/onnx``."""

    dataset_path: str
    """LeRobot dataset the calibration observations are drawn from (local path or hub id)."""

    checkpoint_dir: str = LIBERO_CHECKPOINT
    """Upstream checkpoint directory (``config.json``, ``norm_stats.json``, ``mp_rank_00_model_states.pt``)."""

    llm_scheme: str = schemes.W8A8_SR
    """FoldQuant scheme for the InternVL3 language tower, or ``none`` to keep it float."""

    head_scheme: str = schemes.W4A4_SHG
    """FoldQuant scheme for the flow-matching action head, or ``none`` to keep it float."""

    num_calib: int = 128
    """Calibration observations (episode, step) pairs spread over the dataset."""

    seed: int = 0
    """Seed of the calibration sample and of the flow-matching start replayed during calibration."""

    episodes: str = ""
    """Optional episodes to restrict the dataset load to, as indices or ranges (``0-149``, ``0,3,7``)."""

    arm_key: str = LIBERO_ARM_KEY
    """Normalizer arm key; empty reads it off the checkpoint's norm_stats.json."""

    dataset_key: str = LIBERO_DATASET_KEY
    """Normalizer dataset key; empty reads it off the checkpoint's norm_stats.json."""

    cascade: bool = False
    """Calibrate the head on the activations of the *quantized* tower (fake-quant emulation)."""

    llm_params: str = "{}"
    """JSON overrides for the tower fold (sq_alpha, act_clip_ratio, site_bits, rot_block_size)."""

    head_params: str = "{}"
    """JSON overrides for the head fold (sq_alpha, sq_fold_order)."""

    device: str = "cuda"


def _scheme_or_none(value: str) -> str | None:
    return None if value.strip().lower() in _NONE else value.strip()


def capture_shape_metadata(deployed, request: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """One inference with the tower seam recording: the shapes the engine builder profiles."""
    model = model_of(deployed)
    head = head_module(deployed)
    seen: dict[str, Any] = {}

    def _step_hook(_m, _args):
        seen["head_calls"] = seen.get("head_calls", 0) + 1

    handle = head.norm_out.register_forward_pre_hook(_step_hook)
    try:
        with ContextCapture(deployed) as capture:
            calibration.infer(deployed, request, seed=seed)
    finally:
        handle.remove()
    if len(capture.hidden) != 1:
        raise RuntimeError(f"expected one tower pass per inference, saw {len(capture.hidden)}")
    batch, seq_len, hidden = capture.hidden[0].shape
    if batch != 1:
        raise RuntimeError(f"unexpected fused-token shape {tuple(capture.hidden[0].shape)}")
    config = model.config
    seen.update(
        {
            "batch_size": int(batch),
            "seq_len": int(seq_len),
            "llm_hidden_size": int(hidden),
            "llm_layers": len(llm_module(deployed).model.layers),
            # The head cross-attends the fused tokens plus the state token.
            "context_len": int(seq_len) + (1 if head.state_encoder is not None else 0),
            "horizon": int(model.horizon),
            "per_action_dim": int(model.per_action_dim),
            "action_dim": int(config.action_dim),
            "num_steps": int(seen.get("head_calls", getattr(config, "num_inference_timesteps", 50))),
        }
    )
    return seen



# ---------------------------------------------------------------------------
# Float (unquantized) engines. Two things make Evo-1 more than a plain trace, and
# both mirror what the plugin graph does (foldquant/llm.py, padded_query_mask):
#  * the prompt is padded to a fixed length and upstream runs flash-attn, which
#    unpads/attends/re-pads with zeros. A traced eager attention must reproduce
#    that: an additive key-padding bias at HALF the causal mask's magnitude (so a
#    padded query's own key, both causally blocked and padded, sums to a finite
#    value instead of -inf -> NaN), and each layer's attention output zeroed at
#    padded QUERY positions before o_proj;
#  * the action head's deployed unit is one Euler step, not its forward.
# ---------------------------------------------------------------------------
import math
from contextlib import contextmanager


@contextmanager
def _flash_parity_attention(tower: torch.nn.Module, masks: dict):
    """Swap every ``Qwen2Attention.forward`` for an eager re-implementation that reads
    ``masks['pad_bias']`` ([B,1,1,S]) and ``masks['keep']`` ([B,S,1]) set by the wrapper."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    layers = tower.layers if hasattr(tower, "layers") else tower.model.layers
    saved = []

    def make_forward(attn):
        def forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                    output_attentions=False, use_cache=False, **_):
            bsz, q_len, _h = hidden_states.size()
            q = attn.q_proj(hidden_states).view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            cos, sin = attn.rotary_emb(v, seq_len=q_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            k = repeat_kv(k, attn.num_key_value_groups)
            v = repeat_kv(v, attn.num_key_value_groups)
            w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(attn.head_dim)
            # Under flash_attention_2 the model hands the layers NO mask and relies on the kernel's
            # causal flag, so build causality here and ignore whatever HF passed. Causal at half
            # finfo.min, padding at a quarter: their sum stays finite in bf16 (see header).
            causal = torch.triu(torch.full((q_len, q_len), torch.finfo(torch.bfloat16).min * 0.5,
                                           dtype=w.dtype, device=w.device), diagonal=1)
            w = w + causal[None, None] + masks["pad_bias"].to(w.dtype)
            w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
            o = torch.matmul(w, v).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            o = o * masks["keep"].to(o.dtype)          # flash parity: padded queries emit zeros
            return attn.o_proj(o), None, past_key_value
        return forward

    for layer in layers:
        saved.append((layer.self_attn, layer.self_attn.forward))
        layer.self_attn.forward = make_forward(layer.self_attn)
    try:
        yield
    finally:
        for attn, fwd in saved:
            attn.forward = fwd


def export_llm_float_evo1(tower, onnx_path, *, forward_loop):
    masks: dict = {}

    def call(inputs_embeds, attention_mask, **rest):
        am = attention_mask.to(inputs_embeds.dtype)
        neg = torch.finfo(torch.bfloat16).min * 0.25
        # (mask - 1) * |neg| -> 0 at real keys, -|neg| at padded keys; |neg| = finfo.min/4 (see header)
        masks["pad_bias"] = ((am - 1.0) * (-neg))[:, None, None, :]
        masks["keep"] = am[:, :, None]
        # attention_mask=None: HF builds the causal mask only; padding enters through masks[].
        rest = {k: v for k, v in rest.items() if k not in ("attention_mask",)}
        rest["output_hidden_states"] = True
        with _flash_parity_attention(tower, masks):
            return tower(inputs_embeds=inputs_embeds, attention_mask=None, **rest)

    from foldquant.float_export import capture_call
    c = capture_call(tower, forward_loop)
    kw = dict(c.kwargs)
    if c.args:
        import inspect
        names = [p for p in inspect.signature(tower.forward).parameters if p != "self"]
        for n, v in zip(names, c.args):
            kw.setdefault(n, v)
    return export_with_example(
        tower, onnx_path, module_name="llm",
        bindings=[Binding("inputs_embeds", "inputs_embeds", torch.bfloat16, {1: "seq_len"}),
                  Binding("attention_mask", "attention_mask", torch.int64, {1: "seq_len"})],
        output_name="hidden_states", output_dynamic={1: "seq_len"},
        example_kwargs=kw, extract=lambda o: o.hidden_states[-1], call=call,
    )


class _HeadStep(torch.nn.Module):
    """One Euler step of the flow-matching head: the unit the engine replaces."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, action_seq, context_tokens, time_emb):
        h = self.head
        B = action_seq.shape[0]
        emb = torch.zeros(B, dtype=torch.long, device=action_seq.device)
        x = h._project_actions(action_seq, emb).to(h.dtype)
        ctx = context_tokens.to(h.dtype)
        te = time_emb.to(h.dtype)
        for block in h.transformer_blocks:
            x = block(x, ctx, te)
        x = h.norm_out(x)
        pooled = h.seq_pool_proj(x.reshape(B, -1)) if h.horizon > 1 else x.squeeze(1)
        return h.mlp_head(pooled, emb)


def export_head_float_evo1(head, onnx_path, *, forward_loop):
    seen = {}

    def hook(_m, args, kwargs):
        seen["context_tokens"] = (args[1] if len(args) > 1 else kwargs["context_tokens"]).detach()
        seen["time_emb"] = (args[2] if len(args) > 2 else kwargs["time_emb"]).detach()
        raise _Stop()

    class _Stop(Exception):
        pass

    hnd = head.transformer_blocks[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            try:
                forward_loop(head)
            except _Stop:
                pass
    finally:
        hnd.remove()
    if "context_tokens" not in seen:
        raise RuntimeError("forward_loop never reached the head's first transformer block")
    B = seen["context_tokens"].shape[0]
    action_seq = (torch.rand(B, max(head.horizon, 1), head.per_action_dim,
                             device=seen["context_tokens"].device, dtype=torch.bfloat16) * 2 - 1)
    step = _HeadStep(head)
    return export_with_example(
        head, onnx_path, module_name="action_head",
        bindings=[Binding("action_seq", "action_seq", torch.bfloat16),
                  Binding("context_tokens", "context_tokens", torch.bfloat16),
                  Binding("time_emb", "time_emb", torch.bfloat16)],
        output_name="velocity",
        example_kwargs={"action_seq": action_seq, "context_tokens": seen["context_tokens"], "time_emb": seen["time_emb"]},
        extract=lambda o: o, call=lambda **k: step(**k),
    )

def main(args: ExportConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    llm_scheme = _scheme_or_none(args.llm_scheme)
    head_scheme = _scheme_or_none(args.head_scheme)
    if llm_scheme is None and head_scheme is None:
        raise SystemExit("nothing to export: both --llm-scheme and --head-scheme are none")
    if llm_scheme is not None:
        schemes.validate("llm", llm_scheme)
    if head_scheme is not None:
        schemes.validate("action_head", head_scheme)
    if args.cascade and (llm_scheme is None or head_scheme is None):
        raise SystemExit("--cascade needs both an LLM scheme and a head scheme")
    if args.cascade and llm_scheme not in schemes.LLM_FOLDED_SCHEMES:
        raise SystemExit(f"--cascade emulates a folded tower; {llm_scheme!r} folds nothing")
    llm_params = json.loads(args.llm_params)
    head_params = json.loads(args.head_params)

    out = Path(args.output_dir) / "onnx"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    deployed = calibration.load_policy(
        args.checkpoint_dir, arm_key=args.arm_key, dataset_key=args.dataset_key, device=args.device
    )
    dataset = calibration.load_dataset(args.dataset_path, episodes=calibration.parse_episodes(args.episodes))
    logger.info("model + dataset in %.0fs", time.time() - t0)

    samples, observations = calibration.sample_observations(dataset, args.num_calib, seed=args.seed)
    loop = calibration.make_forward_loop(deployed, observations, seed=args.seed)
    tower = llm_module(deployed)

    shapes = capture_shape_metadata(deployed, observations[0], seed=args.seed)
    logger.info("captured shapes: %s", shapes)

    results = []
    plugin_libs: list = []
    llm_result = None
    if llm_scheme is not None:
        t1 = time.time()
        if llm_scheme == FLOAT:
            llm_result = export_llm_float_evo1(tower, out / "llm_bf16.onnx", forward_loop=loop)
        else:
            llm_result = export_llm(
                tower, out / "llm_bf16.onnx", scheme=llm_scheme, forward_loop=loop, params=llm_params or None
            )
        results.append(llm_result)
        logger.info("tower %s exported in %.0fs", llm_scheme, time.time() - t1)

    if head_scheme is not None:
        t1 = time.time()
        emulation = None
        if args.cascade:
            assert llm_result is not None
            emulation = install_llm_emulation(tower, llm_result)
            logger.info("cascade: head calibration runs under the quantized-tower emulation")
        try:
            if head_scheme == FLOAT:
                head_result = export_head_float_evo1(head_module(deployed), out / "action_head_bf16.onnx", forward_loop=loop)
            else:
                head_result = export_action_head(
                    head_module(deployed),
                    out / "action_head_bf16.onnx",
                    scheme=head_scheme,
                    forward_loop=loop,
                    params=head_params or None,
                )
        finally:
            if emulation is not None:
                emulation.remove()
        results.append(head_result)
        logger.info("action head %s exported in %.0fs", head_scheme, time.time() - t1)

    for r in results:
        for lib in r.plugin_libs:
            if lib not in plugin_libs:
                plugin_libs.append(lib)

    metadata = {
        "model": "evo_1",
        "checkpoint_dir": public_path(args.checkpoint_dir),
        "seq_len": shapes["seq_len"],
        "context_len": shapes["context_len"],
        "llm_hidden_size": shapes["llm_hidden_size"],
        "llm_layers": shapes["llm_layers"],
        "horizon": shapes["horizon"],
        "per_action_dim": shapes["per_action_dim"],
        "action_dim": shapes["action_dim"],
        "num_steps": shapes["num_steps"],
        "export_mode": "foldquant",
        "precision": "bf16",
        "batch_size": shapes["batch_size"],
    }
    (out / EXPORT_METADATA_NAME).write_text(json.dumps(metadata, indent=2))
    manifest = {
        "checkpoint_dir": public_path(args.checkpoint_dir),
        "dataset_path": public_path(args.dataset_path),
        "episodes": args.episodes,
        "arm_key": deployed.arm_key,
        "dataset_key": deployed.dataset_key,
        "schemes": {r.module: r.scheme for r in results},
        "params": {"llm": llm_params, "action_head": head_params},
        "cascade": bool(args.cascade),
        "plugin_libs": plugin_libs,
        "files": {r.module: r.onnx_path.name for r in results},
        "calibration": {
            "seed": args.seed,
            "num_samples": len(samples),
            "samples": [asdict(s) for s in samples],
        },
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s and %s (total %.0fs)", EXPORT_METADATA_NAME, MANIFEST_NAME, time.time() - t0)
    return out


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
