# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""INT8 per-row plugin graph for Evo-1's flow-matching action head.

Replaces the exported ``action_head.onnx`` (one denoise step) with a graph whose
every GEMM runs through the ``gr00t::v1`` INT8 per-row plugins, exactly like the
GR00T DiT surgery — same plugin library, same I/O contract as the graph it
replaces (``action_seq [1,50,24]``, ``context_tokens [1,1025,896]``,
``time_emb [1,896]`` -> ``velocity [1,1200]``; the graph is fully static, see the
``nn.MultiheadAttention`` note below).

Evo-1's block is *not* a GR00T DiT block, but it decomposes onto the same
kernels exactly:

- ``norm1`` is an **affine** LayerNorm where the plugin's fused prologue is
  AdaLN (``LayerNorm(elementwise_affine=False) * (1 + scale) + shift`` — see
  ``fused_adaln_quant.h``). The identity ``LN_gamma_beta(x) =
  LN_noaffine(x) * (1 + (gamma - 1)) + beta`` maps it with **constant**
  ``scale = gamma - 1`` / ``shift = beta`` initializers where GR00T feeds
  runtime AdaLN tensors. Mathematically exact, not an approximation.
- The cross-attention is ``nn.MultiheadAttention(q=normed, k=ctx, v=ctx)``:
  its packed ``in_proj_weight [3E, E]`` splits into the plugin's
  ``weight_q`` / ``weight_kv`` rows in q,k,v order, biases likewise, and
  ``FusedCrossAttnFull`` already includes the residual add.
- The FFN has no plugin-shaped fusion (its norm is followed by a **runtime**
  time-embedding add, which ``FusedFfnBlock``'s baked norm cannot express), so
  it is emitted the way the LLM surgery emits attention math: plain ONNX for
  LayerNorm/Add/GELU, ``PerRowInt8LinearResidual`` for both GEMMs. ``ff.0`` has
  no residual; the graph is static ``B=1``, so a zeros initializer of the exact
  output shape serves as one.
- ``action_encoder`` (24->896 MLP + positional table) and the output head
  (``norm_out`` -> ``seq_pool_proj`` -> ``mlp_head``) stay BF16 ONNX: tiny
  matmuls bracketing the 8 blocks, same reasoning as the LLM graph's
  embed_tokens residual.

All ``CategorySpecific*`` modules are plain ``nn.Linear`` here — the ported
Evo-1 rejects ``num_categories > 1`` at construction.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
import onnx.helper as oh

from . import foldq
from .dit_common import (
    EPS as _EPS,
)
from .dit_common import (
    PLUGIN_DOMAIN as _PLUGIN_DOMAIN,
)
from .dit_common import (
    PLUGIN_NAMESPACE as _PLUGIN_NAMESPACE,
)
from .dit_common import (
    PLUGIN_VERSION as _PLUGIN_VERSION,
)
from .dit_common import (
    bf16_initializer,
    bias_f32,
    to_bytes_f32,
)

#: Rotation block width, matching the INT4 head emitter.
_DEFAULT_BLOCK_SIZE = 64

logger = logging.getLogger(__name__)

__all__ = ["build_evo1_head_plugin_onnx"]


def _torch() -> Any:
    import torch

    return torch


def _i64_init(name: str, values: list) -> onnx.TensorProto:
    return oh.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def _linear(nodes: list, inits: list, name: str, x: str, weight: Any, bias: Any, out: str) -> None:
    """BF16 ``x @ W.T + b`` as MatMul+Add (keeps every initializer BF16)."""
    inits.append(bf16_initializer(f"{name}_w", weight.t().contiguous()))
    nodes.append(oh.make_node("MatMul", [x, f"{name}_w"], [f"{name}_mm"]))
    inits.append(bf16_initializer(f"{name}_b", bias))
    nodes.append(oh.make_node("Add", [f"{name}_mm", f"{name}_b"], [out]))


def _layernorm(nodes: list, inits: list, name: str, x: str, gamma: Any, beta: Any, out: str) -> None:
    inits.append(bf16_initializer(f"{name}_g", gamma))
    inits.append(bf16_initializer(f"{name}_b", beta))
    nodes.append(oh.make_node("LayerNormalization", [x, f"{name}_g", f"{name}_b"], [out], axis=-1, epsilon=float(_EPS)))


def _gelu_erf(nodes: list, inits: list, name: str, x: str, out: str) -> None:
    """``nn.GELU()`` default (erf form), opset-19-safe."""
    torch = _torch()
    inits.append(bf16_initializer(f"{name}_isqrt2", torch.tensor(1.0 / math.sqrt(2.0), dtype=torch.bfloat16)))
    inits.append(bf16_initializer(f"{name}_half", torch.tensor(0.5, dtype=torch.bfloat16)))
    inits.append(bf16_initializer(f"{name}_one", torch.tensor(1.0, dtype=torch.bfloat16)))
    nodes.append(oh.make_node("Mul", [x, f"{name}_isqrt2"], [f"{name}_s"]))
    nodes.append(oh.make_node("Erf", [f"{name}_s"], [f"{name}_erf"]))
    nodes.append(oh.make_node("Add", [f"{name}_erf", f"{name}_one"], [f"{name}_e1"]))
    nodes.append(oh.make_node("Mul", [x, f"{name}_e1"], [f"{name}_xe"]))
    nodes.append(oh.make_node("Mul", [f"{name}_xe", f"{name}_half"], [out]))


def resolve_head_dims(cfg: Any) -> tuple:
    """``(horizon, per_action_dim, total_action_dim)`` for an Evo-1 head config.

    The released Evo-1 config spells the chunk width ``action_dim`` (a property
    over ``action_horizon or horizon`` times ``per_action_dim``); the ported
    config this emitter was first written against spells it
    ``total_action_dim``. Both are the same number, so read whichever is there
    and cross-check it against the factors rather than trusting one name.

    ``action_horizon`` overrides ``horizon`` where it is set, exactly as
    upstream's ``EVO1.__init__`` resolves it — a checkpoint that sets it would
    otherwise emit a graph of the wrong chunk length with no error anywhere.
    """
    horizon = getattr(cfg, "action_horizon", None) or int(cfg.horizon)
    horizon = int(horizon)
    per_action = int(cfg.per_action_dim)
    total = getattr(cfg, "total_action_dim", None)
    if total is None:
        total = getattr(cfg, "action_dim", None)
    if total is None:
        raise AttributeError(
            "the action-head config has neither 'total_action_dim' nor 'action_dim'; "
            "the emitter cannot size the velocity output"
        )
    total = int(total)
    if total != horizon * per_action:
        raise ValueError(
            f"action head config disagrees with itself: total {total} != horizon {horizon} x "
            f"per_action_dim {per_action}"
        )
    return horizon, per_action, total


def _stacked_kv_weights(action_expert: Any) -> Any:
    """Stack every block's ``[K; V]`` in_proj rows -> the shared encoder rotation source.

    Every block cross-attends the same context, so one shared rotation is built
    from the stacked KV weights and reused by ``EncoderPreQuantInt4`` and every
    block's KV pack — preserving the once-per-forward context quant.
    """
    torch = _torch()
    dim = int(action_expert.config.embed_dim)
    kv_stack = []
    for block in action_expert.transformer_blocks:
        w_in = block.attn.in_proj_weight.detach()
        kv_stack.append(w_in[dim:])  # rows [K; V]
    return torch.cat(kv_stack, dim=0)


def build_evo1_head_plugin_onnx(
    action_expert: Any,
    out_path: "str | Path",
    *,
    sq_scales: "dict | None" = None,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    fold_order: str = "after",
    fwht: bool = False,
) -> None:
    """Write the INT8 per-row plugin graph for *action_expert* to *out_path*.

    Takes the same fold knobs as the INT4 emitter, through the same
    :mod:`foldq` calls: bit width is the only thing that differs between the two.
    Passing ``sq_scales=None`` builds the unfolded arm this module shipped
    before, byte for byte.

    Args:
        action_expert: Live ``Evo1FlowMatchingActionExpert`` (weights on any
            device; everything is snapshotted to initializers).
        out_path: Destination ``.onnx``.
        sq_scales: Per-site activation scales from
            :func:`evo1_head_int4.compute_evo1_head_sq_scales` — the capture is
            bit-width agnostic, so one function feeds both emitters.
        block_size: rotation block width.
        fold_order: ``"before"`` (raw frame) or ``"after"`` (rotated frame).
        fwht: fixed butterfly rather than the learned dense rotation.
    """
    torch = _torch()
    sd = {k: v.detach() for k, v in action_expert.state_dict().items()}
    cfg = action_expert.config
    dim = int(cfg.embed_dim)
    horizon, per_action, total_action = resolve_head_dims(cfg)
    num_heads = int(cfg.num_heads)
    head_dim = dim // num_heads
    ff_inner = dim * 4
    num_blocks = len(action_expert.transformer_blocks)
    # The exported graph is fully static (B=1, ctx length baked): read the
    # context length off the positional table the checkpoint pinned... the ctx
    # length is an I/O property; take it from the exported spec: 1024 prompt
    # tokens + 1 state token.
    ctx_len = 1025

    nodes: list = []
    inits: list = []

    def _macro(weight: Any, s_ch: Any) -> tuple:
        """One site of a fused macro node: packed INT8 + scale + the paired rotation."""
        perm, R = foldq.site_rotation(weight, block_size, fwht)
        i8, sc, r_use = foldq.fold_macro_site(weight, perm, R, block_size, s_ch, fold_order=fold_order, bits=8)
        return i8, sc, perm, r_use

    def _per_row(weight: Any, s_ch: Any) -> tuple:
        """One single-site PerRowInt8LinearResidual: foldq's attrs splat straight in."""
        return foldq.fold_site(weight, bits=8, block_size=block_size, s_ch=s_ch, fold_order=fold_order, fwht=fwht)

    def _scale(key: str) -> Any:
        return sq_scales[key] if sq_scales is not None else None

    def _rot_attr() -> Dict[str, Any]:
        """The block width a butterfly node rotates in; absent on the dense arm."""
        return {"rot_block_size": int(block_size)} if (fwht and sq_scales is not None) else {}

    def _pre_vec(field: str, s_ch: Any) -> Dict[str, Any]:
        if not fwht or s_ch is None:
            return {}
        return {field: to_bytes_f32(s_ch.detach().float().cpu().numpy())}

    # The encoder's rotation is shared by every block's KV pack, so it is built
    # once from the stacked KV weights — same site key the INT4 emitter uses.
    # One rotation for the shared context, built once from the stacked KV weights
    # exactly as the INT4 emitter does — every block's KV pack folds under it.
    enc_perm, enc_R = foldq.site_rotation(_stacked_kv_weights(action_expert), block_size, fwht)

    _enc_fold: dict = {}
    if sq_scales is not None and fwht:
        _enc_fold = {
            "rot_block_size": int(block_size),
            "act_scale_pre_enc": to_bytes_f32(_scale("encoder").detach().float().cpu().numpy()),
        }

    x_in = oh.make_tensor_value_info("action_seq", onnx.TensorProto.BFLOAT16, [1, horizon, per_action])
    c_in = oh.make_tensor_value_info("context_tokens", onnx.TensorProto.BFLOAT16, [1, ctx_len, dim])
    t_in = oh.make_tensor_value_info("time_emb", onnx.TensorProto.BFLOAT16, [1, dim])
    y_out = oh.make_tensor_value_info("velocity", onnx.TensorProto.BFLOAT16, [1, total_action])

    # ===== action_encoder: ReLU(W1 a) -> +pos -> ReLU(W2) -> W3 (BF16) =====
    _linear(
        nodes,
        inits,
        "enc_w1",
        "action_seq",
        sd["action_encoder.W1.linear.weight"],
        sd["action_encoder.W1.linear.bias"],
        "enc_h1",
    )
    nodes.append(oh.make_node("Relu", ["enc_h1"], ["enc_h1r"]))
    pos = action_expert.action_encoder.pos_encoding(horizon)
    inits.append(bf16_initializer("enc_pos", pos.reshape(1, horizon, -1)))
    nodes.append(oh.make_node("Add", ["enc_h1r", "enc_pos"], ["enc_h1p"]))
    _linear(
        nodes,
        inits,
        "enc_w2",
        "enc_h1p",
        sd["action_encoder.W2.linear.weight"],
        sd["action_encoder.W2.linear.bias"],
        "enc_h2",
    )
    nodes.append(oh.make_node("Relu", ["enc_h2"], ["enc_h2r"]))
    _linear(
        nodes,
        inits,
        "enc_w3",
        "enc_h2r",
        sd["action_encoder.W3.linear.weight"],
        sd["action_encoder.W3.linear.bias"],
        "x0",
    )

    # time_emb [1, dim] -> [1, 1, dim] for the per-token add inside each block.
    inits.append(_i64_init("time_axes", [1]))
    nodes.append(oh.make_node("Unsqueeze", ["time_emb", "time_axes"], ["time_b"]))

    # ===== EncoderPreQuant: one INT8 quant of the context, shared by all blocks =====
    nodes.append(
        oh.make_node(
            "EncoderPreQuant",
            ["context_tokens"],
            ["encoder_i8", "encoder_scale"],
            name="encoder_prequant",
            domain=_PLUGIN_DOMAIN,
            plugin_namespace=_PLUGIN_NAMESPACE,
            plugin_version=_PLUGIN_VERSION,
            **_enc_fold,
            K_enc=int(dim),
        )
    )

    # Evo-1 cross-attends the whole padded context with NO key-padding mask —
    # that is the trained behaviour. FusedCrossAttnFull treats
    # 5 inputs (no mask tensor) as fully unmasked, so no mask is emitted; a
    # baked int32 "all-ones" mask is BOTH the wrong dtype (the plugin wants
    # BF16 additive) and the wrong semantics (additive, 0=keep).

    # ff.0 has no residual; static shapes make a zeros initializer exact.
    zeros_ff = np.zeros((1, horizon, ff_inner), dtype=np.float32)
    inits.append(bf16_initializer("ff_zero_res", torch.from_numpy(zeros_ff)))

    cur_x = "x0"
    for idx in range(num_blocks):
        b = f"block{idx}"
        p = f"transformer_blocks.{idx}."

        # --- norm1 as constant AdaLN: scale = gamma - 1, shift = beta ---
        g1 = sd[p + "norm1.weight"].float()
        inits.append(bf16_initializer(f"{b}_scale", (g1 - 1.0).reshape(1, dim)))
        inits.append(bf16_initializer(f"{b}_shift", sd[p + "norm1.bias"].reshape(1, dim)))

        # --- packed in_proj -> q / kv in the plugin's layout ---
        w_in = sd[p + "attn.in_proj_weight"]
        b_in = sd[p + "attn.in_proj_bias"]
        wQ, wK, wV = w_in[:dim], w_in[dim : 2 * dim], w_in[2 * dim :]
        bQ, bK, bV = b_in[:dim], b_in[dim : 2 * dim], b_in[2 * dim :]
        wO = sd[p + "attn.out_proj.weight"]
        s_q = _scale(f"{b}_q")
        s_o = _scale(f"{b}_o")
        s_enc = _scale("encoder")
        q_i8, sQ_b, permQ, RQ_use = _macro(wQ, s_q)
        o_i8, sO_b, permO, RO_use = _macro(wO, s_o)
        # KV goes under the SHARED encoder rotation, like the INT4 emitter: every
        # block reads the same pre-quantized context, so one rotation covers them.
        wKV = torch.cat([wK, wV], dim=0)
        kv_i8, sKV_b, _ = foldq.fold_macro_site(wKV, enc_perm, enc_R, block_size, s_enc, fold_order=fold_order, bits=8)
        # The butterfly ships the raw per-channel vector; the dense arm folds it
        # into the matrix instead, so only one of the two ever rides along.
        bf_q = _pre_vec("act_scale_pre_in", s_q)
        bf_o = _pre_vec("act_scale_pre_o", s_o)
        nodes.append(
            oh.make_node(
                "FusedCrossAttnFull",
                [cur_x, f"{b}_scale", f"{b}_shift", "encoder_i8", "encoder_scale"],
                [f"{b}_post_attn"],
                name=f"{b}_crossattn_full",
                domain=_PLUGIN_DOMAIN,
                plugin_namespace=_PLUGIN_NAMESPACE,
                plugin_version=_PLUGIN_VERSION,
                inner_dim=int(dim),
                K=int(dim),
                K_enc=int(dim),
                num_heads=num_heads,
                head_dim=head_dim,
                eps=float(_EPS),
                **_rot_attr(),
                weight_q_i8=q_i8,
                weight_q_scale=sQ_b,
                bias_q=to_bytes_f32(bias_f32(bQ)),
                **bf_q,
                weight_kv_i8=kv_i8,
                weight_kv_scale=sKV_b,
                bias_kv=to_bytes_f32(np.concatenate([bias_f32(bK), bias_f32(bV)])),
                weight_o_i8=o_i8,
                weight_o_scale=sO_b,
                bias_o=to_bytes_f32(bias_f32(sd[p + "attn.out_proj.bias"])),
                **bf_o,
            )
        )
        h = f"{b}_post_attn"

        # --- FFN: LN2 -> +time -> INT8 GEMM -> GELU -> INT8 GEMM + residual ---
        _layernorm(nodes, inits, f"{b}_ln2", h, sd[p + "norm2.weight"], sd[p + "norm2.bias"], f"{b}_x2")
        nodes.append(oh.make_node("Add", [f"{b}_x2", "time_b"], [f"{b}_x2t"]))
        w0_i8, s0_b, ff0_attrs = _per_row(sd[p + "ff.0.weight"], _scale(f"{b}_ffn0"))
        nodes.append(
            oh.make_node(
                "PerRowInt8LinearResidual",
                [f"{b}_x2t", "ff_zero_res"],
                [f"{b}_ffh_raw"],
                name=f"{b}_ff0",
                domain=_PLUGIN_DOMAIN,
                plugin_namespace=_PLUGIN_NAMESPACE,
                plugin_version=_PLUGIN_VERSION,
                N=int(ff_inner),
                K=int(dim),
                weight_i8=w0_i8,
                weight_scale=s0_b,
                **ff0_attrs,
            )
        )
        inits.append(bf16_initializer(f"{b}_ff0_b", sd[p + "ff.0.bias"]))
        nodes.append(oh.make_node("Add", [f"{b}_ffh_raw", f"{b}_ff0_b"], [f"{b}_ffh"]))
        _gelu_erf(nodes, inits, f"{b}_gelu", f"{b}_ffh", f"{b}_ffg")
        w2_i8, s2_b, ff2_attrs = _per_row(sd[p + "ff.2.weight"], _scale(f"{b}_ffn2"))
        nodes.append(
            oh.make_node(
                "PerRowInt8LinearResidual",
                [f"{b}_ffg", h],
                [f"{b}_out_raw"],
                name=f"{b}_ff2",
                domain=_PLUGIN_DOMAIN,
                plugin_namespace=_PLUGIN_NAMESPACE,
                plugin_version=_PLUGIN_VERSION,
                N=int(dim),
                K=int(ff_inner),
                weight_i8=w2_i8,
                weight_scale=s2_b,
                **ff2_attrs,
            )
        )
        inits.append(bf16_initializer(f"{b}_ff2_b", sd[p + "ff.2.bias"]))
        nodes.append(oh.make_node("Add", [f"{b}_out_raw", f"{b}_ff2_b"], [f"{b}_out"]))
        cur_x = f"{b}_out"

    # ===== output head (BF16): norm_out -> reshape -> seq_pool -> mlp_head =====
    _layernorm(nodes, inits, "norm_out", cur_x, sd["norm_out.weight"], sd["norm_out.bias"], "xn")
    inits.append(_i64_init("pool_shape", [1, horizon * dim]))
    nodes.append(oh.make_node("Reshape", ["xn", "pool_shape"], ["x_flat"], allowzero=0))
    _linear(nodes, inits, "seq_pool", "x_flat", sd["seq_pool_proj.weight"], sd["seq_pool_proj.bias"], "x_pool")
    _linear(
        nodes, inits, "head_fc1", "x_pool", sd["mlp_head.fc1.linear.weight"], sd["mlp_head.fc1.linear.bias"], "head_h"
    )
    nodes.append(oh.make_node("Relu", ["head_h"], ["head_hr"]))
    _linear(
        nodes,
        inits,
        "head_fc2",
        "head_hr",
        sd["mlp_head.fc2.linear.weight"],
        sd["mlp_head.fc2.linear.bias"],
        "velocity",
    )

    graph = oh.make_graph(nodes, "evo1_action_head_int8_per_row", [x_in, c_in, t_in], [y_out], initializer=inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 19), oh.make_opsetid(_PLUGIN_DOMAIN, 1)])
    model.ir_version = 9
    onnx.save(model, str(out_path))
    logger.info(
        "Built Evo-1 action-head INT8 per-row plugin graph: %d blocks, %d nodes -> %s",
        num_blocks,
        len(nodes),
        out_path,
    )
