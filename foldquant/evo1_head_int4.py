# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""FoldQuant W4A4 plugin graph for Evo-1's flow-matching action head.

INT4 analogue of :mod:`evo1_head_int8` — same graph shape, same I/O
contract (``action_seq [1,50,24]``, ``context_tokens [1,1025,896]``,
``time_emb [1,896]`` -> ``velocity [1,1200]``), with every quantized GEMM
running through the ``gr00t::v1`` W4A4 plugins under the SmoothQuant-folded
rotation scheme (``w4a4_sr``):

- ``EncoderPreQuantInt4`` rotates + INT4-quantizes the 1025-token context once,
  under the SHARED encoder rotation built from all blocks' stacked KV weights.
- Each block's cross-attention is one ``FusedCrossAttnFullInt4`` node with the
  same constant-AdaLN mapping as the INT8 emitter (``scale = gamma - 1``,
  ``shift = beta``) and NO attention-mask input — Evo-1 attends the whole
  padded context maskless (trained behaviour), and the plugin's mask input is
  optional (present iff 6 inputs).
- The FFN keeps the INT8 emitter's decomposition (BF16 LayerNorm -> runtime
  time-embedding add -> GEMM -> erf-GELU -> GEMM + residual): the fused FFN
  macro plugin bakes a no-affine LayerNorm and cannot express the runtime time
  add, so the two GEMMs run through ``PerRowInt4LinearResidual`` — the INT4
  linear+residual plugin added for exactly this site.

The SmoothQuant scales come from :func:`compute_evo1_head_sq_scales`, which
replays the build's own calibration capture through the live module with hooks
at every rotation group's input; the fold must match the weights and rotations
this very graph bakes, so there is no scales file to keep in sync.

``action_encoder`` and the output head stay BF16 ONNX, same reasoning as the
INT8 graph. Constructs ONNX only — no ``tensorrt`` import, no ``.so`` load.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnx
import onnx.helper as oh

from . import foldq
from . import omega_rotation as omega
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
    bias_f32,
    to_bytes_f32,
)
from .evo1_head_int8 import _gelu_erf, _i64_init, _layernorm, _linear, _stacked_kv_weights

logger = logging.getLogger(__name__)

__all__ = ["build_evo1_head_plugin_onnx_int4", "compute_evo1_head_sq_scales"]

_DEFAULT_BLOCK_SIZE = 64

# Relative floor on a SmoothQuant per-channel scale — same bound (and the same


def _torch() -> Any:
    import torch

    return torch


def _split_in_proj(sd: Dict[str, Any], prefix: str, dim: int) -> tuple:
    w_in = sd[prefix + "attn.in_proj_weight"]
    b_in = sd[prefix + "attn.in_proj_bias"]
    return (
        (w_in[:dim], b_in[:dim]),
        (w_in[dim : 2 * dim], b_in[dim : 2 * dim]),
        (w_in[2 * dim :], b_in[2 * dim :]),
    )


def _head_sites(expert: Any, block_size: int, fwht: bool) -> tuple:
    """The head's quantized sites: their weights and their rotations.

    One place that knows Evo-1's head layout, so the scale pass and the GPTQ
    Hessian pass cannot drift apart on which tensor feeds which site.
    """
    dim = int(expert.config.embed_dim)
    group_w: Dict[str, Any] = {"encoder": _stacked_kv_weights(expert)}
    enc_perm, enc_R = foldq.site_rotation(group_w["encoder"], block_size, fwht)
    sd = {k: v.detach() for k, v in expert.state_dict().items()}
    rot: Dict[str, tuple] = {}
    for i in range(len(expert.transformer_blocks)):
        p = f"transformer_blocks.{i}."
        (wQ, _), _, _ = _split_in_proj(sd, p, dim)
        for key, wt in (
            (f"block{i}_q", wQ),
            (f"block{i}_o", sd[p + "attn.out_proj.weight"]),
            (f"block{i}_ffn0", sd[p + "ff.0.weight"]),
            (f"block{i}_ffn2", sd[p + "ff.2.weight"]),
        ):
            group_w[key] = wt
            rot[key] = foldq.site_rotation(wt, block_size, fwht)
    return rot, group_w, enc_perm, enc_R


def _o_input(block: Any, q_normed: Any, ctx: Any, num_heads: int, head_dim: int) -> Any:
    """Recompute the post-SDPA out_proj input exactly (fp32, eval, no mask)."""
    torch = _torch()
    w_in = block.attn.in_proj_weight.detach().float()
    b_in = block.attn.in_proj_bias.detach().float()
    dim = w_in.shape[1]
    q = torch.nn.functional.linear(q_normed.float(), w_in[:dim], b_in[:dim])
    k = torch.nn.functional.linear(ctx.float(), w_in[dim : 2 * dim], b_in[dim : 2 * dim])
    v = torch.nn.functional.linear(ctx.float(), w_in[2 * dim :], b_in[2 * dim :])
    b, sq, _ = q.shape
    sk = k.shape[1]
    qh = q.reshape(b, sq, num_heads, head_dim).transpose(1, 2)
    kh = k.reshape(b, sk, num_heads, head_dim).transpose(1, 2)
    vh = v.reshape(b, sk, num_heads, head_dim).transpose(1, 2)
    att = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
    return att.transpose(1, 2).reshape(b, sq, dim)


def _install_head_hooks(expert: Any, rot: Dict[str, tuple], enc_perm: Any, enc_R: Any, sink: Any) -> list:
    """Attach *sink(key, x, perm, R)* at every quantized site of the head."""
    num_heads = int(expert.config.num_heads)
    head_dim = int(expert.config.embed_dim) // num_heads
    handles = []
    for i, block in enumerate(expert.transformer_blocks):
        perm_q, R_q = rot[f"block{i}_q"]
        handles.append(
            block.norm1.register_forward_hook(
                lambda _m, _i, out, k=f"block{i}_q", pm=perm_q, rr=R_q: sink(k, out, pm, rr)
            )
        )
        for key, mod in ((f"block{i}_ffn0", block.ff[0]), (f"block{i}_ffn2", block.ff[2])):
            pm, rr = rot[key]
            handles.append(mod.register_forward_pre_hook(lambda _m, args, k=key, p=pm, r=rr: sink(k, args[0], p, r)))

        def _attn_pre(_m: Any, args: tuple, blk: Any = block, idx: int = i) -> None:
            q_normed, ctx = args[0], args[1]
            if idx == 0:  # context identical for every block — accumulate once
                sink("encoder", ctx, enc_perm, enc_R)
            perm_o, R_o = rot[f"block{idx}_o"]
            sink(f"block{idx}_o", _o_input(blk, q_normed, ctx, num_heads, head_dim), perm_o, R_o)

        handles.append(block.attn.register_forward_pre_hook(_attn_pre))
    return handles


def compute_evo1_head_gptq_hessians(
    action_expert: Any,
    forward_loop: Any,
    sq_scales: Dict[str, Any],
    *,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    fold_order: str = "before",
    fwht: bool = True,
) -> Dict[str, Any]:
    """Second calibration pass: each site's GPTQ Hessian, in the kernel's frame.

    Runs after the scales are known because the Hessian must be built on the
    activation the STORED weight multiplies — rotated, and divided by the
    SmoothQuant vector. Taken on the raw activation it would compensate an error
    that never occurs.

    Returns ``{site: (K, K) float64 CPU}``; feed each to ``gptq_prepare``.
    """
    torch = _torch()
    expert = action_expert.eval()
    rot, _, enc_perm, enc_R = _head_sites(expert, block_size, fwht)
    all_rot = dict(rot)
    all_rot["encoder"] = (enc_perm, enc_R)
    hess, accum = foldq.hessian_accumulator(all_rot, sq_scales, fold_order)
    handles = _install_head_hooks(expert, rot, enc_perm, enc_R, lambda k, x, _p, _r: accum(k, x))
    try:
        with torch.inference_mode():
            forward_loop(expert)
    finally:
        for h in handles:
            h.remove()
    logger.info("    Computed Evo-1 head GPTQ Hessians for %d sites.", len(hess))
    return dict(hess)


def compute_evo1_head_sq_scales(
    action_expert: Any,
    forward_loop: Any,
    *,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    fold_order: str = "after",
    fwht: bool = False,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """Per-channel amax of each W4A4 rotation group in Evo-1's action head.

    Args:
        action_expert: the live head (read-only).
        forward_loop: replays the build's calibration through it.
        block_size: rotation block width; must match the build call.
        fold_order: ``"before"`` measures the raw channel, ``"after"`` the
            rotated one. A scale measured in the wrong frame mis-scales every
            channel with no error anywhere.
        fwht: measure under the fixed butterfly rather than the dense rotation,
            again matching what the build bakes.
        alpha: SmoothQuant migration strength; only meaningful in the raw frame.

    Returns:
        ``{group_key: 1-D per-channel scale (CPU, floored)}``.
    """
    torch = _torch()
    expert = action_expert.eval()
    rot, group_w, enc_perm, enc_R = _head_sites(expert, block_size, fwht)

    amax: Dict[str, Any] = {}

    def accum_rotated(key: str, x: Any, perm: Any, R: Any) -> None:
        xf = x.detach().float()
        # The scale lands on whichever axis fold_order names, so it has to be
        # measured there: "before" is the raw channel, "after" the rotated one.
        src = xf if fold_order == "before" else omega.apply_rotation(xf, perm, R, block_size)
        v = src.abs().reshape(-1, src.shape[-1]).amax(dim=0)
        amax[key] = v if key not in amax else torch.maximum(amax[key], v)

    handles = _install_head_hooks(expert, rot, enc_perm, enc_R, accum_rotated)
    try:
        with torch.inference_mode():
            forward_loop(expert)
    finally:
        for h in handles:
            h.remove()
    logger.info(
        "    Computed Evo-1 head W4A4 SmoothQuant scales for %d rotation groups (alpha=%.2f).", len(amax), alpha
    )
    return foldq.finalize_scales(amax, weights=group_w, alpha=alpha)


def build_evo1_head_plugin_onnx_int4(
    action_expert: Any,
    out_path: "str | Path",
    *,
    sq_scales: Optional[Dict[str, Any]] = None,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    fold_order: str = "after",
    fwht: bool = False,
    gptq: "dict | None" = None,
) -> Path:
    """Write the W4A4 plugin graph for *action_expert* to *out_path*.

    Args:
        action_expert: Live ``Evo1FlowMatchingActionExpert`` (read-only).
        out_path: Destination ``.onnx``.
        sq_scales: Per-group rotated-activation amax from
            :func:`compute_evo1_head_sq_scales` -> the calibrated
            ``w4a4_sr`` graph, which is what every build path
            produces. ``None`` builds the uncalibrated dynamic variant and is
            test-only.
        block_size: FoldQuant rotation block width (896 and 3584 both divide 64).
    """
    torch = _torch()
    sd = {k: v.detach() for k, v in action_expert.state_dict().items()}
    cfg = action_expert.config
    dim = int(cfg.embed_dim)
    horizon = int(cfg.horizon)
    num_heads = int(cfg.num_heads)
    head_dim = dim // num_heads
    ff_inner = dim * 4
    per_action = int(cfg.per_action_dim)
    total_action = int(cfg.total_action_dim)
    num_blocks = len(action_expert.transformer_blocks)
    ctx_len = 1025  # 1024 prompt tokens + 1 state token; fully static graph.
    is_sq = sq_scales is not None

    def _g(key: str) -> Any:
        """This site's GPTQ factors, or None for round-to-nearest."""
        return gptq.get(key) if gptq is not None else None

    def _rot_attr() -> Dict[str, Any]:
        """The block width a butterfly node rotates in; absent on the dense arm."""
        return {"rot_block_size": int(block_size)} if fwht else {}

    def _pre_vec(field: str, s_ch: Any) -> Dict[str, Any]:
        """The raw-frame vector a butterfly site ships instead of a folded matrix."""
        if not fwht or s_ch is None:
            return {}
        return {field: to_bytes_f32(s_ch.detach().float().cpu().numpy())}

    def _pack(weight: Any, perm: Any, R: Any, s_ch: Any, key: str = "") -> tuple:
        """Fold + INT4-pack one macro-plugin site -> ``(i4, scale, R_use)``.

        The same shared helper the DiT's macro sites use. In butterfly mode the
        caller ships an empty ``rotation_*`` blob plus ``rot_block_size`` and the
        raw ``act_scale_pre*`` vector, and ignores ``R_use`` — the kernel rebuilds
        the fixed Hadamard per token.
        """
        return foldq.fold_macro_site(weight, perm, R, block_size, s_ch, fold_order=fold_order, gptq=_g(key))

    def _per_row_site(weight: Any, s_ch: Any, key: str = "") -> tuple:
        """Fold + pack one SINGLE-site ``PerRowInt4LinearResidual`` -> ``(i4, scale, attrs)``.

        Unlike the macro sites, this node has exactly one rotation slot, so the
        attribute dict foldq returns can be splatted straight in — the same shape
        the Gemma and SmolVLA experts use on this very plugin. The butterfly arm
        fills ``rot_block_size`` + ``act_scale_pre``; the dense arm fills
        ``perm`` + ``rotation`` with the scale already absorbed into the matrix.
        """
        if fwht:
            return foldq.fold_site(
                weight,
                bits=4,
                block_size=block_size,
                s_ch=s_ch,
                fold_order=fold_order,
                fwht=True,
                gptq=_g(key),
            )
        perm, R = foldq.site_rotation(weight, block_size, False)
        i4, sc, r_use = foldq.fold_macro_site(weight, perm, R, block_size, s_ch, fold_order=fold_order, gptq=_g(key))
        return (
            i4,
            sc,
            {
                "perm": omega.to_bytes_i32(perm.detach().cpu().numpy()),
                "rotation": to_bytes_f32(r_use.detach().float().cpu().numpy()),
            },
        )

    nodes: list = []
    inits: list = []

    x_in = oh.make_tensor_value_info("action_seq", onnx.TensorProto.BFLOAT16, [1, horizon, per_action])
    c_in = oh.make_tensor_value_info("context_tokens", onnx.TensorProto.BFLOAT16, [1, ctx_len, dim])
    t_in = oh.make_tensor_value_info("time_emb", onnx.TensorProto.BFLOAT16, [1, dim])
    y_out = oh.make_tensor_value_info("velocity", onnx.TensorProto.BFLOAT16, [1, total_action])

    # ===== action_encoder (BF16, identical to the INT8 graph) =====
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
    from .dit_common import bf16_initializer

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

    inits.append(_i64_init("time_axes", [1]))
    nodes.append(oh.make_node("Unsqueeze", ["time_emb", "time_axes"], ["time_b"]))

    # ===== EncoderPreQuantInt4: shared rotation, one INT4 quant of the context =====
    enc_perm, enc_R = foldq.site_rotation(_stacked_kv_weights(action_expert), block_size, fwht)
    s_enc = sq_scales["encoder"] if sq_scales is not None else None
    enc_R_use = foldq.fold_rotation(enc_R, enc_perm, s_enc, fold_order)
    nodes.append(
        oh.make_node(
            "EncoderPreQuantInt4",
            ["context_tokens"],
            ["encoder_i4", "encoder_scale"],
            name="encoder_prequant_int4",
            domain=_PLUGIN_DOMAIN,
            plugin_namespace=_PLUGIN_NAMESPACE,
            plugin_version=_PLUGIN_VERSION,
            perm_enc=omega.to_bytes_i32(enc_perm.detach().cpu().numpy()),
            rotation_enc=(b"" if fwht else to_bytes_f32(enc_R_use.detach().float().cpu().numpy())),
            **_rot_attr(),
            **_pre_vec("act_scale_pre_enc", s_enc),
            K_enc=int(dim),
            block_size=int(block_size),
        )
    )

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

        (wQ, bQ), (wK, bK), (wV, bV) = _split_in_proj(sd, p, dim)
        wO, bO = sd[p + "attn.out_proj.weight"], sd[p + "attn.out_proj.bias"]

        permQ, RQ = foldq.site_rotation(wQ, block_size, fwht)
        s_q = sq_scales[f"{b}_q"] if sq_scales is not None else None
        q_i4, q_sc, RQ_use = _pack(wQ, permQ, RQ, s_q, f"{b}_q")

        permO, RO = foldq.site_rotation(wO, block_size, fwht)
        s_o = sq_scales[f"{b}_o"] if sq_scales is not None else None
        o_i4, o_sc, RO_use = _pack(wO, permO, RO, s_o, f"{b}_o")

        # KV packed under the SHARED encoder rotation (+ shared encoder SQ scale).
        wKV = torch.cat([wK, wV], dim=0)
        kv_i4, kv_sc, _ = _pack(wKV, enc_perm, enc_R, s_enc, "encoder")

        # Butterfly sites ship the raw per-channel vector instead of a baked
        # matrix; the kernel rebuilds the Hadamard per token.
        bf_q = _pre_vec("act_scale_pre_in", s_q)
        bf_o = _pre_vec("act_scale_pre_o", s_o)

        # 5 inputs — no attention mask: Evo-1 attends the whole context (trained
        # behaviour); the plugin treats the mask input as optional.
        nodes.append(
            oh.make_node(
                "FusedCrossAttnFullInt4",
                [cur_x, f"{b}_scale", f"{b}_shift", "encoder_i4", "encoder_scale"],
                [f"{b}_post_attn"],
                name=f"{b}_crossattn_int4",
                domain=_PLUGIN_DOMAIN,
                plugin_namespace=_PLUGIN_NAMESPACE,
                plugin_version=_PLUGIN_VERSION,
                **_rot_attr(),
                inner_dim=int(dim),
                K=int(dim),
                K_enc=int(dim),
                num_heads=num_heads,
                head_dim=head_dim,
                block_size=int(block_size),
                eps=float(_EPS),
                weight_q_i4=q_i4,
                weight_q_scale=q_sc,
                bias_q=to_bytes_f32(bias_f32(bQ)),
                weight_kv_i4=kv_i4,
                weight_kv_scale=kv_sc,
                bias_kv=to_bytes_f32(np.concatenate([bias_f32(bK), bias_f32(bV)])),
                weight_o_i4=o_i4,
                weight_o_scale=o_sc,
                bias_o=to_bytes_f32(bias_f32(bO)),
                perm_q=omega.to_bytes_i32(permQ.detach().cpu().numpy()),
                rotation_q=(b"" if fwht else omega.to_bytes_bf16(RQ_use)),
                **bf_q,
                perm_o=omega.to_bytes_i32(permO.detach().cpu().numpy()),
                rotation_o=(b"" if fwht else omega.to_bytes_bf16(RO_use)),
                **bf_o,
                emit_kv=0,
            )
        )
        h = f"{b}_post_attn"

        # --- FFN: LN2 -> +time -> INT4 GEMM -> GELU -> INT4 GEMM + residual ---
        _layernorm(nodes, inits, f"{b}_ln2", h, sd[p + "norm2.weight"], sd[p + "norm2.bias"], f"{b}_x2")
        nodes.append(oh.make_node("Add", [f"{b}_x2", "time_b"], [f"{b}_x2t"]))

        w0 = sd[p + "ff.0.weight"]
        s0 = sq_scales[f"{b}_ffn0"] if sq_scales is not None else None
        p0_i4, p0_sc, ff0_attrs = _per_row_site(w0, s0, f"{b}_ffn0")
        nodes.append(
            oh.make_node(
                "PerRowInt4LinearResidual",
                [f"{b}_x2t", "ff_zero_res"],
                [f"{b}_ffh_raw"],
                name=f"{b}_ff0_int4",
                domain=_PLUGIN_DOMAIN,
                plugin_namespace=_PLUGIN_NAMESPACE,
                plugin_version=_PLUGIN_VERSION,
                N=int(ff_inner),
                K=int(dim),
                block_size=int(block_size),
                weight_i4=p0_i4,
                weight_scale=p0_sc,
                **ff0_attrs,
            )
        )
        inits.append(bf16_initializer(f"{b}_ff0_b", sd[p + "ff.0.bias"]))
        nodes.append(oh.make_node("Add", [f"{b}_ffh_raw", f"{b}_ff0_b"], [f"{b}_ffh"]))
        _gelu_erf(nodes, inits, f"{b}_gelu", f"{b}_ffh", f"{b}_ffg")

        w2 = sd[p + "ff.2.weight"]
        s2 = sq_scales[f"{b}_ffn2"] if sq_scales is not None else None
        p2_i4, p2_sc, ff2_attrs = _per_row_site(w2, s2, f"{b}_ffn2")
        nodes.append(
            oh.make_node(
                "PerRowInt4LinearResidual",
                [f"{b}_ffg", h],
                [f"{b}_out_raw"],
                name=f"{b}_ff2_int4",
                domain=_PLUGIN_DOMAIN,
                plugin_namespace=_PLUGIN_NAMESPACE,
                plugin_version=_PLUGIN_VERSION,
                N=int(dim),
                K=int(ff_inner),
                block_size=int(block_size),
                weight_i4=p2_i4,
                weight_scale=p2_sc,
                **ff2_attrs,
            )
        )
        inits.append(bf16_initializer(f"{b}_ff2_b", sd[p + "ff.2.bias"]))
        nodes.append(oh.make_node("Add", [f"{b}_out_raw", f"{b}_ff2_b"], [f"{b}_out"]))
        cur_x = f"{b}_out"

    # ===== output head (BF16, identical to the INT8 graph) =====
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

    graph = oh.make_graph(nodes, "evo1_action_head_foldquant_w4a4", [x_in, c_in, t_in], [y_out], initializer=inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 19), oh.make_opsetid(_PLUGIN_DOMAIN, 1)])
    model.ir_version = 9
    from .onnx_io import save_plugin_onnx

    path = Path(out_path)
    save_plugin_onnx(model, path)
    c = Counter(n.op_type for n in nodes)
    logger.info(
        "Built Evo-1 action-head W4A4 (%s) plugin graph: %d blocks, %d nodes, ops=%s -> %s",
        "sq" if is_sq else "dynamic",
        num_blocks,
        len(nodes),
        dict(sorted(c.items(), key=lambda x: -x[1])),
        path,
    )
    return path
