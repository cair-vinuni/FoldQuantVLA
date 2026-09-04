# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""INT8 per-row plugin graph for SmolVLA's dual-stream denoise step.

Replaces the exported ``expert.onnx`` (one flow-matching denoise step) with a
graph whose transformer GEMMs run through the ``gr00t::v1`` INT8 per-row
plugins — same I/O contract as the graph it replaces
(``x_t [1,50,32]``, ``timestep [1]``, ``prefix_pad_masks [1,S]`` BOOL,
``kv_stack [L,2,1,S,H_kv,D]`` -> ``velocity [1,50,32]`` FP32).

SmolVLA's expert is not a DiT and none of the fused attention macro plugins
fit it (RMSNorm not AdaLN, GQA 15/5, no biases, RoPE, per-layer cached KV from
two separate tensors), so the graph uses the same decomposition as the LLM
surgery: ``FusedRmsNormLinearInt8`` for pre-norm + merged projections,
``PerRowInt8LinearResidual`` for o_proj/down_proj (+ the odd layers'
cache-side k/v projections, with a dynamic zeros residual), and plain BF16
ONNX for RoPE / GQA repeat / SDPA / SwiGLU glue.

Two alternating layer bodies, exactly as the runtime
(``self_attn_every_n_layers = 2``):

- **even** (``forward_attn_layer``): expert q/k/v from the suffix; fresh K/V
  RoPE'd at ABSOLUTE positions (prefix_len + 0..chunk-1) and concatenated
  after the cached prefix K/V (``kv_stack[i]``, consumed raw); full attention
  over ``[S+chunk]`` keys under (prefix-pad ++ static causal-suffix) masking.
- **odd** (``forward_cross_attn_layer``): q only from the suffix, RoPE'd at
  REBASED positions (0..chunk-1, a static table); K/V re-projected from the
  flattened cache through the layer's 320->320 projections; attention over
  the ``S`` prefix keys under the prefix-pad mask.

What stays BF16/FP32 ONNX: the suffix embedding (action_in_proj, the
period-parametrised sinusoidal time embedding with constants baked from the
exact float64 tables, action_time_mlp_in/out + SiLU), the final expert
RMSNorm, and the float32 ``action_out_proj`` — tiny, precision-critical
matmuls bracketing the 16 blocks (the runtime itself casts the suffix to
float32 before action_out_proj).

Constructs ONNX only: no ``tensorrt`` import, no ``.so`` load.
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import onnx
import onnx.helper as oh

from .llm import (
    _bf16_initializer,
    _i64_init,
    _torch,
)

logger = logging.getLogger(__name__)

#: Diagnostic: emit BF16 MatMul instead of the quantized linear, leaving the
#: rest of the expert graph untouched. Used to localise a defect between the
#: scaffolding and the quantized GEMM; never set in a shipped build.
_FLOAT_LINEAR = os.environ.get("FOLDQUANT_SMOLVLA_FLOAT_LINEAR") == "1"

__all__ = ["build_smolvla_expert_plugin_onnx"]

# The plugin contract lives in dit_common ("must match the compiled plugin .so.
# Do not change") — imported, not redeclared, so the four emitters cannot drift.
# PLUGIN_VERSION is a STRING: TensorRT reads plugin_version as a string
# attribute, and an INT attr fails creator lookup ("Plugin not found") at parse.
from . import foldq  # noqa: E402  (contract imports kept next to their use)
from . import omega_rotation as omega  # noqa: E402
from .dit_common import PLUGIN_DOMAIN as _PLUGIN_DOMAIN  # noqa: E402
from .dit_common import PLUGIN_NAMESPACE as _PLUGIN_NAMESPACE  # noqa: E402
from .dit_common import PLUGIN_VERSION as _PLUGIN_VERSION  # noqa: E402


def _f32_initializer(name: str, tensor: Any) -> onnx.TensorProto:
    arr = np.ascontiguousarray(tensor.detach().float().cpu().numpy(), dtype=np.float32)
    return oh.make_tensor(name, onnx.TensorProto.FLOAT, list(arr.shape), arr.tobytes(), raw=True)


def build_smolvla_expert_plugin_onnx(
    action_expert: Any,
    out_path: "str | Path",
    *,
    rope_max_seq_len: int = 4096,
    int4: bool = False,
    sq_scales: "dict | None" = None,
    block_size: int = 64,
    fold_order: str = "after",
    fwht: bool = False,
    gptq: "dict | None" = None,
) -> Path:
    """Write the INT8 per-row denoise-step plugin graph for *action_expert*.

    Args:
        action_expert: Live ``SmolVLAFlowMatchingActionExpert`` (read-only;
            weights are snapshotted into initializers).
        out_path: Destination ``.onnx``.
        rope_max_seq_len: Size of the baked absolute-position RoPE table
            (prefix_len + chunk must stay below it).
        int4: Emit the FoldQuant W4A4 variant (``w4a4_sr``): every
            projection GEMM through ``PerRowInt4LinearResidual`` (rotated
            per-row INT4), the pre-norms decomposed to BF16 ONNX RMSNorm (no
            fused RMSNorm+INT4 plugin exists). Requires *sq_scales*.
        sq_scales: Per-site rotated-activation amax from
            :func:`compute_smolvla_expert_sq_scales`.
        block_size: FoldQuant rotation block width (int4 only).
    """
    if int4 and sq_scales is None:
        raise ValueError("int4=True requires sq_scales (the W4A4 scheme is SmoothQuant-folded).")
    torch = _torch()
    sd = {k: v.detach() for k, v in action_expert.state_dict().items()}
    cfg = action_expert.config
    backbone = action_expert.backbone

    chunk = int(cfg.chunk_size)
    action_dim = int(cfg.max_action_dim)
    hidden = int(action_expert.expert_hidden_size)  # 720
    num_layers = int(backbone.num_vlm_layers)  # 16 expert layers, one per VLM layer
    ecfg = action_expert.lm_expert.config
    h = int(ecfg.num_attention_heads)
    hkv = int(ecfg.num_key_value_heads)
    d = int(getattr(ecfg, "head_dim", hidden // h))
    q_dim = h * d
    kv_dim = hkv * d
    kv_groups = h // hkv
    ff = int(sd["lm_expert.layers.0.mlp.gate_proj.weight"].shape[0])
    eps = float(getattr(ecfg, "rms_norm_eps", 1e-5))
    every_n = int(action_expert.self_attn_every_n_layers)
    theta = 10000.0  # apply_rope hardcodes max_wavelength=10_000

    inits: list = []
    nodes: list = []

    x_in = oh.make_tensor_value_info("x_t", onnx.TensorProto.FLOAT, [1, chunk, action_dim])
    t_in = oh.make_tensor_value_info("timestep", onnx.TensorProto.FLOAT, [1])
    p_in = oh.make_tensor_value_info("prefix_pad_masks", onnx.TensorProto.BOOL, [1, "prefix_len"])
    kv_in = oh.make_tensor_value_info("kv_stack", onnx.TensorProto.BFLOAT16, [num_layers, 2, 1, "prefix_len", hkv, d])
    y_out = oh.make_tensor_value_info("velocity", onnx.TensorProto.FLOAT, [1, chunk, action_dim])

    # ===== suffix embedding (BF16; time constants baked from exact fp64) =====
    nodes.append(oh.make_node("Cast", ["x_t"], ["x_t_bf16"], to=onnx.TensorProto.BFLOAT16))

    def _linear(name: str, x: str, w_key: str, b_key: "str | None", out: str) -> None:
        inits.append(_bf16_initializer(f"{name}_w", sd[w_key].t().contiguous()))
        nodes.append(oh.make_node("MatMul", [x, f"{name}_w"], [f"{name}_mm"]))
        if b_key is not None and b_key in sd:
            inits.append(_bf16_initializer(f"{name}_b", sd[b_key]))
            nodes.append(oh.make_node("Add", [f"{name}_mm", f"{name}_b"], [out]))
        else:
            nodes.append(oh.make_node("Identity", [f"{name}_mm"], [out]))

    _linear("act_in", "x_t_bf16", "action_in_proj.weight", "action_in_proj.bias", "action_emb")

    # sinusoidal_time_embedding_period: constants computed here in float64,
    # exactly as the runtime does; only the timestep multiply happens at
    # runtime (fp32 — the argument magnitude keeps the error far below the
    # bf16 quantization of the embedding itself).
    half = hidden // 2
    fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float64)
    period = float(cfg.min_period) * (float(cfg.max_period) / float(cfg.min_period)) ** fraction
    scaling = (1.0 / period * 2 * math.pi).to(torch.float64)
    inits.append(_f32_initializer("time_scaling", scaling.reshape(1, half)))
    inits.append(_i64_init("time_unsq_1", [1]))
    nodes.append(oh.make_node("Unsqueeze", ["timestep", "time_unsq_1"], ["t_2d"]))  # [1,1]
    nodes.append(oh.make_node("Mul", ["t_2d", "time_scaling"], ["sin_input"]))  # [1,half]
    nodes.append(oh.make_node("Sin", ["sin_input"], ["t_sin"]))
    nodes.append(oh.make_node("Cos", ["sin_input"], ["t_cos"]))
    nodes.append(oh.make_node("Concat", ["t_sin", "t_cos"], ["time_emb_f32"], axis=1))  # [1,hidden]
    nodes.append(oh.make_node("Cast", ["time_emb_f32"], ["time_emb_bf16"], to=onnx.TensorProto.BFLOAT16))
    nodes.append(oh.make_node("Unsqueeze", ["time_emb_bf16", "time_unsq_1"], ["time_emb_3d"]))  # [1,1,hidden]
    inits.append(
        oh.make_tensor(
            "time_expand_shape",
            onnx.TensorProto.INT64,
            [3],
            np.array([1, chunk, hidden], dtype=np.int64).tobytes(),
            raw=True,
        )
    )
    nodes.append(oh.make_node("Expand", ["time_emb_3d", "time_expand_shape"], ["time_emb_x"]))
    nodes.append(oh.make_node("Concat", ["action_emb", "time_emb_x"], ["action_time_cat"], axis=2))
    _linear("atm_in", "action_time_cat", "action_time_mlp_in.weight", "action_time_mlp_in.bias", "atm_h")
    nodes.append(oh.make_node("Sigmoid", ["atm_h"], ["atm_sig"]))
    nodes.append(oh.make_node("Mul", ["atm_h", "atm_sig"], ["atm_silu"]))
    _linear("atm_out", "atm_silu", "action_time_mlp_out.weight", "action_time_mlp_out.bias", "suffix_embs")

    # ===== positions and masks (built once, shared by all layers) =====
    # prefix_len (scalar [1] int64) from Shape(prefix_pad_masks)[1].
    nodes.append(oh.make_node("Shape", ["prefix_pad_masks"], ["_pp_shape"]))
    inits.append(_i64_init("_idx_1", [1]))
    nodes.append(oh.make_node("Gather", ["_pp_shape", "_idx_1"], ["_S"], axis=0))

    # Absolute suffix positions = sum(prefix_pad) + [0..chunk-1].
    nodes.append(oh.make_node("Cast", ["prefix_pad_masks"], ["_pp_i64"], to=onnx.TensorProto.INT64))
    inits.append(_i64_init("_sum_axes_1", [1]))
    nodes.append(oh.make_node("ReduceSum", ["_pp_i64", "_sum_axes_1"], ["_offset"], keepdims=0))  # [1]
    inits.append(
        oh.make_tensor(
            "suffix_arange", onnx.TensorProto.INT64, [1, chunk], np.arange(chunk, dtype=np.int64).tobytes(), raw=True
        )
    )
    inits.append(_i64_init("_off_unsq", [1]))
    nodes.append(oh.make_node("Unsqueeze", ["_offset", "_off_unsq"], ["_offset_2d"]))  # [1,1]
    nodes.append(oh.make_node("Add", ["_offset_2d", "suffix_arange"], ["abs_positions"]))  # [1,chunk]

    # RoPE tables baked at rope_max_seq_len (theta=10000, half-split layout).
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    pos = torch.arange(rope_max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    inits.append(_bf16_initializer("rope_cos_table", emb.cos().to(torch.bfloat16)))
    inits.append(_bf16_initializer("rope_sin_table", emb.sin().to(torch.bfloat16)))
    # Absolute (even layers): gather -> [1, chunk, d] -> [1, 1, chunk, d].
    nodes.append(oh.make_node("Gather", ["rope_cos_table", "abs_positions"], ["cos_abs_g"], axis=0))
    nodes.append(oh.make_node("Gather", ["rope_sin_table", "abs_positions"], ["sin_abs_g"], axis=0))
    nodes.append(oh.make_node("Unsqueeze", ["cos_abs_g", "_idx_1"], ["cos_abs"]))
    nodes.append(oh.make_node("Unsqueeze", ["sin_abs_g", "_idx_1"], ["sin_abs"]))
    # Rebased (odd layers): positions 0..chunk-1 — a static table slice.
    inits.append(_bf16_initializer("cos_reb", emb.cos()[:chunk].to(torch.bfloat16).view(1, 1, chunk, d)))
    inits.append(_bf16_initializer("sin_reb", emb.sin()[:chunk].to(torch.bfloat16).view(1, 1, chunk, d)))

    # Additive masks. Same margin as the LLM surgery's causal mask.
    mask_val = float(torch.finfo(torch.bfloat16).min * 0.5)
    inits.append(_bf16_initializer("m_one", torch.tensor(1.0, dtype=torch.bfloat16)))
    inits.append(_bf16_initializer("m_scale", torch.tensor(mask_val, dtype=torch.bfloat16)))
    nodes.append(oh.make_node("Cast", ["prefix_pad_masks"], ["_pp_bf16"], to=onnx.TensorProto.BFLOAT16))
    nodes.append(oh.make_node("Sub", ["m_one", "_pp_bf16"], ["_pp_inv"]))  # 1 where padded
    nodes.append(oh.make_node("Mul", ["_pp_inv", "m_scale"], ["_pp_bias_2d"]))  # [1,S]
    inits.append(_i64_init("_bias_axes", [1, 2]))
    nodes.append(oh.make_node("Unsqueeze", ["_pp_bias_2d", "_bias_axes"], ["prefix_bias"]))  # [1,1,1,S]
    # Even layers: full [1,1,chunk,S+chunk] = Expand(prefix_bias) ++ static causal suffix.
    inits.append(_i64_init("_exp_const_11c", [1, 1, chunk]))
    nodes.append(oh.make_node("Concat", ["_exp_const_11c", "_S"], ["_exp_shape"], axis=0))
    nodes.append(oh.make_node("Expand", ["prefix_bias", "_exp_shape"], ["prefix_bias_q"]))  # [1,1,chunk,S]
    causal = torch.triu(torch.full((chunk, chunk), mask_val), diagonal=1).to(torch.bfloat16)
    inits.append(_bf16_initializer("suffix_causal", causal.view(1, 1, chunk, chunk)))
    nodes.append(oh.make_node("Concat", ["prefix_bias_q", "suffix_causal"], ["full_mask"], axis=3))

    inits.append(_bf16_initializer("attn_scale", torch.tensor(1.0 / math.sqrt(d), dtype=torch.bfloat16)))
    inits.append(_bf16_initializer("eps_bf16", torch.tensor(eps, dtype=torch.bfloat16)))
    inits.append(_bf16_initializer("two_bf16", torch.tensor(2.0, dtype=torch.bfloat16)))
    inits.append(_i64_init("_axes_2", [2]))
    inits.append(_i64_init("split_half_d", [d // 2, d // 2]))

    def _rmsnorm(name: str, x: str, gamma_key: str, out: str) -> None:
        inits.append(_bf16_initializer(f"{name}_g", sd[gamma_key]))
        nodes.append(oh.make_node("Pow", [x, "two_bf16"], [f"{name}_sq"]))
        nodes.append(oh.make_node("ReduceMean", [f"{name}_sq"], [f"{name}_mean"], axes=[-1], keepdims=1))
        nodes.append(oh.make_node("Add", [f"{name}_mean", "eps_bf16"], [f"{name}_var"]))
        nodes.append(oh.make_node("Sqrt", [f"{name}_var"], [f"{name}_std"]))
        nodes.append(oh.make_node("Div", [x, f"{name}_std"], [f"{name}_normed"]))
        nodes.append(oh.make_node("Mul", [f"{name}_normed", f"{name}_g"], [out]))

    def _rope(b: str, qk_in: str, cos_name: str, sin_name: str, qk_out: str) -> None:
        nodes.append(oh.make_node("Split", [qk_in, "split_half_d"], [f"{qk_in}_a", f"{qk_in}_b"], axis=-1))
        nodes.append(oh.make_node("Neg", [f"{qk_in}_b"], [f"{qk_in}_bn"]))
        nodes.append(oh.make_node("Concat", [f"{qk_in}_bn", f"{qk_in}_a"], [f"{qk_in}_rot"], axis=-1))
        nodes.append(oh.make_node("Mul", [qk_in, cos_name], [f"{qk_in}_c"]))
        nodes.append(oh.make_node("Mul", [f"{qk_in}_rot", sin_name], [f"{qk_in}_s"]))
        nodes.append(oh.make_node("Add", [f"{qk_in}_c", f"{qk_in}_s"], [qk_out]))

    def _repeat_kv(b: str, in_name: str, out_name: str) -> None:
        # (1, HKV, S*, D) -> (1, H, S*, D)
        nodes.append(oh.make_node("Unsqueeze", [in_name, "_axes_2"], [f"{out_name}_u"]))
        inits.append(_i64_init(f"{out_name}_reps", [1, 1, kv_groups, 1, 1]))
        nodes.append(oh.make_node("Tile", [f"{out_name}_u", f"{out_name}_reps"], [f"{out_name}_t"]))
        inits.append(_i64_init(f"{out_name}_shape", [1, h, -1, d]))
        nodes.append(oh.make_node("Reshape", [f"{out_name}_t", f"{out_name}_shape"], [out_name], allowzero=1))

    def _fused_norm_linear(b: str, x: str, gamma_key: str, w: Any, n_out: int, k_in: int, out: str) -> None:
        if _FLOAT_LINEAR:
            _rmsnorm(f"{b}_n", x, gamma_key, f"{b}_normed")
            _float_linear(b, f"{b}_normed", None, w, out)
            return
        if int4 or sq_scales is not None:
            # No fused RMSNorm+INT4 plugin exists, and a FOLDED INT8 site cannot
            # use the fused INT8 one either: this expert's 480-wide projections
            # have to be zero-padded to the rotation block, and a plugin that
            # normalises over K internally has nowhere to put a Pad node.
            # Un-fusing the norm costs one kernel and buys both widths the SAME
            # fold — which is the point. The unfolded INT8 arm below keeps the
            # fused plugin, since with no scale there is nothing to pad for.
            _rmsnorm(f"{b}_n", x, gamma_key, f"{b}_normed")
            zeros = f"{b}_zres"
            inits.append(_bf16_initializer(zeros, _torch().zeros(1, chunk, n_out)))
            _linear_residual(b, f"{b}_normed", zeros, w, n_out, k_in, out)
            return
        w_i8_b, s_b, extra8 = _fold_int8(b, w, k_in)
        nodes.append(
            oh.make_node(
                "FusedRmsNormLinearInt8",
                [x],
                [out],
                name=f"{b}_frl",
                domain=_PLUGIN_DOMAIN,
                plugin_namespace=_PLUGIN_NAMESPACE,
                plugin_version=_PLUGIN_VERSION,
                N=int(n_out),
                K=int(k_in),
                eps=float(eps),
                gamma=sd[gamma_key].float().cpu().numpy().tobytes(),
                weight_i8=w_i8_b,
                weight_scale=s_b,
                **extra8,
            )
        )

    def _int8_block(k_in: int) -> int:
        """Largest power-of-two rotation block <= block_size that divides ``k_in``.

        The INT4 path pads the input axis up to a multiple of the block, but the
        fused-norm INT8 plugin normalises over K internally and cannot be handed a
        padded activation. Narrowing the block instead keeps the fold exact — the
        kernel accepts any power-of-two block dividing K — at the cost of mixing
        fewer channels (SmolVLA's 480-wide sites drop from 64 to 32).
        """
        bs = int(block_size)
        while bs > 1 and k_in % bs != 0:
            bs //= 2
        return bs

    def _fold_int8(b: str, w: Any, k_in: int) -> tuple:
        """Folded INT8 weight plus the node attributes that describe the fold.

        The action module has never had a folded INT8 arm — every folded INT8 key
        was LLM-only — so this is the same fold its INT4 twin applies, at 8 bit.
        Callers pad the input axis first, exactly as the INT4 path does, so both
        widths fold at the same block. That is why a folded site un-fuses the
        norm: the fused INT8 plugin normalises over K internally and has nowhere
        to put the Pad node.
        """
        return foldq.fold_site(
            w,
            bits=8,
            block_size=block_size,
            s_ch=sq_scales.get(b) if sq_scales is not None else None,
            fold_order=fold_order,
            fwht=True,
        )

    def _pad_to_block(tag: str, x: str, w: Any, k_in: int) -> tuple:
        """Zero-pad the input axis up to a multiple of the block width.

        SmolVLA's expert is 720 wide and 720 = 2^4·3^2·5, which blocks the INT4
        path twice: 16 is the only power-of-two block that divides it, and the
        INT4 GEMM loads 128 bits at a time (alignment 32 elements) so K=720 is
        refused outright by can_implement. Padding to 768 clears both, and it is
        exact — the weight's extra columns are zero, so W_pad·x_pad = W·x, and
        the rotation stays orthonormal on the padded space.

        Returns ``(x_name, weight, k)`` already padded, or the inputs unchanged
        when the width already fits.
        """
        k_pad = ((k_in + block_size - 1) // block_size) * block_size
        if k_pad == k_in:
            return x, w, k_in
        pads = np.array([0, 0, 0, 0, 0, k_pad - k_in], dtype=np.int64)
        inits.append(oh.make_tensor(f"{tag}_pads", onnx.TensorProto.INT64, [6], pads.tobytes(), raw=True))
        nodes.append(oh.make_node("Pad", [x, f"{tag}_pads"], [f"{tag}_xpad"], mode="constant"))
        from . import omega_rotation as omega

        return f"{tag}_xpad", omega.pad_in_dim(w, block_size), k_pad

    def _float_linear(b: str, x: str, residual: "str | None", w: Any, out: str) -> None:
        """BF16 MatMul standing in for a quantized linear (FOLDQUANT_SMOLVLA_FLOAT_LINEAR).

        Diagnostic only. Keeps every surrounding node identical so a build with
        this on isolates the expert's graph scaffolding from its quantized GEMM.
        """
        inits.append(_bf16_initializer(f"{b}_wT", w.float().t().contiguous()))
        mm = f"{b}_mm" if residual is not None else out
        nodes.append(oh.make_node("MatMul", [x, f"{b}_wT"], [mm]))
        if residual is not None:
            nodes.append(oh.make_node("Add", [mm, residual], [out]))

    def _linear_residual(b: str, x: str, residual: str, w: Any, n_out: int, k_in: int, out: str) -> None:
        if _FLOAT_LINEAR:
            _float_linear(b, x, residual, w, out)
            return
        if int4:
            x, w, k_in = _pad_to_block(out, x, w, k_in)
            # One fold, decided in one place: foldq picks the rotation, folds the
            # SmoothQuant scale onto the axis fold_order names, packs for the bit
            # width and returns the node attributes that describe all of it.
            s_ch = sq_scales[b] if sq_scales is not None else None
            packed_b, scale_b, fold_attrs = foldq.fold_site(
                w,
                bits=4,
                block_size=block_size,
                s_ch=s_ch,
                fold_order=fold_order,
                fwht=fwht,
                gptq=(gptq or {}).get(b),
            )
            nodes.append(
                oh.make_node(
                    "PerRowInt4LinearResidual",
                    [x, residual],
                    [out],
                    name=f"{b}_plr4",
                    domain=_PLUGIN_DOMAIN,
                    plugin_namespace=_PLUGIN_NAMESPACE,
                    plugin_version=_PLUGIN_VERSION,
                    N=int(n_out),
                    K=int(k_in),
                    block_size=int(block_size),
                    weight_i4=packed_b,
                    weight_scale=scale_b,
                    **fold_attrs,
                )
            )
            return
        # Same padding as the INT4 branch — one fold, one width policy.
        if sq_scales is not None:
            x, w, k_in = _pad_to_block(out, x, w, k_in)
        w_i8_b, s_b, extra8 = _fold_int8(b, w, k_in)
        nodes.append(
            oh.make_node(
                "PerRowInt8LinearResidual",
                [x, residual],
                [out],
                name=f"{b}_plr",
                domain=_PLUGIN_DOMAIN,
                plugin_namespace=_PLUGIN_NAMESPACE,
                plugin_version=_PLUGIN_VERSION,
                N=int(n_out),
                K=int(k_in),
                weight_i8=w_i8_b,
                weight_scale=s_b,
                **extra8,
            )
        )

    # Dynamic zeros residual for the odd layers' cache-side k/v projections
    # ([1, S, kv_dim] — S is runtime, so ConstantOfShape rather than a baked
    # initializer; the static-shape trick Evo-1 uses does not survive it).
    inits.append(_i64_init("_zeros_const_1", [1]))
    inits.append(_i64_init("_zeros_const_kv", [kv_dim]))
    nodes.append(oh.make_node("Concat", ["_zeros_const_1", "_S", "_zeros_const_kv"], ["_zeros_shape"], axis=0))
    nodes.append(
        oh.make_node(
            "ConstantOfShape",
            ["_zeros_shape"],
            ["kv_zero_res"],
            value=oh.make_tensor("zval", onnx.TensorProto.BFLOAT16, [1], b"\x00\x00", raw=True),
        )
    )

    inits.append(
        oh.make_tensor("_scalar_0", onnx.TensorProto.INT64, [], np.array(0, dtype=np.int64).tobytes(), raw=True)
    )
    inits.append(
        oh.make_tensor("_scalar_1", onnx.TensorProto.INT64, [], np.array(1, dtype=np.int64).tobytes(), raw=True)
    )

    cur = "suffix_embs"
    for i in range(num_layers):
        b = f"E{i}"
        p = f"lm_expert.layers.{i}."
        is_even = every_n > 0 and (i % every_n == 0)

        # kv_stack[i] -> cached K/V [1, S, HKV, D] (scalar Gather indices drop the axis).
        inits.append(
            oh.make_tensor(f"{b}_gidx", onnx.TensorProto.INT64, [], np.array(i, dtype=np.int64).tobytes(), raw=True)
        )
        nodes.append(oh.make_node("Gather", ["kv_stack", f"{b}_gidx"], [f"{b}_kv_i"], axis=0))  # [2,1,S,HKV,D]
        nodes.append(oh.make_node("Gather", [f"{b}_kv_i", "_scalar_0"], [f"{b}_kc_5d"], axis=0))  # [1,S,HKV,D]
        nodes.append(oh.make_node("Gather", [f"{b}_kv_i", "_scalar_1"], [f"{b}_vc_5d"], axis=0))  # [1,S,HKV,D]

        if is_even:
            # merged qkv over the suffix
            w_q = sd[p + "self_attn.q_proj.weight"]
            w_k = sd[p + "self_attn.k_proj.weight"]
            w_v = sd[p + "self_attn.v_proj.weight"]
            w_qkv = _torch().cat([w_q, w_k, w_v], dim=0)
            _fused_norm_linear(
                f"{b}_qkv", cur, p + "input_layernorm.weight", w_qkv, q_dim + 2 * kv_dim, hidden, f"{b}_qkv_o"
            )
            inits.append(_i64_init(f"{b}_qkv_split", [q_dim, kv_dim, kv_dim]))
            nodes.append(
                oh.make_node("Split", [f"{b}_qkv_o", f"{b}_qkv_split"], [f"{b}_qf", f"{b}_kf", f"{b}_vf"], axis=-1)
            )
            inits.append(_i64_init(f"{b}_q_shape", [1, chunk, h, d]))
            inits.append(_i64_init(f"{b}_kv_shape", [1, chunk, hkv, d]))
            nodes.append(oh.make_node("Reshape", [f"{b}_qf", f"{b}_q_shape"], [f"{b}_q4"], allowzero=0))
            nodes.append(oh.make_node("Reshape", [f"{b}_kf", f"{b}_kv_shape"], [f"{b}_k4"], allowzero=0))
            nodes.append(oh.make_node("Reshape", [f"{b}_vf", f"{b}_kv_shape"], [f"{b}_v4"], allowzero=0))
            nodes.append(oh.make_node("Transpose", [f"{b}_q4"], [f"{b}_qt"], perm=[0, 2, 1, 3]))
            nodes.append(oh.make_node("Transpose", [f"{b}_k4"], [f"{b}_kt"], perm=[0, 2, 1, 3]))
            nodes.append(oh.make_node("Transpose", [f"{b}_v4"], [f"{b}_vt"], perm=[0, 2, 1, 3]))
            _rope(b, f"{b}_qt", "cos_abs", "sin_abs", f"{b}_qr")
            _rope(b, f"{b}_kt", "cos_abs", "sin_abs", f"{b}_kr")
            # cached K/V [1,S,HKV,D] -> [1,HKV,S,D]; concat fresh after cache.
            nodes.append(oh.make_node("Transpose", [f"{b}_kc_5d"], [f"{b}_kc_t"], perm=[0, 2, 1, 3]))
            nodes.append(oh.make_node("Transpose", [f"{b}_vc_5d"], [f"{b}_vc_t"], perm=[0, 2, 1, 3]))
            nodes.append(oh.make_node("Concat", [f"{b}_kc_t", f"{b}_kr"], [f"{b}_k_all"], axis=2))
            nodes.append(oh.make_node("Concat", [f"{b}_vc_t", f"{b}_vt"], [f"{b}_v_all"], axis=2))
            _repeat_kv(b, f"{b}_k_all", f"{b}_k_full")
            _repeat_kv(b, f"{b}_v_all", f"{b}_v_full")
            mask_name = "full_mask"
            q_name = f"{b}_qr"
        else:
            # q only from the suffix
            _fused_norm_linear(
                f"{b}_q", cur, p + "input_layernorm.weight", sd[p + "self_attn.q_proj.weight"], q_dim, hidden, f"{b}_qf"
            )
            inits.append(_i64_init(f"{b}_q_shape", [1, chunk, h, d]))
            nodes.append(oh.make_node("Reshape", [f"{b}_qf", f"{b}_q_shape"], [f"{b}_q4"], allowzero=0))
            nodes.append(oh.make_node("Transpose", [f"{b}_q4"], [f"{b}_qt"], perm=[0, 2, 1, 3]))
            _rope(b, f"{b}_qt", "cos_reb", "sin_reb", f"{b}_qr")
            # cache -> flat [1, S, kv_dim] -> k/v projections (320 -> 320, INT8)
            inits.append(_i64_init(f"{b}_flat_shape", [1, -1, kv_dim]))
            nodes.append(oh.make_node("Reshape", [f"{b}_kc_5d", f"{b}_flat_shape"], [f"{b}_kc_flat"], allowzero=0))
            nodes.append(oh.make_node("Reshape", [f"{b}_vc_5d", f"{b}_flat_shape"], [f"{b}_vc_flat"], allowzero=0))
            _linear_residual(
                f"{b}_kp",
                f"{b}_kc_flat",
                "kv_zero_res",
                sd[p + "self_attn.k_proj.weight"],
                kv_dim,
                kv_dim,
                f"{b}_k_proj",
            )
            _linear_residual(
                f"{b}_vp",
                f"{b}_vc_flat",
                "kv_zero_res",
                sd[p + "self_attn.v_proj.weight"],
                kv_dim,
                kv_dim,
                f"{b}_v_proj",
            )
            inits.append(_i64_init(f"{b}_kv4_shape", [1, -1, hkv, d]))
            nodes.append(oh.make_node("Reshape", [f"{b}_k_proj", f"{b}_kv4_shape"], [f"{b}_k4c"], allowzero=0))
            nodes.append(oh.make_node("Reshape", [f"{b}_v_proj", f"{b}_kv4_shape"], [f"{b}_v4c"], allowzero=0))
            nodes.append(oh.make_node("Transpose", [f"{b}_k4c"], [f"{b}_k_prefix"], perm=[0, 2, 1, 3]))
            nodes.append(oh.make_node("Transpose", [f"{b}_v4c"], [f"{b}_v_prefix"], perm=[0, 2, 1, 3]))
            _repeat_kv(b, f"{b}_k_prefix", f"{b}_k_full")
            _repeat_kv(b, f"{b}_v_prefix", f"{b}_v_full")
            mask_name = "prefix_bias"
            q_name = f"{b}_qr"

        # SDPA
        nodes.append(oh.make_node("Transpose", [f"{b}_k_full"], [f"{b}_kT"], perm=[0, 1, 3, 2]))
        nodes.append(oh.make_node("MatMul", [q_name, f"{b}_kT"], [f"{b}_qk"]))
        nodes.append(oh.make_node("Mul", [f"{b}_qk", "attn_scale"], [f"{b}_qks"]))
        nodes.append(oh.make_node("Add", [f"{b}_qks", mask_name], [f"{b}_qkm"]))
        nodes.append(oh.make_node("Softmax", [f"{b}_qkm"], [f"{b}_aw"], axis=-1))
        nodes.append(oh.make_node("MatMul", [f"{b}_aw", f"{b}_v_full"], [f"{b}_a4"]))
        nodes.append(oh.make_node("Transpose", [f"{b}_a4"], [f"{b}_ap"], perm=[0, 2, 1, 3]))
        inits.append(_i64_init(f"{b}_a_flat_shape", [1, chunk, q_dim]))
        nodes.append(oh.make_node("Reshape", [f"{b}_ap", f"{b}_a_flat_shape"], [f"{b}_a_flat"], allowzero=0))

        # o_proj + residual, post-norm SwiGLU MLP + residual
        _linear_residual(
            f"{b}_o", f"{b}_a_flat", cur, sd[p + "self_attn.o_proj.weight"], hidden, q_dim, f"{b}_post_attn"
        )
        w_gate = sd[p + "mlp.gate_proj.weight"]
        w_up = sd[p + "mlp.up_proj.weight"]
        w_gu = _torch().cat([w_gate, w_up], dim=0)
        _fused_norm_linear(
            f"{b}_gu", f"{b}_post_attn", p + "post_attention_layernorm.weight", w_gu, 2 * ff, hidden, f"{b}_gu_o"
        )
        inits.append(_i64_init(f"{b}_gu_split", [ff, ff]))
        nodes.append(oh.make_node("Split", [f"{b}_gu_o", f"{b}_gu_split"], [f"{b}_gate", f"{b}_up"], axis=-1))
        nodes.append(oh.make_node("Sigmoid", [f"{b}_gate"], [f"{b}_gsig"]))
        nodes.append(oh.make_node("Mul", [f"{b}_gate", f"{b}_gsig"], [f"{b}_gsilu"]))
        nodes.append(oh.make_node("Mul", [f"{b}_gsilu", f"{b}_up"], [f"{b}_ff_in"]))
        _linear_residual(
            f"{b}_dn", f"{b}_ff_in", f"{b}_post_attn", sd[p + "mlp.down_proj.weight"], hidden, ff, f"{b}_out"
        )
        cur = f"{b}_out"

    # ===== final expert norm + fp32 action_out_proj =====
    _rmsnorm("final_norm", cur, "lm_expert.norm.weight", "suffix_normed")
    nodes.append(oh.make_node("Cast", ["suffix_normed"], ["suffix_f32"], to=onnx.TensorProto.FLOAT))
    inits.append(_f32_initializer("out_w", sd["action_out_proj.weight"].t().contiguous()))
    nodes.append(oh.make_node("MatMul", ["suffix_f32", "out_w"], ["out_mm"]))
    inits.append(_f32_initializer("out_b", sd["action_out_proj.bias"]))
    nodes.append(oh.make_node("Add", ["out_mm", "out_b"], ["velocity"]))

    graph = oh.make_graph(nodes, "smolvla_expert_int8_per_row", [x_in, t_in, p_in, kv_in], [y_out], initializer=inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(_PLUGIN_DOMAIN, 1)])
    model.ir_version = 9
    from .onnx_io import save_plugin_onnx

    path = Path(out_path)
    save_plugin_onnx(model, path)
    c = Counter(n.op_type for n in nodes)
    logger.info(
        "Built SmolVLA expert INT8 per-row plugin graph: %d layers, %d nodes, ops=%s -> %s",
        num_layers,
        len(nodes),
        dict(sorted(c.items(), key=lambda x: -x[1])),
        path,
    )
    return path


def compute_smolvla_expert_sq_scales(
    action_expert: Any,
    forward_loop: Callable[[Any], None],
    *,
    block_size: int = 64,
    fwht: bool = False,
    fold_order: str = "after",
    alpha: float = 1.0,
    gptq_scales: "dict | None" = None,
) -> Dict[str, Any]:
    """Rotated-activation amax for every SmolVLA-expert INT4 GEMM site.

    Takes the same ``fwht`` and ``fold_order`` as every other capture so one call
    site can serve all four: ``"after"`` measures in the rotated frame,
    ``"before"`` in the raw one, and ``fwht`` selects the butterfly over the
    learned dense rotation. A scale measured through the wrong rotation, or in
    the wrong frame, mis-scales every channel with no error anywhere.
            gptq_scales: when given, this call switches from measuring SCALES to
            measuring each site's GPTQ Hessian, through the SAME taps. Pass the
            scales this capture returned on its first pass: the Hessian must be
            built on the activation the STORED weight multiplies (rotated, and
            divided by that vector), so it needs a second replay after the scales
            exist. Returns ``{site: (K, K) float64}`` in that mode.
    """
    if fold_order not in ("before", "after"):
        raise ValueError(f"fold_order must be 'before' or 'after', got {fold_order!r}")
    torch = _torch()
    expert = action_expert.eval()
    sd = {k: v.detach() for k, v in expert.state_dict().items()}
    every_n = int(expert.self_attn_every_n_layers)
    layers = expert.lm_expert.layers

    # The weight that reads each group's input channel, for the SmoothQuant
    # term. Padded exactly like the rotation: the scale is per PADDED channel.
    group_w: Dict[str, Any] = {}
    rot: Dict[str, tuple] = {}
    for i, _ in enumerate(layers):
        p = f"lm_expert.layers.{i}."
        b = f"E{i}"
        is_even = every_n > 0 and (i % every_n == 0)
        if is_even:
            w_qkv = torch.cat(
                [
                    sd[p + "self_attn.q_proj.weight"],
                    sd[p + "self_attn.k_proj.weight"],
                    sd[p + "self_attn.v_proj.weight"],
                ],
                dim=0,
            )
            group_w[f"{b}_qkv"] = omega.pad_in_dim(w_qkv, block_size)
            rot[f"{b}_qkv"] = foldq.site_rotation(group_w[f"{b}_qkv"], block_size, fwht)
        else:
            group_w[f"{b}_q"] = omega.pad_in_dim(sd[p + "self_attn.q_proj.weight"], block_size)
            rot[f"{b}_q"] = foldq.site_rotation(group_w[f"{b}_q"], block_size, fwht)
            group_w[f"{b}_kp"] = omega.pad_in_dim(sd[p + "self_attn.k_proj.weight"], block_size)
            rot[f"{b}_kp"] = foldq.site_rotation(group_w[f"{b}_kp"], block_size, fwht)
            group_w[f"{b}_vp"] = omega.pad_in_dim(sd[p + "self_attn.v_proj.weight"], block_size)
            rot[f"{b}_vp"] = foldq.site_rotation(group_w[f"{b}_vp"], block_size, fwht)
        group_w[f"{b}_o"] = omega.pad_in_dim(sd[p + "self_attn.o_proj.weight"], block_size)
        rot[f"{b}_o"] = foldq.site_rotation(group_w[f"{b}_o"], block_size, fwht)
        w_gu = torch.cat([sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]], dim=0)
        group_w[f"{b}_gu"] = omega.pad_in_dim(w_gu, block_size)
        rot[f"{b}_gu"] = foldq.site_rotation(group_w[f"{b}_gu"], block_size, fwht)
        group_w[f"{b}_dn"] = omega.pad_in_dim(sd[p + "mlp.down_proj.weight"], block_size)
        rot[f"{b}_dn"] = foldq.site_rotation(group_w[f"{b}_dn"], block_size, fwht)

    # One set of taps, two accumulators: amax on the first pass, the GPTQ
    # Hessian on the second. Duplicating the hook wiring is how the two passes
    # would drift on which tensor feeds which site.
    if gptq_scales is not None:
        amax, accum = foldq.hessian_accumulator(rot, gptq_scales, fold_order)
    else:
        amax, accum = foldq.scale_accumulator(rot, block_size, fold_order)
    handles = []
    for i, layer in enumerate(layers):
        b = f"E{i}"
        is_even = every_n > 0 and (i % every_n == 0)
        norm_key = f"{b}_qkv" if is_even else f"{b}_q"
        handles.append(layer.input_layernorm.register_forward_hook(lambda _m, _i, out, k=norm_key: accum(k, out)))
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(lambda _m, _i, out, k=f"{b}_gu": accum(k, out))
        )
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda _m, args, k=f"{b}_o": accum(k, args[0])))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda _m, args, k=f"{b}_dn": accum(k, args[0])))
        if not is_even:
            handles.append(
                layer.self_attn.k_proj.register_forward_pre_hook(lambda _m, args, k=f"{b}_kp": accum(k, args[0]))
            )
            handles.append(
                layer.self_attn.v_proj.register_forward_pre_hook(lambda _m, args, k=f"{b}_vp": accum(k, args[0]))
            )
    try:
        with torch.inference_mode():
            forward_loop(expert)
    finally:
        for h in handles:
            h.remove()
    logger.info("    Computed SmolVLA expert W4A4 SmoothQuant scales for %d sites.", len(amax))
    if gptq_scales is not None:
        return dict(amax)  # Hessians, not scales
    return foldq.finalize_scales(amax, weights=group_w, alpha=alpha)
