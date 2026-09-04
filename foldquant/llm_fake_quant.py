# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Kernel-matched fake-quant emulation of the folded per-row LLM schemes.

Cascade calibration needs the *downstream* modules (DiT/expert/action head) to be
calibrated against the context a **quantized** LLM produces — the distribution the
deployed full-W4A4 engine actually feeds them — instead of the float baseline's.
Measured motivation: the solo arms pass while the composed ``act_w4a4_sr_llm_w4a4_srg`` arm
degrades super-additively (Pi0: the interaction term is 44% of end-to-end error;
Evo-1: 34/40 ∧ 21/40 solo → 15/40 composed), because every static quantizer
parameter of the downstream module (SmoothQuant fold, rotated-activation amax,
GPTQ/RTN rounding) was measured with a full-precision LLM upstream.

:func:`install_llm_per_row_emulation` mutates the live torch decoder in place so
one more calibration replay reproduces the deployed LLM numerics exactly:

* weights become the deployed frame — SmoothQuant fold (:func:`~.llm_rotation_sq.
  apply_sq_fold`), block-Hadamard fold (:func:`~.llm_rotation_sq.apply_rot_fold`),
  then per-output-row symmetric quantize-dequantize (GPTQ rounding at 4 bits via
  the same :func:`~.llm_gptq.gptq_prepare`/:func:`~.llm_gptq.gptq_quant_codes`
  the plugin-graph builder bakes, RTN at 8 bits);
* every quantized Linear gets a forward-pre-hook applying the runtime activation
  transform — block-64 FWHT then per-row dynamic symmetric QDQ — the fused
  ``RMSNorm→rotate→quant`` plugin performs. The SQ divide needs no hook: it is
  already inside the folded gammas / up-proj rows, exactly as deployed.

The returned handle restores the original weights bit-exact and removes every
hook, so the policy leaves this function's scope unchanged.

Float QDQ here is distribution-exact w.r.t. the s4 kernels: INT4 code products
(≤49) summed over ≤2^18 columns stay inside fp32's exact-integer range, so the
only divergence from the int32-accumulating tensor-core path is scale-multiply
rounding — negligible against the 4-bit grid.

Supported: Qwen2 / Qwen3 / Qwen3-VL decoders (GR00T + Evo-1 families) and Gemma
(Pi0/Pi0.5) — Gemma's ``(1+γ)`` RMSNorm is handled by folding in the effective
gamma and writing back ``gamma_folded − 1``, matching the deployed emitter's
``gemma_mode``. SmolLM2/Llama are refused rather than emulated unvalidated.

Torch is imported lazily (build-time only). No ``tensorrt`` / ``.so`` /
``foldquant.runtime`` imports.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Which GPTQ Hessian site rounds each rotated Linear's weight. Mirrors
#: ``llm_gptq.SITE_PROBES`` (merged qkv / gateup share one input, hence one site).
_WEIGHT_SITE: Dict[str, str] = {
    "self_attn.q_proj.weight": "qkv",
    "self_attn.k_proj.weight": "qkv",
    "self_attn.v_proj.weight": "qkv",
    "self_attn.o_proj.weight": "o",
    "mlp.gate_proj.weight": "gateup",
    "mlp.up_proj.weight": "gateup",
    "mlp.down_proj.weight": "down",
}

#: Norm gammas the SQ fold rewrites (in addition to the Linear weights above).
_SQ_GAMMA_KEYS: Tuple[str, str] = ("input_layernorm.weight", "post_attention_layernorm.weight")


def _perrow_qdq(x: Any, qmax: float, clip_ratio: float = 1.0) -> Any:
    """Per-row (last-dim) dynamic symmetric quantize-dequantize — the plugin's activation path.

    ``clip_ratio`` < 1 shrinks the per-row scale to ``clip*amax/qmax`` and clamps,
    trading saturation of the largest entry per row for a finer grid on the rest
    (QuaRot ships 0.9). The deployed kernel computes ``amax/qmax`` (ratio 1.0);
    a non-unit ratio is a candidate one-scalar plugin attribute, evaluated here
    first so the attribute is only added if it measurably pays.
    """
    import torch

    xf = x.float()
    scale = (clip_ratio * xf.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-12)
    return (torch.round(xf / scale).clamp(-qmax, qmax) * scale).to(x.dtype)


def _quantize_folded_weight(
    w: Any,
    *,
    qmax: float,
    prep: Optional[dict],
    row_clip: Any = None,
) -> Any:
    """QDQ one already-folded ``(N, K)`` weight the way the deployed graph rounds it.

    ``row_clip`` (``(N,)`` in (0, 1]) is a learned per-output-row clipping of the
    weight scale (LWC); the packed engine carries the clipped scale unchanged.
    """
    import torch

    from .llm_gptq import gptq_quant_codes

    if prep is not None:
        codes, scale = gptq_quant_codes(w.float(), prep, qmax=qmax, row_clip=row_clip)
        return (codes.float() * scale.unsqueeze(1)).to(w.dtype)
    wf = w.float()
    scale = (wf.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-12)
    if row_clip is not None:
        scale = scale * row_clip.detach().float().to(wf.device).reshape(-1, 1).clamp(min=1e-3, max=1.0)
    return (torch.round(wf / scale).clamp(-qmax, qmax) * scale).to(w.dtype)


class LlmEmulationHandle:
    """Undo token for :func:`install_llm_per_row_emulation` — restores weights, removes hooks."""

    def __init__(self, originals: List[Tuple[Any, Any]], hooks: List[Any]) -> None:
        self._originals = originals
        self._hooks = hooks

    def remove(self) -> None:
        import torch

        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        with torch.no_grad():
            for param, saved in self._originals:
                param.data.copy_(saved.to(param.device))
        self._originals = []


def install_llm_per_row_emulation(
    module: Any,
    *,
    sq_scales: Dict[str, Any],
    gptq_hessians: Optional[Dict[str, Any]],
    rot_bs: int,
    bits: int,
    qmax: Optional[float] = None,
    layer_bits: Optional[Dict[int, int]] = None,
    gptq_factors: Optional[Dict[str, dict]] = None,
    act_clip_ratio: Any = 1.0,
    site_bits: Optional[Dict[str, int]] = None,
    act_bits: Optional[int] = None,
    weight_clip: Optional[Dict[str, Any]] = None,
) -> LlmEmulationHandle:
    """Mutate the live decoder into a kernel-matched fake-quant of the folded per-row scheme.

    Args:
        module: The live LLM module (any wrapper ``resolve_qwen3_decoder`` unwraps).
        sq_scales: ``compute_sq_scales_llm`` output for this module (keys ``L{i}_{site}``).
        gptq_hessians: ``compute_gptq_hessians_llm`` output at 4 bits; ``None`` selects
            RTN weight rounding (the INT8 schemes).
        rot_bs: Block-Hadamard size (0/1 disables the rotation, matching the ``_sq``
            ablation schemes).
        bits: 4 or 8 — selects the symmetric code limit (7 / 127).
        qmax: Test override for the code limit; production callers leave it ``None``.
        layer_bits: Optional ``{layer_index: bits}`` override, for emulating a
            mixed-precision assignment (a few sensitive layers held at INT8
            inside an otherwise INT4 stack — both plugin widths already exist,
            so this is a deployable configuration, and emulating it here scores
            candidates without building an engine per candidate). Layers absent
            from the mapping use *bits*. GPTQ rounding is applied at whichever
            width a layer ends up on.
        site_bits: Optional ``{site: bits}`` override for ``site`` in ``qkv`` / ``o`` /
            ``gateup`` / ``down`` — a per-SITE mixed assignment (e.g. the FWHT residual
            sites ``o``/``down`` at INT8 inside an otherwise INT4 layer). Applies to both
            the weight rounding and that site's activation quantizer; ``layer_bits``
            still wins per layer when both are given for the same site.
        act_bits: Optional ACTIVATION width override (4 or 8) applied to every site's
            quantizer while the weights keep ``bits``/``site_bits`` — e.g. ``bits=4,
            act_bits=8`` emulates a W4A8 arm (INT4 weights, INT8 per-token activations),
            a configuration with no FoldQuant kernel yet; the emulation prices it before one
            is written.
        act_clip_ratio: Activation clip — one float for every site, or a dict keyed
            ``L{i}_{site}`` (learned per-layer/per-site clips; missing keys use 1.0).
        weight_clip: Optional learned per-output-row weight clips keyed
            ``L{i}_{weight_name}`` (``(N,)`` tensors in (0, 1]).
        gptq_factors: Optional pre-computed ``{site_key: gptq_prepare(...)}``
            reuse cache. The factorization is deterministic in the Hessian, so a
            caller that installs the emulation repeatedly over one calibration
            set (the per-layer sensitivity sweep) can factorize once instead of
            re-running a Cholesky per site per install. Populated in place when
            an empty dict is passed.

    Returns:
        A :class:`LlmEmulationHandle`; call ``.remove()`` to restore the module.

    Raises:
        NotImplementedError: For decoder conventions this emulation has not been
            validated against (SmolLM2 / Llama).
        KeyError: A missing SQ scale or GPTQ Hessian site — the emulation never
            silently skips a site the deployed graph quantizes.
    """
    import torch

    from .llm import resolve_qwen3_decoder
    from .llm_gptq import gptq_prepare
    from .llm_rotation_sq import ROT_WEIGHTS, _hadamard, _rot_last, apply_rot_fold, apply_sq_fold

    decoder = resolve_qwen3_decoder(module)
    cls_name = type(decoder).__name__
    if "SmolLM" in cls_name or "Llama" in cls_name:
        raise NotImplementedError(
            f"install_llm_per_row_emulation: {cls_name} is not a validated decoder convention for "
            "this emulation — cascade calibration is not wired for SmolLM2/Llama LLMs yet."
        )
    # GemmaRMSNorm applies its weight as (1+w) (y = rms(x)·(1+w); rms from x
    # BEFORE the gamma, same as Qwen) — so the SQ fold must see gamma=(1+w),
    # exactly as the deployed emitter materializes it, and the value written
    # back to the live module is gamma_folded − 1. The MLP difference
    # (gelu_tanh vs silu) does not touch the fold: the down fold is linear in
    # up and the activation applies to gate only.
    gemma_plus_one = "Gemma" in cls_name

    _QMAX = {4: 7.0, 8: 127.0}
    override_qmax = qmax
    layer_bits = dict(layer_bits or {})
    site_bits = dict(site_bits or {})
    unknown_sites = sorted(set(site_bits) - {"qkv", "o", "gateup", "down"})
    if unknown_sites:
        raise KeyError(f"install_llm_per_row_emulation: unknown site(s) in site_bits: {unknown_sites}")

    def _site_qmax(layer_index: int, site: str) -> float:
        """Code limit for one site of one layer: test override > layer override > site override > width."""
        if override_qmax is not None:
            return float(override_qmax)
        if layer_index in layer_bits:
            return _QMAX[int(layer_bits[layer_index])]
        return _QMAX[int(site_bits.get(site, bits))]

    def _act_qmax(layer_index: int, site: str) -> float:
        """Activation code limit: the explicit activation width wins, else the site's weight width."""
        if override_qmax is None and act_bits is not None:
            return _QMAX[int(act_bits)]
        return _site_qmax(layer_index, site)

    rotate = int(rot_bs) > 1
    H = _hadamard(int(rot_bs)) if rotate else None

    originals: List[Tuple[Any, Any]] = []
    hooks: List[Any] = []

    def _param(layer: Any, dotted: str) -> Any:
        sub, _, attr = dotted.rpartition(".")
        return getattr(layer.get_submodule(sub), attr)

    clip_map = act_clip_ratio if isinstance(act_clip_ratio, dict) else None
    clip_global = 1.0 if clip_map is not None else float(act_clip_ratio)
    weight_clip = dict(weight_clip or {})

    def _make_pre_hook(layer_qmax: float, clip: float) -> Any:
        def hook(_m: Any, args: tuple, kwargs: dict) -> Optional[tuple]:
            x = args[0]
            xt = _rot_last(x.float(), H.to(x.device)).to(x.dtype) if H is not None else x
            return (_perrow_qdq(xt, layer_qmax, clip), *args[1:]), kwargs

        return hook

    with torch.no_grad():
        for i, layer in enumerate(decoder.layers):
            state = {k: v for k, v in layer.state_dict().items()}
            if gemma_plus_one:
                for gk in _SQ_GAMMA_KEYS:
                    state[gk] = state[gk] + 1.0
            folded = apply_sq_fold(
                state,
                sq_scales[f"L{i}_qkv"].to(state["input_layernorm.weight"].device),
                sq_scales[f"L{i}_gateup"].to(state["input_layernorm.weight"].device),
                sq_scales[f"L{i}_down"].to(state["input_layernorm.weight"].device),
            )
            if rotate:
                folded = apply_rot_fold(folded, int(rot_bs))

            preps: Dict[str, dict] = {}
            if gptq_hessians is not None:
                # Factorized once per SITE (qkv/o/gateup/down), the same sharing the
                # plugin-graph builder uses. gptq_prepare copies the Hessian, so the
                # stashed dict survives for later callers.
                for site in ("qkv", "o", "gateup", "down"):
                    key = f"L{i}_{site}"
                    if gptq_factors is None:
                        preps[site] = gptq_prepare(gptq_hessians[key])
                    else:
                        if key not in gptq_factors:
                            gptq_factors[key] = gptq_prepare(gptq_hessians[key])
                        preps[site] = gptq_factors[key]

            new_weights: Dict[str, Any] = {
                k: (folded[k] - 1.0 if gemma_plus_one else folded[k]) for k in _SQ_GAMMA_KEYS
            }
            for wname in ROT_WEIGHTS:
                new_weights[wname] = _quantize_folded_weight(
                    folded[wname],
                    qmax=_site_qmax(i, _WEIGHT_SITE[wname]),
                    prep=preps.get(_WEIGHT_SITE[wname]) if gptq_hessians is not None else None,
                    row_clip=weight_clip.get(f"L{i}_{wname}"),
                )

            for key, tensor in new_weights.items():
                param = _param(layer, key)
                originals.append((param, param.data.detach().to("cpu", copy=True)))
                param.data.copy_(tensor.to(param.device, param.dtype))

            for wname in ROT_WEIGHTS:
                sub, _, _attr = wname.rpartition(".")
                site = _WEIGHT_SITE[wname]
                clip = float(clip_map.get(f"L{i}_{site}", 1.0)) if clip_map is not None else clip_global
                hooks.append(
                    layer.get_submodule(sub).register_forward_pre_hook(
                        _make_pre_hook(_act_qmax(i, site), clip), with_kwargs=True
                    )
                )

    logger.info(
        "LLM per-row emulation installed: %d layers, bits=%d%s, rot_bs=%d, weights=%s.",
        len(decoder.layers),
        bits,
        (f" ({len(layer_bits)} layer override(s))" if layer_bits else "")
        + (f" site_bits={site_bits}" if site_bits else "")
        + (f" act_bits={act_bits}" if act_bits is not None else ""),
        rot_bs,
        "gptq" if gptq_hessians is not None else "rtn",
    )
    return LlmEmulationHandle(originals, hooks)
