#!/usr/bin/env python3
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Learnable calibration for the FoldQuant W4A4 LLM (OmniQuant-style, kernel unchanged).

Per decoder layer, three fold-compatible parameter sets are learned by block-wise
reconstruction against the float layer under the deployed W4A4 numerics (STE):

* ``s``    — the SmoothQuant per-channel scales of the qkv / gateup / down sites
             (learnable equivalent transform; folded into gamma / up rows exactly
             as ``apply_sq_fold`` does), initialised from the ARC ``sq_alpha``.
* ``clip`` — the per-(layer, site) activation clip ratio (the plugin's
             ``act_clip_ratio`` attribute, one value per node), initialised from
             the ARC global value.
* ``gamma``— a per-output-row weight clipping (LWC) of the INT4 weight scale —
             any per-row scale is what the plugin's ``weight_scale`` epilogue
             consumes, so it deploys unchanged.

The rotation stays the fixed block Hadamard. Layers are trained in order with the
quantized stack's own outputs as inputs (OmniQuant / BRECQ convention) and the
float stack's outputs as targets. The result is scored on HELD-OUT observations
(episodes disjoint from the calibration split) with the kernel-statistics-matched
emulation (GPTQ weights, as deployed), paired per observation against the ARC
baseline, on both the LLM seam and the decoded action.

Outputs ``<out>.pt`` (learned tensors; consumed by the export through
``--llm-params '{"learned_calib": "<out>.pt", ...}'``, see
``foldquant.calibrate.load_learned_calib``) and ``<out>.json`` (scores).

Run from ``models/<family>/`` in that family's virtualenv::

  python ../../scripts/llm_learn_calib.py --family groot_n1_6 \\
      --model-path <ckpt> --embodiment-tag libero_panda --dataset-path <libero_4suite> \\
      --alpha 0.6 --clip 0.85 --output ../../exports/learn_calib_n16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _groot_family import FAMILIES, Loaded  # noqa: E402

logger = logging.getLogger("llm_learn_calib")

SITES = ("qkv", "o", "gateup", "down")
_QMAX = 7.0


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def _qdq_act(x: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    amax = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = clip * amax / _QMAX
    return _round_ste(x / scale).clamp(-_QMAX, _QMAX) * scale


def _qdq_w(w: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    amax = w.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = gamma.reshape(-1, 1) * amax / _QMAX
    return _round_ste(w / scale).clamp(-_QMAX, _QMAX) * scale


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def _dec_out(out: Any) -> torch.Tensor:
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _layer_out(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out


def _to_f32(v: Any) -> Any:
    if torch.is_tensor(v):
        return v.float() if v.is_floating_point() else v
    if isinstance(v, tuple):
        return tuple(_to_f32(x) for x in v)
    if isinstance(v, list):
        return [_to_f32(x) for x in v]
    return v


_LAYER_KW = ("attention_mask", "position_ids", "position_embeddings", "cache_position")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--embodiment-tag", default=None)
    ap.add_argument(
        "--dataset-path", required=True, help="LeRobot-layout calibration dataset (all four LIBERO suites)."
    )
    ap.add_argument("--video-backend", default="torchcodec")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=128, help="Calibration observations (training set).")
    ap.add_argument("--score-samples", type=int, default=32, help="Held-out observations (episode-disjoint).")
    ap.add_argument("--alpha", type=float, default=0.6, help="ARC sq_alpha (initial s).")
    ap.add_argument("--clip", type=float, default=0.85, help="ARC act_clip_ratio (initial clips).")
    ap.add_argument("--rot-block-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr-s", type=float, default=1e-3)
    ap.add_argument("--lr-clip", type=float, default=5e-3)
    ap.add_argument("--learn", default="s,clip,gamma", help="Subset of s,clip,gamma to learn.")
    ap.add_argument(
        "--objective",
        default="both",
        choices=["layer", "e2e", "both", "score"],
        help="layer = per-layer reconstruction; e2e = joint decoder-output cosine; both = layer then e2e; "
        "score = no training, score --init-from on held-out.",
    )
    ap.add_argument("--init-from", default=None, help="Learned file (.pt) to start from / to score.")
    ap.add_argument("--e2e-epochs", type=int, default=3)
    ap.add_argument("--lr-e2e", type=float, default=5e-4)
    ap.add_argument("--output", required=True, help="Output stem: writes <output>.pt and <output>.json.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    learn = set(x.strip() for x in args.learn.split(",") if x.strip())

    from foldquant.llm_fake_quant import _SQ_GAMMA_KEYS, _WEIGHT_SITE, install_llm_per_row_emulation
    from foldquant.llm_gptq import compute_gptq_hessians_llm
    from foldquant.llm_rotation_sq import (
        ROT_WEIGHTS,
        _hadamard,
        _rot_last,
        apply_rot_fold,
        apply_sq_fold,
        compute_sq_scales_llm,
    )

    t0 = time.time()
    run = Loaded(
        args.family,
        args.model_path,
        args.embodiment_tag,
        args.dataset_path,
        num_calib=args.max_samples,
        num_heldout=args.score_samples,
        seed=args.seed,
        video_backend=args.video_backend,
    )
    decoder = run.decoder
    inner = run.policy.model if hasattr(run.policy, "model") else run.policy
    device = next(decoder.parameters()).device
    layers = list(decoder.layers)
    H = _hadamard(args.rot_block_size).to(device)

    # ---------------- capture: decoder-level snapshots (calib + held-out) ----------------
    calib_snaps = run.capture()[: args.max_samples]
    ho_snaps = run.capture(run.heldout)[: args.score_samples]
    ho_observations = run.heldout[: len(ho_snaps)]
    logger.info(
        "captured %d calibration + %d held-out decoder snapshots (%.0fs)",
        len(calib_snaps),
        len(ho_snaps),
        time.time() - t0,
    )

    def _fwd(snap: Any) -> None:
        decoder(*snap[0], **snap[1])

    sq0 = compute_sq_scales_llm(decoder, calib_snaps, alpha=args.alpha, forward_fn=_fwd)

    # ---------------- capture: layer-0 inputs + per-layer kwargs on the float stack ----------------
    x_f: List[torch.Tensor] = []
    kw_f: List[Dict[str, Any]] = []

    def _l0_hook(_m: Any, a: tuple, kw: dict) -> None:
        x_f.append(a[0].detach().to("cpu"))
        kw_f.append({k: (v.detach().to("cpu") if torch.is_tensor(v) else v) for k, v in kw.items() if k in _LAYER_KW})

    h0 = layers[0].register_forward_pre_hook(_l0_hook, with_kwargs=True)
    try:
        with torch.no_grad():  # not inference_mode: these tensors feed autograd below
            for snap in calib_snaps:
                _fwd(snap)
    finally:
        h0.remove()
    x_f = [t.clone() for t in x_f]
    kw_f = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()} for kw in kw_f]
    if not kw_f or "position_embeddings" not in kw_f[0]:
        raise RuntimeError(f"layer-0 kwargs missing position_embeddings: {list(kw_f[0]) if kw_f else 'none'}")
    x_q: List[torch.Tensor] = [t.clone() for t in x_f]
    n_cal = len(x_f)

    def _kw(i: int) -> Dict[str, Any]:
        kw = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw_f[i].items()}
        pe = kw.get("position_embeddings")
        if isinstance(pe, (tuple, list)):
            kw["position_embeddings"] = tuple(x.to(device) for x in pe)
        kw["use_cache"] = False
        return kw

    # ---------------- per-layer reconstruction ----------------
    learned_sq: Dict[str, torch.Tensor] = {}
    learned_clip: Dict[str, float] = {}
    learned_gamma: Dict[str, torch.Tensor] = {}
    if args.init_from:
        init = torch.load(args.init_from, map_location="cpu")
        learned_sq = {k: torch.as_tensor(v).float() for k, v in init["sq_scales"].items()}
        learned_clip = {k: float(v) for k, v in (init.get("act_clip") or {}).items()}
        if "gamma" in learn:
            learned_gamma = {k: torch.as_tensor(v).float() for k, v in (init.get("weight_clip") or {}).items()}
        logger.info("initialised from %s (%d scales, %d clips)", args.init_from, len(learned_sq), len(learned_clip))
    site_keys = {"qkv": "L{i}_qkv", "gateup": "L{i}_gateup", "down": "L{i}_down"}
    clip_logit0 = math.log((args.clip - 0.5) / (1.0 - args.clip)) if 0.5 < args.clip < 1.0 else 6.0

    for i, layer in enumerate(layers if args.objective in ("layer", "both") else []):
        tl = time.time()
        base = {
            k: v.detach().float().clone()
            for k, v in layer.state_dict().items()
            if k in _SQ_GAMMA_KEYS or k in ROT_WEIGHTS
        }
        log_s = {
            site: torch.nn.Parameter(
                sq0[site_keys[site].format(i=i)].float().to(device).clone().log(), requires_grad="s" in learn
            )
            for site in site_keys
        }
        clip_u = {
            site: torch.nn.Parameter(torch.full((), clip_logit0, device=device), requires_grad="clip" in learn)
            for site in SITES
        }
        gamma_v = {
            w: torch.nn.Parameter(torch.ones(base[w].shape[0], device=device), requires_grad="gamma" in learn)
            for w in ROT_WEIGHTS
        }
        cur = {"clip": {s: 0.5 + 0.5 * torch.sigmoid(clip_u[s]) for s in SITES}}

        def _mk_hook(site: str) -> Any:
            def hook(_m: Any, a: tuple, kw: dict) -> Optional[tuple]:
                x = _rot_last(a[0].float(), H)
                return (_qdq_act(x, cur["clip"][site]).to(a[0].dtype), *a[1:]), kw

            return hook

        # teacher outputs (float layer, float inputs) — BEFORE the quantization hooks go on
        with torch.no_grad():
            teach = []
            for j in range(n_cal):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    teach.append(_layer_out(layer(x_f[j].to(device), **_kw(j))).detach().float().to("cpu"))
        hooks = [
            layer.get_submodule(w.rpartition(".")[0]).register_forward_pre_hook(
                _mk_hook(_WEIGHT_SITE[w]), with_kwargs=True
            )
            for w in ROT_WEIGHTS
        ]

        def _student_params() -> Dict[str, torch.Tensor]:
            s = {site: log_s[site].exp() for site in site_keys}
            folded = apply_sq_fold(base, s["qkv"], s["gateup"], s["down"])
            folded = apply_rot_fold(folded, args.rot_block_size)
            out = {k: folded[k] for k in _SQ_GAMMA_KEYS}
            for w in ROT_WEIGHTS:
                out[w] = _qdq_w(folded[w], gamma_v[w].clamp(0.5, 1.0))
            return out

        params = [p for group in (log_s, clip_u, gamma_v) for p in group.values() if p.requires_grad]
        if params:
            opt = torch.optim.Adam(
                [
                    {"params": [p for p in log_s.values() if p.requires_grad], "lr": args.lr_s},
                    {"params": [p for p in clip_u.values() if p.requires_grad], "lr": args.lr_clip},
                    {"params": [p for p in gamma_v.values() if p.requires_grad], "lr": args.lr_clip},
                ]
            )
            loss0 = loss1 = 0.0
            for ep in range(args.epochs):
                torch.manual_seed(ep)
                perm = torch.randperm(n_cal).tolist()
                tot = 0.0
                for b in range(0, n_cal, args.batch):
                    idx = perm[b : b + args.batch]
                    cur["clip"] = {s: 0.5 + 0.5 * torch.sigmoid(clip_u[s]) for s in SITES}
                    sp = _student_params()
                    loss = 0.0
                    for j in idx:  # per-sample calls (sequence lengths differ between samples)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            y = _layer_out(torch.func.functional_call(layer, sp, (x_q[j].to(device),), _kw(j)))
                        t = teach[j].to(device)
                        rel = ((y.float() - t) ** 2).sum(-1) / (t**2).sum(-1).clamp(min=1e-6)
                        loss = loss + rel.mean() / len(idx)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    tot += float(loss) * len(idx)
                if ep == 0:
                    loss0 = tot / n_cal
                loss1 = tot / n_cal
            logger.info(
                "layer %2d: rel-mse %.4e -> %.4e (%d epochs, %.0fs)", i, loss0, loss1, args.epochs, time.time() - tl
            )
        # freeze: record + propagate both stacks
        with torch.no_grad():
            cur["clip"] = {s: 0.5 + 0.5 * torch.sigmoid(clip_u[s]) for s in SITES}
            sp = _student_params()
            for site in site_keys:
                learned_sq[site_keys[site].format(i=i)] = log_s[site].exp().detach().to("cpu")
            for site in SITES:
                learned_clip[f"L{i}_{site}"] = float(cur["clip"][site])
            for w in ROT_WEIGHTS:
                learned_gamma[f"L{i}_{w}"] = gamma_v[w].clamp(0.5, 1.0).detach().to("cpu")
            for j in range(n_cal):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x_q[j] = (
                        _layer_out(torch.func.functional_call(layer, sp, (x_q[j].to(device),), _kw(j)))
                        .detach()
                        .to(torch.bfloat16)
                        .to("cpu")
                    )
                x_f[j] = teach[j].to(torch.bfloat16)
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

    if learned_sq:
        torch.save(
            {"sq_scales": learned_sq, "act_clip": learned_clip, "weight_clip": learned_gamma, "config": vars(args)},
            args.output + ".layer.pt",
        )
        logger.info("saved per-layer stage to %s.layer.pt", args.output)
    if args.objective in ("e2e", "both"):
        # ---------------- joint end-to-end stage: decoder-output cosine ----------------
        te = time.time()
        inner.to("cpu")
        decoder.to(device)
        torch.cuda.empty_cache()

        def _clone_snap(snap: Any) -> Any:
            a = tuple(v.clone() if torch.is_tensor(v) else v for v in snap[0])
            kw = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in snap[1].items()}
            kw.pop("past_key_values", None)
            kw["use_cache"] = False
            return (a, kw)

        e2e_snaps = [_clone_snap(sn) for sn in calib_snaps]
        with torch.no_grad():
            e2e_ref = [_dec_out(decoder(*sn[0], **sn[1])).detach().float() for sn in e2e_snaps]
        bases = {
            i: {
                k: v.detach().float().clone()
                for k, v in layer.state_dict().items()
                if k in _SQ_GAMMA_KEYS or k in ROT_WEIGHTS
            }
            for i, layer in enumerate(layers)
        }
        e_log_s = {
            (i, site): torch.nn.Parameter(
                (learned_sq.get(site_keys[site].format(i=i), sq0[site_keys[site].format(i=i)]))
                .float()
                .to(device)
                .clone()
                .log(),
                requires_grad="s" in learn,
            )
            for i in range(len(layers))
            for site in site_keys
        }

        def _logit(c: float) -> float:
            c = min(max(c, 0.501), 0.999)
            return math.log((c - 0.5) / (1.0 - c))

        e_clip_u = {
            (i, site): torch.nn.Parameter(
                torch.tensor(_logit(learned_clip.get(f"L{i}_{site}", args.clip)), device=device),
                requires_grad="clip" in learn,
            )
            for i in range(len(layers))
            for site in SITES
        }
        e_gamma = {
            (i, w): torch.nn.Parameter(
                learned_gamma.get(f"L{i}_{w}", torch.ones(bases[i][w].shape[0])).float().to(device).clone(),
                requires_grad="gamma" in learn,
            )
            for i in range(len(layers))
            for w in ROT_WEIGHTS
        }
        e_cur: Dict[Any, torch.Tensor] = {}

        def _e_hook(i: int, site: str) -> Any:
            def hook(_m: Any, a: tuple, kw: dict) -> Optional[tuple]:
                x = _rot_last(a[0].float(), H)
                return (_qdq_act(x, e_cur[(i, site)]).to(a[0].dtype), *a[1:]), kw

            return hook

        e_hooks = [
            layer.get_submodule(w.rpartition(".")[0]).register_forward_pre_hook(
                _e_hook(i, _WEIGHT_SITE[w]), with_kwargs=True
            )
            for i, layer in enumerate(layers)
            for w in ROT_WEIGHTS
        ]

        from torch.utils.checkpoint import checkpoint

        def _fold_layer(i: int, ls_qkv: Any, ls_gu: Any, ls_dn: Any, *gammas: Any) -> tuple:
            folded = apply_rot_fold(
                apply_sq_fold(bases[i], ls_qkv.exp(), ls_gu.exp(), ls_dn.exp()), args.rot_block_size
            )
            # STE in fp32, handed to the bf16 decoder as bf16: only the bf16 copies stay live for the forward
            outs = [folded[k].to(torch.bfloat16) for k in _SQ_GAMMA_KEYS]
            outs += [_qdq_w(folded[w], g.clamp(0.5, 1.0)).to(torch.bfloat16) for w, g in zip(ROT_WEIGHTS, gammas)]
            return tuple(outs)

        def _e_params() -> Dict[str, torch.Tensor]:
            out: Dict[str, torch.Tensor] = {}
            for i in range(len(layers)):
                # recomputed in backward: one layer's fold intermediates live at a time, not sixteen
                vals = checkpoint(
                    _fold_layer,
                    i,
                    e_log_s[(i, "qkv")],
                    e_log_s[(i, "gateup")],
                    e_log_s[(i, "down")],
                    *[e_gamma[(i, w)] for w in ROT_WEIGHTS],
                    use_reentrant=False,
                )
                for k, v in zip(list(_SQ_GAMMA_KEYS) + list(ROT_WEIGHTS), vals):
                    out[f"layers.{i}.{k}"] = v
            return out

        def _refresh_clips() -> None:
            for key, u in e_clip_u.items():
                e_cur[key] = 0.5 + 0.5 * torch.sigmoid(u)

        e_params = [q for grp in (e_log_s, e_clip_u, e_gamma) for q in grp.values() if q.requires_grad]
        if e_params:
            opt = torch.optim.Adam(
                [
                    {"params": [q for q in e_log_s.values() if q.requires_grad], "lr": args.lr_e2e},
                    {"params": [q for q in e_clip_u.values() if q.requires_grad], "lr": args.lr_e2e * 5},
                    {"params": [q for q in e_gamma.values() if q.requires_grad], "lr": args.lr_e2e * 5},
                ]
            )
            for ep in range(args.e2e_epochs):
                torch.manual_seed(100 + ep)
                tot = 0.0
                for j in torch.randperm(len(e2e_snaps)).tolist():
                    _refresh_clips()
                    sp = _e_params()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y = _dec_out(torch.func.functional_call(decoder, sp, e2e_snaps[j][0], e2e_snaps[j][1]))
                    cos_t = torch.nn.functional.cosine_similarity(y.float(), e2e_ref[j], dim=-1)
                    loss = 1.0 - cos_t.mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    tot += float(loss)
                logger.info("e2e epoch %d: mean(1-cos) %.5f (%.0fs)", ep, tot / len(e2e_snaps), time.time() - te)
        with torch.no_grad():
            _refresh_clips()
            for i in range(len(layers)):
                for site in site_keys:
                    learned_sq[site_keys[site].format(i=i)] = e_log_s[(i, site)].exp().detach().to("cpu")
                for site in SITES:
                    learned_clip[f"L{i}_{site}"] = float(e_cur[(i, site)])
                for w in ROT_WEIGHTS:
                    learned_gamma[f"L{i}_{w}"] = e_gamma[(i, w)].clamp(0.5, 1.0).detach().to("cpu")
        for h in e_hooks:
            h.remove()
        del e2e_snaps, e2e_ref, bases
        torch.cuda.empty_cache()
        inner.to(device)

    clips = sorted(learned_clip.values()) or [float(args.clip)]
    gmean = float(torch.cat(list(learned_gamma.values())).mean()) if learned_gamma else 1.0
    logger.info(
        "learned clips: min %.3f median %.3f max %.3f; weight gammas mean %.3f",
        clips[0],
        clips[len(clips) // 2],
        clips[-1],
        gmean,
    )
    torch.save(
        {"sq_scales": learned_sq, "act_clip": learned_clip, "weight_clip": learned_gamma, "config": vars(args)},
        args.output + ".pt",
    )
    logger.info("saved %s.pt", args.output)

    # ---------------- held-out scoring (deployed numerics: GPTQ weights) ----------------
    with torch.inference_mode():
        ho_ref = [_dec_out(decoder(*s[0], **s[1])).detach().clone() for s in ho_snaps]

    def _actions() -> List[torch.Tensor]:
        return run.actions(ho_observations)

    act_ref = _actions()

    def _score(name: str, sq: Dict[str, Any], clip: Any, wclip: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        hess = compute_gptq_hessians_llm(
            decoder, calib_snaps, sq_scales=sq, rot_bs=args.rot_block_size, forward_fn=_fwd
        )
        emu = install_llm_per_row_emulation(
            decoder,
            sq_scales=sq,
            gptq_hessians=hess,
            rot_bs=args.rot_block_size,
            bits=4,
            act_clip_ratio=clip,
            weight_clip=wclip,
        )
        try:
            with torch.inference_mode():
                seam = [_cos(_dec_out(decoder(*s[0], **s[1])), r) for s, r in zip(ho_snaps, ho_ref)]
            acts = [_cos(a, r) for a, r in zip(_actions(), act_ref)]
        finally:
            emu.remove()
        r = {
            "seam": sum(seam) / len(seam),
            "actions": sum(acts) / len(acts),
            "per_obs": {"seam": seam, "actions": acts},
        }
        logger.info("  %-10s seam=%.6f actions=%.6f", name, r["seam"], r["actions"])
        return r

    results: Dict[str, Any] = {"config": vars(args), "n_calib": n_cal, "n_heldout": len(ho_snaps)}
    results["baseline"] = _score("baseline", sq0, args.clip, None)
    results["learned"] = _score("learned", learned_sq, learned_clip, learned_gamma or None)
    # ablations: which learned group carries the effect (and which one fights GPTQ)
    if learned_gamma:
        results["s+clip"] = _score("s+clip", learned_sq, learned_clip, None)
        results["s_only"] = _score("s_only", learned_sq, args.clip, None)
        results["clip_only"] = _score("clip_only", sq0, learned_clip, None)
        results["gamma_only"] = _score("gamma_only", sq0, args.clip, learned_gamma)
    try:
        from scipy.stats import wilcoxon

        for variant in [v for v in ("learned", "s+clip", "s_only", "clip_only", "gamma_only") if v in results]:
            for key in ("seam", "actions"):
                a = results[variant]["per_obs"][key]
                b = results["baseline"]["per_obs"][key]
                d = [x - y for x, y in zip(a, b)]
                wins = sum(1 for x in d if x > 0)
                p = float(wilcoxon(a, b).pvalue) if any(d) else 1.0
                results[f"paired_{variant}_{key}"] = {"mean_delta": sum(d) / len(d), "wins": wins, "n": len(d), "p": p}
                logger.info(
                    "  paired %-10s %-8s Δ=%+.5f wins=%d/%d p=%.4f", variant, key, sum(d) / len(d), wins, len(d), p
                )
    except ImportError:
        pass
    json.dump(results, open(args.output + ".json", "w"), indent=2)
    logger.info("wrote %s.json (%.0fs total)", args.output, time.time() - t0)


if __name__ == "__main__":
    main()
