#!/usr/bin/env python3
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""ARC: grid-sweep the W4A4 LLM calibration knobs and rank presets by decoded action.

The W4A4 LLM fold has two calibration constants with no closed-form optimum,
both inherited from the INT8 recipe:

* ``sq_alpha`` — the SmoothQuant migration strength (0.4 in the registry,
  shared by every layer and site). Per-channel scaling is the load-bearing
  mechanism of the fold, so its knob is the highest-leverage constant in the
  pipeline — and refitting it is a pure offline re-fold, deployable with zero
  kernel or runtime change.
* ``act_clip_ratio`` — the per-row dynamic INT4 activation scale's clip (the
  kernel ships 1.0). Saturates the largest entry per row for a finer grid on the
  rest; deploys as one scalar plugin attribute.

Two stages keep the search cheap: a fast RTN grid ranks every combination on a
handful of calibration observations, then the top candidates plus the registry
default are re-scored with GPTQ rounding — the deployed numerics; GPTQ Hessians
are re-measured per alpha because they live in the transformed frame
``rot(x / s_alpha)``. GPTQ damping is fixed (``foldquant.llm_gptq.PERCDAMP``)
and is not a knob.

Two objectives. ``seam`` ranks by the decoder-output cosine at the LM/expert
seam — the customary criterion. ``actions`` runs the whole policy per combo and
ranks by the cosine of the DECODED ACTION CHUNK against the BF16 reference (each
observation seeded so both integrate from identical flow-matching noise). The
two can disagree: on the Qwen3-VL backbone the seam-optimal preset lowers action
fidelity, which is why the shipped ARC presets were picked on ``actions``.

The winner is an ordinary export input::

  python -m foldquant_integration.export_foldquant ... \\
      --llm-scheme w4a4_srg --llm-params '{"sq_alpha": 0.6, "act_clip_ratio": 0.85}'

Run from ``models/<family>/`` in that family's virtualenv::

  python ../../scripts/sweep_llm_quant_knobs.py --family groot_n1_6 \\
      --model-path <ckpt> --embodiment-tag libero_panda --dataset-path <libero_4suite> \\
      --objective actions --output ../../results/groot_n1_6/w4a4/sweep_llm_quant_knobs.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _groot_family import FAMILIES, Loaded, cosine, decoder_output  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sweep")

#: The registry default the sweep is measured against.
SHIPPED_ALPHA, SHIPPED_CLIP = 0.4, 1.0


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
    ap.add_argument("--max-samples", type=int, default=16, help="Observations scored per combo.")
    ap.add_argument("--rot-block-size", type=int, default=64)
    ap.add_argument("--alphas", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--clips", default="1.0,0.95,0.9,0.85")
    ap.add_argument("--top", type=int, default=3, help="Combos promoted to the GPTQ stage (plus the registry default).")
    ap.add_argument(
        "--per-site",
        action="store_true",
        help="After the global grid, coordinate-descent a PER-SITE alpha (qkv/gateup/down) around the winner; "
        "the fold is per-site already, so a per-site alpha deploys as the same offline re-fold.",
    )
    ap.add_argument("--objective", choices=("seam", "actions"), default="seam")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    from foldquant.llm_fake_quant import install_llm_per_row_emulation
    from foldquant.llm_gptq import compute_gptq_hessians_llm
    from foldquant.llm_rotation_sq import compute_sq_scales_llm

    run = Loaded(
        args.family,
        args.model_path,
        args.embodiment_tag,
        args.dataset_path,
        num_calib=args.max_samples,
        seed=args.seed,
        video_backend=args.video_backend,
    )
    decoder = run.decoder
    snapshots = run.capture()[: args.max_samples]
    logger.info("captured %d decoder snapshots", len(snapshots))

    with torch.inference_mode():
        reference = [decoder_output(decoder(*s[0], **s[1])).detach().clone() for s in snapshots]

    def _forward_fn(snap: Any) -> None:
        decoder(*snap[0], **snap[1])

    action_reference: List[torch.Tensor] = run.actions() if args.objective == "actions" else []
    for i, vec in enumerate(action_reference):
        # Fail here rather than sweep a grid of NaN.
        if vec.numel() == 0:
            raise RuntimeError(f"actions objective: observation {i} produced an empty action vector")
        if not torch.isfinite(vec).all():
            raise RuntimeError(f"actions objective: observation {i} reference contains non-finite values")
    if action_reference:
        logger.info(
            "actions objective: %d reference vectors, dim=%d", len(action_reference), action_reference[0].numel()
        )

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    clips = [float(c) for c in args.clips.split(",") if c.strip()]

    # SQ scales per alpha, measured once each and reused by every clip value.
    sq_by_alpha: Dict[float, Dict[str, Any]] = {}
    for alpha in alphas:
        logger.info("measuring SQ scales for alpha=%.2f ...", alpha)
        sq_by_alpha[alpha] = compute_sq_scales_llm(decoder, snapshots, alpha=alpha, forward_fn=_forward_fn)

    def _seam_scores() -> List[float]:
        with torch.inference_mode():
            return [cosine(decoder_output(decoder(*s[0], **s[1])), r) for s, r in zip(snapshots, reference)]

    def score(sq_scales: Dict[str, Any], clip: float, hessians: Optional[Dict[str, Any]]) -> float:
        emu = install_llm_per_row_emulation(
            decoder,
            sq_scales=sq_scales,
            gptq_hessians=hessians,
            rot_bs=args.rot_block_size,
            bits=4,
            act_clip_ratio=clip,
        )
        try:
            if args.objective == "actions":
                scores = [cosine(v, r) for v, r in zip(run.actions(), action_reference)]
            else:
                scores = _seam_scores()
            return float(sum(scores) / len(scores))
        finally:
            emu.remove()

    logger.info("=== stage 1: RTN grid (%d combos) ===", len(alphas) * len(clips))
    stage1: List[Dict[str, float]] = []
    for alpha in alphas:
        for clip in clips:
            value = score(sq_by_alpha[alpha], clip, None)
            stage1.append({"alpha": alpha, "clip": clip, "rtn_cosine": value})
            logger.info("  alpha=%.2f clip=%.2f  rtn cosine=%.6f", alpha, clip, value)

    ranked = sorted(stage1, key=lambda e: -e["rtn_cosine"])
    baseline_combo = next((e for e in stage1 if e["alpha"] == SHIPPED_ALPHA and e["clip"] == SHIPPED_CLIP), None)
    promote = ranked[: args.top]
    if baseline_combo is not None and baseline_combo not in promote:
        promote = promote + [baseline_combo]  # always score the registry default with GPTQ too

    logger.info("=== stage 2: GPTQ rescoring of %d combo(s) ===", len(promote))
    stage2: List[Dict[str, float]] = []
    hessians_by_alpha: Dict[float, Dict[str, Any]] = {}
    for entry in promote:
        alpha, clip = entry["alpha"], entry["clip"]
        if alpha not in hessians_by_alpha:
            logger.info("measuring GPTQ Hessians in the alpha=%.2f frame ...", alpha)
            hessians_by_alpha[alpha] = compute_gptq_hessians_llm(
                decoder, snapshots, sq_scales=sq_by_alpha[alpha], rot_bs=args.rot_block_size, forward_fn=_forward_fn
            )
        value = score(sq_by_alpha[alpha], clip, hessians_by_alpha[alpha])
        stage2.append({"alpha": alpha, "clip": clip, "gptq_cosine": value})
        logger.info("  alpha=%.2f clip=%.2f  GPTQ cosine=%.6f", alpha, clip, value)

    best = max(stage2, key=lambda e: e["gptq_cosine"])
    shipped = next((e for e in stage2 if e["alpha"] == SHIPPED_ALPHA and e["clip"] == SHIPPED_CLIP), None)

    per_site_result = None
    if args.per_site:
        # Coordinate descent in RTN (ranking fidelity) from the global winner, on
        # the seam objective; SQ scales for a mixed assignment are assembled per site.
        sites = ("qkv", "gateup", "down")
        n_layers = len(decoder.layers)

        def mixed_scales(assign: Dict[str, float]) -> Dict[str, Any]:
            return {
                f"L{i}_{site}": sq_by_alpha[assign[site]][f"L{i}_{site}"] for i in range(n_layers) for site in sites
            }

        def score_mixed(assign: Dict[str, float], clip: float, hessians: Optional[Dict[str, Any]] = None) -> float:
            emu = install_llm_per_row_emulation(
                decoder,
                sq_scales=mixed_scales(assign),
                gptq_hessians=hessians,
                rot_bs=args.rot_block_size,
                bits=4,
                act_clip_ratio=clip,
            )
            try:
                vals = _seam_scores()
                return float(sum(vals) / len(vals))
            finally:
                emu.remove()

        clip = best["clip"]
        assign = {s: best["alpha"] for s in sites}
        current = score_mixed(assign, clip)
        logger.info(
            "=== per-site coordinate descent (RTN, clip=%.2f) from alpha=%.2f: %.6f ===", clip, best["alpha"], current
        )
        trace = []
        for sweep_round in range(2):
            for site in sites:
                best_a, best_v = assign[site], current
                for a in alphas:
                    if a == assign[site]:
                        continue
                    trial = dict(assign)
                    trial[site] = a
                    v = score_mixed(trial, clip)
                    logger.info("  round%d %s alpha=%.2f -> %.6f", sweep_round, site, a, v)
                    if v > best_v:
                        best_a, best_v = a, v
                assign[site], current = best_a, best_v
                trace.append({"round": sweep_round, "site": site, "alpha": best_a, "cosine": current})
            logger.info("  after round %d: %s -> %.6f", sweep_round, assign, current)
        logger.info("GPTQ confirm of per-site assignment %s ...", assign)
        hess = compute_gptq_hessians_llm(
            decoder, snapshots, sq_scales=mixed_scales(assign), rot_bs=args.rot_block_size, forward_fn=_forward_fn
        )
        gptq_mixed = score_mixed(assign, clip, hess)
        logger.info(
            "PER-SITE %s clip=%.2f GPTQ cosine=%.6f (global best %.6f)", assign, clip, gptq_mixed, best["gptq_cosine"]
        )
        per_site_result = {
            "assignment": assign,
            "clip": clip,
            "rtn_cosine": current,
            "gptq_cosine": gptq_mixed,
            "trace": trace,
        }

    result = {
        "family": args.family,
        "objective": args.objective,
        "samples": len(snapshots),
        "seed": args.seed,
        "rot_block_size": args.rot_block_size,
        "stage1_rtn": stage1,
        "stage2_gptq": stage2,
        "best": best,
        "shipped_config": shipped,
        "gain_over_shipped": (best["gptq_cosine"] - shipped["gptq_cosine"]) if shipped else None,
        "per_site": per_site_result,
        "llm_params": {"sq_alpha": best["alpha"], "act_clip_ratio": best["clip"]},
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        logger.info("wrote %s", args.output)
    logger.info(
        "BEST alpha=%.2f clip=%.2f GPTQ cosine=%.6f  (registry %.2f/%.2f -> %s)",
        best["alpha"],
        best["clip"],
        best["gptq_cosine"],
        SHIPPED_ALPHA,
        SHIPPED_CLIP,
        f"{shipped['gptq_cosine']:.6f}" if shipped else "n/a",
    )


if __name__ == "__main__":
    main()
