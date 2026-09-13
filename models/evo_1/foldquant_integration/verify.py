# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Held-out drift of a FoldQuant engine directory against the bf16 PyTorch model.

Two seams are scored on ``--num-samples`` observations drawn from episodes the
calibration never saw (read off the export manifest): the fused tokens the
language tower hands the head, and the denormalised action chunk upstream's
request handler returns — what the LIBERO client receives over the wire. Both
passes start their Euler integration from the same sample, because upstream
draws it from the global RNG and both passes are seeded identically per
observation, so the difference measures the engines rather than the sampler.

No float-engine floor is measured: upstream Evo-1 ships no TensorRT path.
Compare arms against each other on a common held-out set with ``--split-from``.

Example::

    python -m foldquant_integration.verify --checkpoint-dir <ckpt> --dataset-path <LeRobot LIBERO> \\
        --engine-dir exports/evo1_w4a4/engines --output exports/evo1_w4a4/verify.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from foldquant.drift import worst_channel
from foldquant.provenance import public_path

from . import calibration
from ._upstream import LIBERO_ARM_KEY, LIBERO_CHECKPOINT, LIBERO_DATASET_KEY, MANIFEST_NAME
from .runtime import ContextCapture, install_engines

logger = logging.getLogger("foldquant.evo_1.verify")


@dataclass
class VerifyConfig:
    engine_dir: str
    """Engine directory produced by :mod:`.build_engines`."""

    dataset_path: str
    """LeRobot dataset the held-out observations are drawn from (local path or hub id)."""

    checkpoint_dir: str = LIBERO_CHECKPOINT
    """Upstream checkpoint directory, as given to the export."""

    output: str | None = None
    """Write the report here as JSON (default: ``<engine_dir>/verify.json``)."""

    num_samples: int = 16
    """Held-out observations, from episodes outside the calibration split."""

    seed: int = 42
    """Seed of the held-out sample and of observation ``i``'s flow-matching start (``seed + i``), both passes."""

    episodes: str = ""
    """Optional episodes to restrict the dataset load to, as indices or ranges (``0-149``, ``0,3,7``)."""

    arm_key: str = LIBERO_ARM_KEY
    dataset_key: str = LIBERO_DATASET_KEY

    allow_calibration_episodes: bool = False
    """Sample from every episode, calibration ones included — for datasets too small to hold any out.
    The report then measures fit, not generalisation, and says so."""

    split_from: str | None = None
    """Engine directory whose FoldQuant manifest defines the calibration split (default: ``engine_dir``)."""

    components: str = ""
    """Restrict the swap to these engines (``llm``, ``action_head``), comma separated; empty installs
    every engine the directory holds. Scoring one seam at a time is how a drift figure is attributed
    to the tower or to the head rather than to their sum."""

    device: str = "cuda"


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """float64: a flat fp32 cosine over thousands of bf16 values reads above 1 for identical tensors."""
    return float(torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0))


def _token_cos_min(a: torch.Tensor, b: torch.Tensor) -> float:
    """Minimum over fused-token positions of the cosine across the hidden dimension."""
    a2 = a.double().reshape(-1, a.shape[-1])
    b2 = b.double().reshape(-1, b.shape[-1])
    return float(torch.nn.functional.cosine_similarity(a2, b2, dim=1).min())


def _load_manifest(engine_dir: Path) -> dict[str, Any]:
    path = engine_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: not a FoldQuant engine directory")
    return json.loads(path.read_text())


def run_pass(deployed, observations: list[dict[str, Any]], seed: int) -> dict[str, list[torch.Tensor]]:
    acts: list[torch.Tensor] = []
    width = 0
    with ContextCapture(deployed) as capture:
        for i, request in enumerate(observations):
            chunk = torch.as_tensor(calibration.infer(deployed, request, seed=seed + i)).float()
            width = int(chunk.shape[-1])
            acts.append(chunk.flatten())
    if len(capture.hidden) != len(observations):
        raise RuntimeError(f"captured {len(capture.hidden)} tower passes for {len(observations)} observations")
    return {"fused_tokens": [h.float().cpu() for h in capture.hidden], "actions": acts, "action_width": width}


def main(args: VerifyConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    engine_dir = Path(args.engine_dir)
    manifest = _load_manifest(engine_dir)
    split = _load_manifest(Path(args.split_from)) if args.split_from else manifest
    calib_episodes = sorted({s["episode"] for s in split["calibration"]["samples"]})
    excluded = [] if args.allow_calibration_episodes else calib_episodes

    t0 = time.time()
    deployed = calibration.load_policy(
        args.checkpoint_dir, arm_key=args.arm_key, dataset_key=args.dataset_key, device=args.device
    )
    dataset = calibration.load_dataset(args.dataset_path, episodes=calibration.parse_episodes(args.episodes))
    samples, observations = calibration.sample_observations(
        dataset, args.num_samples, seed=args.seed, exclude_episodes=excluded, heldout=True
    )
    logger.info(
        "%d %s observations from %d episodes (calibration used %d episodes) in %.0fs",
        len(samples),
        "held-out" if excluded else "NOT held-out",
        len({s.episode for s in samples}),
        len(calib_episodes),
        time.time() - t0,
    )

    ref = run_pass(deployed, observations, args.seed)
    again = run_pass(deployed, observations, args.seed)
    repeat = min(_cos(a, b) for a, b in zip(ref["actions"], again["actions"], strict=True))
    logger.info("PyTorch repeatability (seeded): action cosine min %.6f", repeat)

    wanted = [c.strip() for c in args.components.split(",") if c.strip()] or None
    installed = install_engines(deployed, engine_dir, components=wanted)
    components = sorted(installed.engines)
    logger.info("engines installed: %s", ", ".join(components))
    try:
        got = run_pass(deployed, observations, args.seed)
    finally:
        installed.remove()

    tok_cos = [_cos(a, b) for a, b in zip(got["fused_tokens"], ref["fused_tokens"], strict=True)]
    tok_min = [_token_cos_min(a, b) for a, b in zip(got["fused_tokens"], ref["fused_tokens"], strict=True)]
    act_cos = [_cos(a, b) for a, b in zip(got["actions"], ref["actions"], strict=True)]
    act_abs = [float((a - b).abs().max()) for a, b in zip(got["actions"], ref["actions"], strict=True)]
    act_worst = [
        worst_channel(a - b, order="step_major", width=got["action_width"])
        for a, b in zip(got["actions"], ref["actions"], strict=True)
    ]
    report = {
        "engine_dir": public_path(str(engine_dir)),
        "schemes": manifest["schemes"],
        "cascade": manifest.get("cascade", False),
        "components": components,
        "components_requested": wanted,
        "split_from": public_path(args.split_from),
        "num_samples": len(samples),
        # What a reader needs to re-run this and land on the same observations. Without
        # them a drift figure cannot be reproduced: the held-out plan is seeded, but it is
        # drawn from whatever episodes the dataset offers, so the same seed over a
        # different slice gives a different set. Two families were measured under an
        # episode restriction that nothing recorded, and re-running them without it
        # shared 0-1 of 32 samples with the committed record.
        "dataset_path": public_path(args.dataset_path),
        "seed": args.seed,
        "episodes": args.episodes or None,
        "held_out": bool(excluded),
        "samples": [
            {
                "episode": s.episode,
                "step": s.step,
                "fused_tokens_cos": tc,
                "fused_tokens_position_cos_min": tm,
                "action_cos": ac,
                "action_max_abs": aa,
                "action_worst": aw,
            }
            for s, tc, tm, ac, aa, aw in zip(samples, tok_cos, tok_min, act_cos, act_abs, act_worst, strict=True)
        ],
        "pytorch_repeat_action_cos_min": repeat,
        "fused_tokens": {
            "cos_mean": float(np.mean(tok_cos)),
            "cos_median": float(np.median(tok_cos)),
            "cos_min": float(np.min(tok_cos)),
            "position_cos_min": float(np.min(tok_min)),
        },
        "actions": {
            "cos_mean": float(np.mean(act_cos)),
            "cos_median": float(np.median(act_cos)),
            "cos_min": float(np.min(act_cos)),
            "max_abs": float(np.max(act_abs)),
            "max_abs_mean": float(np.mean(act_abs)),
        },
    }
    logger.info(
        "fused_tokens cos mean %.5f min %.5f position-min %.4f | actions cos median %.5f (mean %.5f min %.5f) max_abs %.4f",
        report["fused_tokens"]["cos_mean"],
        report["fused_tokens"]["cos_min"],
        report["fused_tokens"]["position_cos_min"],
        report["actions"]["cos_median"],
        report["actions"]["cos_mean"],
        report["actions"]["cos_min"],
        report["actions"]["max_abs"],
    )
    out = Path(args.output) if args.output else engine_dir / "verify.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out)
    return report


if __name__ == "__main__":
    main(tyro.cli(VerifyConfig))
