# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Held-out drift of a FoldQuant engine directory against the bf16 PyTorch policy.

Upstream N1.5 ships an fp16 TensorRT path and no drift check of its own. This
tool scores two seams — ``backbone_features`` (what the LLM engine hands the
action head) and the decoded action chunk — on ``--num-samples`` observations
drawn from episodes the calibration never saw (read off the export manifest),
reporting per-seam cosine mean / min and the action max-abs error. Both
passes integrate from the same flow-matching noise (``torch.manual_seed(seed
+ i)`` before observation ``i``), so the difference measures the engines, not
the sampler.

No float-engine floor is measured for N1.5: upstream's fp16 engines follow a
different graph contract (see ``build_engines.py``) and are not loadable
here. Compare arms against each other on a common held-out set with
``--split-from``.

Example::

    python -m foldquant_integration.verify --model-path ... --dataset-path ... \\
        --engine-dir exports/n15_w4a4/engines --output exports/n15_w4a4/verify.json
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
from foldquant.provenance import public_path

from . import calibration
from ._upstream import LIBERO_DATA_CONFIG, MANIFEST_NAME
from .runtime import install_engines

logger = logging.getLogger("foldquant.groot_n1_5.verify")


@dataclass
class VerifyConfig:
    model_path: str
    dataset_path: str
    engine_dir: str
    """Engine directory produced by :mod:`.build_engines`."""

    output: str | None = None
    """Write the report here as JSON (default: ``<engine_dir>/verify.json``)."""

    embodiment_tag: str | None = None
    data_config: str = LIBERO_DATA_CONFIG
    """``module:Class`` data config, as given to the export."""

    denoising_steps: int | None = None
    """Flow-matching steps for both passes; the checkpoint's own value when omitted."""

    num_samples: int = 16
    """Held-out observations, from episodes outside the calibration split."""

    seed: int = 42
    """``torch.manual_seed(seed + i)`` before observation ``i``'s get_action, both passes."""

    allow_calibration_episodes: bool = False
    """Sample from every episode, calibration ones included — for datasets too small to hold any out.
    The report then measures fit, not generalisation, and says so."""

    split_from: str | None = None
    """Engine directory whose FoldQuant manifest defines the calibration split (default: ``engine_dir``).
    Point every arm of a comparison at the same directory and they all score the same held-out
    observations."""

    components: str = ""
    """Restrict the swap to these engines (``llm``, ``dit``), comma separated; empty installs every
    engine the directory holds. Scoring one seam at a time is how a drift figure is attributed to the
    LLM or to the DiT rather than to their sum."""

    video_backend: str = "torchcodec"


# float64: the LLM stream carries Qwen's massive-activation tokens (|x| ~ 1e3),
# and a flat fp32 cosine over seq x 2048 of them accumulates enough rounding to
# read 1.00015 for two identical-to-bf16 tensors.
def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0))


def _token_cos_min(a: torch.Tensor, b: torch.Tensor) -> float:
    a2 = a.double().reshape(-1, a.shape[-1])
    b2 = b.double().reshape(-1, b.shape[-1])
    return float(torch.nn.functional.cosine_similarity(a2, b2, dim=1).min())


def _load_manifest(engine_dir: Path) -> dict[str, Any]:
    path = engine_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: not a FoldQuant engine directory")
    return json.loads(path.read_text())


def _action_vector(action: dict[str, Any]) -> torch.Tensor:
    parts = []
    for k in sorted(action.keys()):
        v = action[k]
        t = v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v))
        parts.append(t.float().flatten().cpu())
    return torch.cat(parts)


def run_pass(policy, observations: list[dict[str, Any]], seed: int) -> dict[str, list[torch.Tensor]]:
    feats: list[torch.Tensor] = []
    acts: list[torch.Tensor] = []

    def _hook(_m, _args, output):
        feats.append(output["backbone_features"].detach().float().cpu().clone())

    handle = policy.model.backbone.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            for i, obs in enumerate(observations):
                torch.manual_seed(seed + i)
                acts.append(_action_vector(policy.get_action(obs)))
    finally:
        handle.remove()
    if len(feats) != len(observations):
        raise RuntimeError(f"captured {len(feats)} backbone outputs for {len(observations)} observations")
    return {"backbone_features": feats, "actions": acts}


def main(args: VerifyConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    engine_dir = Path(args.engine_dir)
    manifest = _load_manifest(engine_dir)
    split = _load_manifest(Path(args.split_from)) if args.split_from else manifest
    calib_episodes = sorted({s["episode"] for s in split["calibration"]["samples"]})
    excluded = [] if args.allow_calibration_episodes else calib_episodes

    t0 = time.time()
    policy = calibration.load_policy(
        args.model_path,
        args.embodiment_tag,
        "cuda",
        data_config=args.data_config,
        denoising_steps=args.denoising_steps,
    )
    dataset = calibration.load_dataset(policy, args.dataset_path, args.video_backend)
    samples, observations = calibration.sample_observations(
        policy,
        dataset,
        args.num_samples,
        seed=args.seed,
        exclude_episodes=excluded,
        heldout=True,
    )
    logger.info(
        "%d %s observations from %d episodes (calibration used %d episodes) in %.0fs",
        len(samples),
        "held-out" if excluded else "NOT held-out",
        len({s.episode for s in samples}),
        len(calib_episodes),
        time.time() - t0,
    )

    ref = run_pass(policy, observations, args.seed)
    again = run_pass(policy, observations, args.seed)
    repeat = min(_cos(a, b) for a, b in zip(ref["actions"], again["actions"]))
    logger.info("PyTorch repeatability (seeded): action cosine min %.6f", repeat)

    wanted = [c.strip() for c in args.components.split(",") if c.strip()] or None
    installed = install_engines(policy, engine_dir, components=wanted)
    components = sorted(installed.engines)
    logger.info("engines installed: %s", ", ".join(components))
    got = run_pass(policy, observations, args.seed)
    installed.remove()

    feat_cos = [_cos(a, b) for a, b in zip(got["backbone_features"], ref["backbone_features"])]
    feat_tok = [_token_cos_min(a, b) for a, b in zip(got["backbone_features"], ref["backbone_features"])]
    act_cos = [_cos(a, b) for a, b in zip(got["actions"], ref["actions"])]
    act_abs = [float((a - b).abs().max()) for a, b in zip(got["actions"], ref["actions"])]
    report = {
        "engine_dir": public_path(str(engine_dir)),
        "schemes": manifest["schemes"],
        "cascade": manifest.get("cascade", False),
        "components": components,
        "components_requested": wanted,
        "split_from": public_path(args.split_from),
        "denoising_steps": int(policy.denoising_steps),
        "num_samples": len(samples),
        "held_out": bool(excluded),
        "samples": [
            {
                "episode": s.episode,
                "step": s.step,
                "backbone_cos": fc,
                "backbone_token_cos_min": ft,
                "action_cos": ac,
                "action_max_abs": aa,
            }
            for s, fc, ft, ac, aa in zip(samples, feat_cos, feat_tok, act_cos, act_abs)
        ],
        "pytorch_repeat_action_cos_min": repeat,
        "backbone_features": {
            "cos_mean": float(np.mean(feat_cos)),
            "cos_median": float(np.median(feat_cos)),
            "cos_min": float(np.min(feat_cos)),
            "token_cos_min": float(np.min(feat_tok)),
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
        "backbone_features cos mean %.5f min %.5f token-min %.4f | actions cos mean %.5f min %.5f max_abs %.4f",
        report["backbone_features"]["cos_mean"],
        report["backbone_features"]["cos_min"],
        report["backbone_features"]["token_cos_min"],
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
