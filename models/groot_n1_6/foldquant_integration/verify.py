# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Held-out drift of a FoldQuant engine directory against the bf16 PyTorch policy.

Upstream N1.6 ships a DiT-only TensorRT path and no drift check of its own.
This tool scores two seams — ``backbone_features`` (what the LLM engine hands
the action head) and the decoded action chunk — on ``--num-samples``
observations drawn from episodes the calibration never saw (read off the
export manifest), reporting per-seam cosine mean / min and the action
max-abs error. Both passes integrate from the same flow-matching noise
(``torch.manual_seed(seed + i)`` before observation ``i``), so the difference
measures the engines, not the sampler.

Example::

    python -m foldquant_integration.verify --model-path ... --dataset-path ... \\
        --engine-dir exports/n16_w4a4/engines --output exports/n16_w4a4/verify.json
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tyro

from . import calibration
from ._upstream import MANIFEST_NAME
from .runtime import install_engines


logger = logging.getLogger("foldquant.groot_n1_6.verify")


@dataclass
class VerifyConfig:
    model_path: str
    dataset_path: str
    engine_dir: str
    """Engine directory produced by :mod:`.build_engines`."""

    output: Optional[str] = None
    """Write the report here as JSON (default: ``<engine_dir>/verify.json``)."""

    embodiment_tag: Optional[str] = None
    num_samples: int = 16
    """Held-out observations, from episodes outside the calibration split."""

    seed: int = 42
    """``torch.manual_seed(seed + i)`` before observation ``i``'s get_action, both passes."""

    allow_calibration_episodes: bool = False
    """Sample from every episode, calibration ones included — for datasets too small to hold any out.
    The report then measures fit, not generalisation, and says so."""

    split_from: Optional[str] = None
    """Engine directory whose FoldQuant manifest defines the calibration split (default: ``engine_dir``).
    A float directory built by upstream has no manifest; point this at the quantized arm it is
    compared with, and both score the same held-out observations — the float engines are the
    floor of the drift metric, not zero."""

    video_backend: str = "torchcodec"


# float64: the LLM stream carries Qwen's massive-activation tokens (|x| ~ 1e3),
# and a flat fp32 cosine over seq x 2048 of them accumulates enough rounding to
# read 1.00015 for two identical-to-bf16 tensors.
def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0)
    )


def _token_cos_min(a: torch.Tensor, b: torch.Tensor) -> float:
    a2 = a.double().reshape(-1, a.shape[-1])
    b2 = b.double().reshape(-1, b.shape[-1])
    return float(torch.nn.functional.cosine_similarity(a2, b2, dim=1).min())


def _load_manifest(engine_dir: Path) -> Optional[Dict[str, Any]]:
    """The ``foldquant_export.json`` of a FoldQuant engine directory; ``None`` for a float one."""
    path = engine_dir / MANIFEST_NAME
    return json.loads(path.read_text()) if path.is_file() else None


def _action_vector(result: Any) -> torch.Tensor:
    action = result[0] if isinstance(result, tuple) else result
    parts = []
    for k in sorted(action.keys()):
        v = action[k]
        t = v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v))
        parts.append(t.float().flatten().cpu())
    return torch.cat(parts)


def run_pass(
    policy, observations: List[Dict[str, Any]], seed: int
) -> Dict[str, List[torch.Tensor]]:
    feats: List[torch.Tensor] = []
    acts: List[torch.Tensor] = []

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
        raise RuntimeError(
            f"captured {len(feats)} backbone outputs for {len(observations)} observations"
        )
    return {"backbone_features": feats, "actions": acts}


def main(args: VerifyConfig) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    engine_dir = Path(args.engine_dir)
    manifest = _load_manifest(engine_dir)
    split = _load_manifest(Path(args.split_from)) if args.split_from else manifest
    if split is None:
        raise FileNotFoundError(
            f"{engine_dir / MANIFEST_NAME} not found: a float engine directory carries no calibration "
            "split. Pass --split-from <quantized engine dir> to score it on that arm's held-out set."
        )
    calib_episodes = sorted({s["episode"] for s in split["calibration"]["samples"]})
    excluded = [] if args.allow_calibration_episodes else calib_episodes

    t0 = time.time()
    policy = calibration.load_policy(args.model_path, args.embodiment_tag, "cuda")
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

    installed = install_engines(policy, engine_dir)
    components = sorted(installed.engines)
    logger.info("engines installed: %s", ", ".join(components))
    got = run_pass(policy, observations, args.seed)
    installed.remove()

    feat_cos = [_cos(a, b) for a, b in zip(got["backbone_features"], ref["backbone_features"])]
    feat_tok = [
        _token_cos_min(a, b) for a, b in zip(got["backbone_features"], ref["backbone_features"])
    ]
    act_cos = [_cos(a, b) for a, b in zip(got["actions"], ref["actions"])]
    act_abs = [float((a - b).abs().max()) for a, b in zip(got["actions"], ref["actions"])]
    report = {
        "engine_dir": str(engine_dir),
        "schemes": manifest["schemes"] if manifest is not None else {},
        "cascade": manifest.get("cascade", False) if manifest is not None else False,
        "components": components,
        "split_from": args.split_from,
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
            "cos_min": float(np.min(feat_cos)),
            "token_cos_min": float(np.min(feat_tok)),
        },
        "actions": {
            "cos_mean": float(np.mean(act_cos)),
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
