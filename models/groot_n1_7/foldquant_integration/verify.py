# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Held-out drift of a FoldQuant engine directory against the bf16 PyTorch policy.

Upstream ``verify_n1d7_trt.py`` compares one observation (trajectory 0,
step 0) under ``torch.manual_seed(42)``. This tool keeps its two seams —
``backbone_features`` (what the LLM engine hands the action head) and the
decoded action chunk — and its seeding, but scores ``--num-samples``
observations drawn from episodes the calibration never saw (read off the
export manifest), reporting per-seam cosine mean / min and the action
max-abs error. Both passes integrate from the same flow-matching noise, so
the difference measures the engines, not the sampler.

Example::

    python -m foldquant_integration.verify --model-path ... --dataset-path ... \\
        --engine-dir exports/n17_w4a4/engines --output exports/n17_w4a4/verify.json
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from foldquant.drift import worst_channel
from foldquant.provenance import public_path
from foldquant.runtime.plugins import load_plugins
import numpy as np
import torch
import tyro

from . import calibration
from ._upstream import MANIFEST_NAME, ensure_deployment_on_path


logger = logging.getLogger("foldquant.groot_n1_7.verify")


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
    dump_positions: Optional[str] = None
    """Write per-token cosine and reference-token norm for the worst positions of every sample
    here as JSON. Answers whether a low ``backbone_token_cos_min`` is a badly quantized position
    or a numerically empty one, which the min alone cannot say."""

    mode: str = "n17_full_pipeline"
    """``trt_model_forward.setup_tensorrt_engines`` mode."""


# float64: the LLM stream carries Qwen's massive-activation tokens (|x| ~ 1e3),
# and a flat fp32 cosine over 151x2048 of them accumulates enough rounding to
# read 1.00015 for two identical-to-bf16 tensors.
def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0)
    )


def _token_cos_min(a: torch.Tensor, b: torch.Tensor) -> float:
    a2 = a.double().reshape(-1, a.shape[-1])
    b2 = b.double().reshape(-1, b.shape[-1])
    return float(torch.nn.functional.cosine_similarity(a2, b2, dim=1).min())


def _position_report(engine: torch.Tensor, reference: torch.Tensor) -> Dict[str, Any]:
    """Per-token cosine beside the reference token's own norm.

    A cosine on a near-zero vector is ill-conditioned: a tiny absolute
    perturbation swings it, while the token contributes almost nothing
    downstream. Reporting the norm next to the cosine is what separates "this
    position is badly quantized" from "this position is numerically empty and
    the metric is noise", which a min alone cannot distinguish.
    """
    a = engine.double().reshape(-1, engine.shape[-1])
    b = reference.double().reshape(-1, reference.shape[-1])
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    ref_norm = b.norm(dim=1)
    order = torch.argsort(cos)
    return {
        "num_positions": int(cos.numel()),
        "reference_norm_median": float(ref_norm.median()),
        "worst": [
            {
                "position": int(i),
                "cos": float(cos[i]),
                "reference_norm": float(ref_norm[i]),
                "norm_rank_pct": float((ref_norm < ref_norm[i]).double().mean() * 100.0),
            }
            for i in order[:16].tolist()
        ],
    }


def _load_manifest(engine_dir: Path) -> Optional[Dict[str, Any]]:
    """The ``foldquant_export.json`` of a FoldQuant engine directory; ``None`` for a float one."""
    path = engine_dir / MANIFEST_NAME
    return json.loads(path.read_text()) if path.is_file() else None


def _action_vector(result: Any) -> torch.Tensor:
    return _action_vector_with_layout(result)[0]


def _action_vector_with_layout(result: Any) -> "tuple[torch.Tensor, list, int]":
    """The flat action, plus the channel names and the steps per channel.

    The policy answers with a dict of named channels, each a whole chunk, so
    the flat vector is channel-major and element ``i`` belongs to channel
    ``i // steps``. Returning the names with it is what lets a drift figure say
    which channel moved instead of only how far.
    """
    action = result[0] if isinstance(result, tuple) else result
    labels = sorted(action.keys())
    parts = []
    for k in labels:
        v = action[k]
        t = v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v))
        parts.append(t.float().flatten().cpu())
    per = int(parts[0].numel()) if parts else 0
    return torch.cat(parts), labels, per


def run_pass(
    policy, observations: List[Dict[str, Any]], seed: int
) -> Dict[str, List[torch.Tensor]]:
    feats: List[torch.Tensor] = []
    acts: List[torch.Tensor] = []
    labels: list = []
    per = 0

    def _hook(_m, _args, output):
        feats.append(output["backbone_features"].detach().float().cpu().clone())

    handle = policy.model.backbone.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            for i, obs in enumerate(observations):
                torch.manual_seed(seed + i)
                vector, labels, per = _action_vector_with_layout(policy.get_action(obs))
                acts.append(vector)
    finally:
        handle.remove()
    if len(feats) != len(observations):
        raise RuntimeError(
            f"captured {len(feats)} backbone outputs for {len(observations)} observations"
        )
    return {
        "backbone_features": feats,
        "actions": acts,
        "action_labels": labels,
        "action_steps": per,
    }


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

    if manifest is not None:
        load_plugins(manifest["plugin_libs"])
    ensure_deployment_on_path()
    from trt_model_forward import setup_tensorrt_engines

    setup_tensorrt_engines(policy, str(engine_dir), mode=args.mode)
    got = run_pass(policy, observations, args.seed)

    feat_cos = [_cos(a, b) for a, b in zip(got["backbone_features"], ref["backbone_features"])]
    feat_tok = [
        _token_cos_min(a, b) for a, b in zip(got["backbone_features"], ref["backbone_features"])
    ]
    act_cos = [_cos(a, b) for a, b in zip(got["actions"], ref["actions"])]
    act_abs = [float((a - b).abs().max()) for a, b in zip(got["actions"], ref["actions"])]
    act_worst = [
        worst_channel(
            a - b, order="channel_major", width=got["action_steps"], labels=got["action_labels"]
        )
        for a, b in zip(got["actions"], ref["actions"])
    ]
    report = {
        "engine_dir": public_path(str(engine_dir)),
        "schemes": manifest["schemes"] if manifest is not None else {},
        # The float arm comes off upstream's pipeline and carries no FoldQuant manifest, so
        # `schemes` is empty for it. Listing the engines that are actually installed states
        # the arm's scope either way — without it two arms of different scope can share the
        # name "float" and nothing in the record distinguishes them.
        "components": sorted(p.name for p in sorted(engine_dir.glob("*.engine"))),
        "cascade": manifest.get("cascade", False) if manifest is not None else False,
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
        "held_out": bool(excluded),
        "samples": [
            {
                "episode": s.episode,
                "step": s.step,
                "backbone_cos": fc,
                "backbone_token_cos_min": ft,
                "action_cos": ac,
                "action_max_abs": aa,
                "action_worst": aw,
            }
            for s, fc, ft, ac, aa, aw in zip(
                samples, feat_cos, feat_tok, act_cos, act_abs, act_worst
            )
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
        "backbone_features cos mean %.5f min %.5f token-min %.4f | actions cos median %.5f (mean %.5f min %.5f) max_abs %.4f",
        report["backbone_features"]["cos_mean"],
        report["backbone_features"]["cos_min"],
        report["backbone_features"]["token_cos_min"],
        report["actions"]["cos_median"],
        report["actions"]["cos_mean"],
        report["actions"]["cos_min"],
        report["actions"]["max_abs"],
    )
    if args.dump_positions:
        dump = {
            "engine_dir": public_path(str(engine_dir)),
            "samples": [
                {"episode": s.episode, "step": s.step, **_position_report(g, r)}
                for s, g, r in zip(samples, got["backbone_features"], ref["backbone_features"])
            ],
        }
        Path(args.dump_positions).write_text(json.dumps(dump, indent=2))
        logger.info("wrote per-position report %s", args.dump_positions)

    out = Path(args.output) if args.output else engine_dir / "verify.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out)
    return report


if __name__ == "__main__":
    main(tyro.cli(VerifyConfig))
