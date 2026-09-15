# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Calibrate, build and sanity-check an emulated W4A4 baseline arm for GR00T N1.7.

Two recipes, selected with ``--method``:

* ``holoq`` — HoloQ-VLA style: zigzag permutation + block-64 SVD·Hadamard
  rotation, GPTQ LLM weights, RTN DiT weights, per-token INT4 LLM activations,
  static per-denoising-step per-channel INT4 DiT activations (q99.9).
* ``duquant`` — DuQuant style as Omega-QVLA's baseline runs it: the same
  permutation and solvers, block-64 SVD-only rotation (eigenvectors of
  ``W^T W``), and a frozen per-channel q99.9 activation scale on both towers.

Calibration replays ``--num-calib`` seeded observations from a LeRobot dataset
through the bf16 policy, exactly as :mod:`.export_foldquant` does for the
FoldQuant arms, so the two families of arms see the same data. The pack is then
served with ``serve --baseline-pack`` or rolled out with
``eval_libero --baseline-pack``. Both arms are *emulated* — INT4 codes are
dequantised before a BF16 ``F.linear`` — so they measure a recipe's rounding,
never its speed.

Example::

    python -m foldquant_integration.baseline_w4a4 --command all --method duquant \\
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 --embodiment-tag libero_sim \\
        --dataset-path data/libero_10_no_noops_1.0.0_lerobot --output-dir exports/baseline_duquant
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Literal, Optional

from foldquant.provenance import public_path
import numpy as np
import torch
import tyro

from . import calibration
from .baselines import (
    METHODS,
    BaselineCalibrationCollector,
    apply_pack,
    build_pack,
    install_dit_step_context,
)


logger = logging.getLogger("foldquant.groot_n1_7.baseline_w4a4")

CALIBRATION_NAME = "calibration.pt"
PACK_NAME = "pack.pt"
CHECK_NAME = "check.json"


@dataclass
class BaselineConfig:
    model_path: str
    """Checkpoint directory (the same one the arm will be served from)."""

    dataset_path: str
    """LeRobot dataset the calibration observations are drawn from."""

    output_dir: str
    """Where ``calibration.pt``, ``pack.pt`` (+ ``.sha256``) and ``check.json`` go."""

    method: Literal["holoq", "duquant"] = "holoq"
    command: Literal["calibrate", "build", "check", "all"] = "all"
    embodiment_tag: Optional[str] = None
    """Embodiment tag (resolved from the checkpoint when omitted)."""

    num_calib: int = 128
    """Calibration observations, sampled as :mod:`.export_foldquant` samples them."""

    seed: int = 0
    """Seed of the calibration sample and of the denoising noise replayed during calibration."""

    suite: Optional[str] = None
    """Label stored in the artifacts; defaults to the dataset directory name."""

    num_check: int = 8
    """Held-out observations for the post-build action-cosine check (``check``/``all``)."""

    check_seed: int = 42
    """``torch.manual_seed(check_seed + i)`` before observation ``i``, both passes of the check."""

    percentile: float = 99.9
    topk: int = 512
    """Streaming order-statistic capacity of the HoloQ-style per-step DiT tables."""

    checkpoint_revision: str = ""
    """Recorded verbatim in the pack manifest (e.g. the Hub revision of the checkpoint)."""

    video_backend: str = "torchcodec"
    device: str = "cuda"


def _source_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def _model_device(policy) -> torch.device:
    return next(policy.model.parameters()).device


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0)
    )


def _action_vector(result: Any) -> torch.Tensor:
    action = result[0] if isinstance(result, tuple) else result
    parts = []
    for key in sorted(action.keys()):
        value = action[key]
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
        parts.append(tensor.float().flatten().cpu())
    return torch.cat(parts)


def _run_pass(policy, observations: List[Dict[str, Any]], seed: int) -> List[torch.Tensor]:
    actions = []
    with torch.inference_mode():
        for i, obs in enumerate(observations):
            torch.manual_seed(seed + i)
            actions.append(_action_vector(policy.get_action(obs)))
    return actions


def calibrate(args: BaselineConfig, policy, dataset, out: Path) -> Path:
    samples, observations = calibration.sample_observations(
        policy, dataset, args.num_calib, seed=args.seed
    )
    suite = args.suite or Path(args.dataset_path).name
    step_context = install_dit_step_context(policy.model)
    collector = BaselineCalibrationCollector(
        policy.model,
        suite=suite,
        run_id=f"{suite}-{args.method}-n{args.num_calib}-seed{args.seed}",
        method=args.method,
        percentile=args.percentile,
        topk=args.topk,
        seed=args.seed,
    )
    t0 = time.time()
    calibration.make_forward_loop(policy, observations, seed=args.seed)(None)
    logger.info(
        "%s calibration: %d observations replayed in %.0fs", args.method, len(observations), time.time() - t0
    )
    step_context.remove()
    path = collector.finalize(
        out / CALIBRATION_NAME,
        extra_manifest={
            "num_calib": args.num_calib,
            "calibration_seed": args.seed,
            "dataset": public_path(args.dataset_path),
            "calibration_source": "lerobot-dataset-replay",
            "samples": [{"episode": s.episode, "step": s.step} for s in samples],
        },
    )
    logger.info("wrote %s", path)
    return path


def build(args: BaselineConfig, policy, out: Path) -> Path:
    t0 = time.time()
    path = build_pack(
        policy.model,
        calibration_path=out / CALIBRATION_NAME,
        output_path=out / PACK_NAME,
        checkpoint=public_path(args.model_path),
        checkpoint_revision=args.checkpoint_revision,
        source_revision=_source_revision(),
    )
    logger.info("%s pack built in %.0fs: %s", args.method, time.time() - t0, path)
    return path


def check(args: BaselineConfig, policy, dataset, out: Path) -> Dict[str, Any]:
    """Action cosine of the emulated arm against the bf16 policy on held-out observations."""

    calibration_manifest = torch.load(out / CALIBRATION_NAME, map_location="cpu", weights_only=True)[
        "manifest"
    ]
    exclude = sorted({s["episode"] for s in calibration_manifest.get("samples", [])})
    _, observations = calibration.sample_observations(
        policy, dataset, args.num_check, seed=args.check_seed, exclude_episodes=exclude, heldout=True
    )
    device = _model_device(policy)
    step_context = install_dit_step_context(policy.model)
    reference = _run_pass(policy, observations, args.check_seed)
    summary = apply_pack(policy.model, out / PACK_NAME, backend="fake")
    policy.model.to(device=device)
    quantized = _run_pass(policy, observations, args.check_seed)
    step_context.remove()
    cosines = [_cos(a, b) for a, b in zip(quantized, reference)]
    worst = [float((a - b).abs().max()) for a, b in zip(quantized, reference)]
    report = {
        "method": args.method,
        "pack": public_path(str(out / PACK_NAME)),
        "quantized_linears": summary.total_linears,
        "num_observations": len(observations),
        "held_out_from_calibration": bool(exclude),
        "action_cos_mean": float(np.mean(cosines)),
        "action_cos_median": float(np.median(cosines)),
        "action_cos_min": float(np.min(cosines)),
        "action_max_abs_mean": float(np.mean(worst)),
        "action_max_abs_max": float(np.max(worst)),
        "seed": args.check_seed,
    }
    (out / CHECK_NAME).write_text(json.dumps(report, indent=2))
    logger.info(
        "%s vs bf16 on %d held-out observations: action cosine mean %.5f median %.5f min %.5f",
        args.method,
        len(observations),
        report["action_cos_mean"],
        report["action_cos_median"],
        report["action_cos_min"],
    )
    return report


def main(args: BaselineConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.method not in METHODS:
        raise ValueError(f"--method must be one of {sorted(METHODS)}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    policy = calibration.load_policy(args.model_path, args.embodiment_tag, args.device)
    dataset = calibration.load_dataset(policy, args.dataset_path, args.video_backend)
    logger.info("policy + dataset (%d episodes) in %.0fs", len(dataset), time.time() - t0)

    if args.command in ("calibrate", "all"):
        calibrate(args, policy, dataset, out)
    if args.command in ("build", "all"):
        build(args, policy, out)
    if args.command in ("check", "all"):
        report = check(args, policy, dataset, out)
        print(json.dumps(report))


if __name__ == "__main__":
    main(tyro.cli(BaselineConfig))
