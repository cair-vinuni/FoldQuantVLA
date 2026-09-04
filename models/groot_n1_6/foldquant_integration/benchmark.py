# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Component latency of the bf16 PyTorch policy and of FoldQuant engine directories.

The timing loop is upstream's ``scripts/deployment/benchmark_inference.py``
(``benchmark_components``: per-iteration ``prepare_input`` + backbone, then
``action_head.get_action``, CUDA-synchronised, medians over ``--num-iterations``
after ``--warmup``; data processing timed once and shared across arms). Upstream's
script benchmarks a DiT-only engine through its own installer; this one times
the same loop with :func:`.runtime.install_engines`, so the FoldQuant arms
(LLM + DiT engines) and the float arm are measured by identical code.

Example::

    python -m foldquant_integration.benchmark --model-path checkpoints/GR00T-N1.6-LIBERO \\
        --dataset-path data/libero_calib --embodiment-tag libero_panda \\
        --arms w4a4=exports/n16_w4a4/engines float=exports/n16_float/engines
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tyro

from . import calibration
from ._upstream import ensure_deployment_on_path
from .runtime import install_engines


logger = logging.getLogger("foldquant.groot_n1_6.benchmark")


@dataclass
class BenchmarkConfig:
    model_path: str
    dataset_path: str
    """Dataset the benchmark observation is read from (episode 0, step 0 by default)."""

    arms: List[str] = field(default_factory=list)
    """Engine directories to time, as ``LABEL=DIR`` (or a bare ``DIR``, labelled by its name)."""

    embodiment_tag: Optional[str] = None
    num_iterations: int = 20
    warmup: int = 5
    seed: int = 42
    skip_eager: bool = False
    """Skip the PyTorch Eager arm (the speedup table then has no baseline)."""

    output: Optional[str] = None
    """Write per-arm medians and the raw per-iteration timings here as JSON."""

    video_backend: str = "torchcodec"


def _parse_arms(specs: List[str]) -> List[tuple]:
    arms = []
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep:
            path, label = spec, Path(spec).resolve().parent.name or Path(spec).name
        arms.append((label, Path(path)))
    return arms


def _time_arm(bench, policy, observation, args: BenchmarkConfig, shared_dp) -> Dict[str, Any]:
    comp = bench.benchmark_components(policy, observation, args.num_iterations, args.warmup)
    components = {
        "data_processing": shared_dp,
        "backbone": comp["backbone"],
        "action_head": comp["action_head"],
    }
    components["e2e"] = bench.compute_e2e_from_components(components)
    e2e = float(np.median(components["e2e"]))
    logger.info(
        "E2E %.0f ms (%.1f Hz) | data %.0f | backbone %.0f | action head %.0f ms",
        e2e,
        1000 / e2e,
        np.median(components["data_processing"]),
        np.median(components["backbone"]),
        np.median(components["action_head"]),
    )
    return components


def main(args: BenchmarkConfig) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ensure_deployment_on_path()
    import benchmark_inference as bench

    bench.set_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("benchmarking needs a CUDA device")
    device_name = bench.get_device_name()

    policy = calibration.load_policy(args.model_path, args.embodiment_tag, "cuda")
    dataset = calibration.load_dataset(policy, args.dataset_path, args.video_backend)
    observation = calibration.build_observations(policy, dataset, [calibration.SampleId(0, 0)])[0]
    denoising_steps = policy.model.action_head.num_inference_timesteps
    logger.info(
        "%s | action horizon %d | %d denoising steps",
        torch.cuda.get_device_name(0),
        policy.model.action_head.action_horizon,
        denoising_steps,
    )

    shared_dp = bench.benchmark_data_processing(policy, observation, args.num_iterations, warmup=10)
    results: Dict[str, Dict[str, Any]] = {}
    if not args.skip_eager:
        logger.info("arm: PyTorch Eager")
        results["PyTorch Eager"] = _time_arm(bench, policy, observation, args, shared_dp)

    components: Dict[str, List[str]] = {}
    for label, engine_dir in _parse_arms(args.arms):
        logger.info("arm: %s (%s)", label, engine_dir)
        installed = install_engines(policy, engine_dir)
        components[label] = sorted(installed.engines)
        try:
            results[label] = _time_arm(bench, policy, observation, args, shared_dp)
        finally:
            installed.remove()
        torch.cuda.empty_cache()

    bench.print_markdown_table(results, device_name, denoising_steps)

    report = {
        "device": torch.cuda.get_device_name(0),
        "model_path": args.model_path,
        "denoising_steps": int(denoising_steps),
        "num_iterations": args.num_iterations,
        "warmup": args.warmup,
        "arms": {
            label: {
                "components": components.get(label, []),
                "median_ms": {k: float(np.median(v)) for k, v in data.items()},
                "raw_ms": {k: [float(x) for x in np.asarray(v)] for k, v in data.items()},
            }
            for label, data in results.items()
        },
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        logger.info("wrote %s", args.output)
    return report


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))
