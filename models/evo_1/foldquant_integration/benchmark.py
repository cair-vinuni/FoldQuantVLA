# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Component latency of the bf16 PyTorch model and of FoldQuant engine directories.

Upstream reports no per-component timing for a request, so this module carries
its own timer: one observation from the dataset, ``--warmup`` untimed calls,
then ``--num-iterations`` timed ones, CUDA-synchronised, medians reported. Per
iteration it times the pieces of one served request where they run:

* ``embed``        — the InternVL3 embedder end to end (tiles, vision tower,
  fusion) *including* the language tower;
* ``llm``          — the language tower alone, inside that;
* ``denoise_loop`` — ``get_action``: all 50 Euler steps;
* ``e2e``          — a separate, whole ``infer_from_json_dict`` call, which is
  what the websocket handler runs per request (image decode and normalisation
  included).

Every arm — PyTorch Eager and each FoldQuant engine directory — goes through
the same loop with :func:`.runtime.install_engines` swapping the engines in.
Upstream serves Evo-1 eagerly, so the eager arm is the deployed reference.

Example::

    python -m foldquant_integration.benchmark --checkpoint-dir <ckpt> --dataset-path <LeRobot LIBERO> \\
        --arms w4a4=exports/evo1_w4a4/engines
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from . import calibration
from ._upstream import LIBERO_ARM_KEY, LIBERO_CHECKPOINT, LIBERO_DATASET_KEY
from .runtime import head_module, install_engines, llm_module, model_of

logger = logging.getLogger("foldquant.evo_1.benchmark")

COMPONENT_ORDER = ("embed", "llm", "denoise_loop", "e2e")
_MISSING = object()


@dataclass
class BenchmarkConfig:
    dataset_path: str
    """Dataset the benchmark observation is read from (first episode, step 0)."""

    checkpoint_dir: str = LIBERO_CHECKPOINT
    arms: list[str] = field(default_factory=list)
    """Engine directories to time, as ``LABEL=DIR`` (or a bare ``DIR``, labelled by its parent's name)."""

    num_iterations: int = 20
    warmup: int = 5
    seed: int = 42
    episodes: str = ""
    """Optional episodes to restrict the dataset load to, as indices or ranges (``0-149``, ``0,3,7``)."""

    arm_key: str = LIBERO_ARM_KEY
    dataset_key: str = LIBERO_DATASET_KEY
    skip_eager: bool = False
    """Skip the PyTorch Eager arm (the speedup table then has no baseline)."""

    output: str | None = None
    """Write per-arm medians and the raw per-iteration timings here as JSON."""

    device: str = "cuda"


def _parse_arms(specs: list[str]) -> list[tuple]:
    arms = []
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep:
            path, label = spec, Path(spec).resolve().parent.name or Path(spec).name
        arms.append((label, Path(path)))
    return arms


def _timed(fn: Callable[[], Any]) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


class _Stopwatches:
    """Instance-level timing wrappers around bound methods, summed per bucket; ``restore()`` undoes them."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self._undo: list[Callable[[], None]] = []

    def reset(self) -> None:
        self.ms = {}

    def wrap(self, obj, name: str, bucket: str) -> None:
        previous = obj.__dict__.get(name, _MISSING)
        fn = getattr(obj, name)

        def wrapper(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            self.ms[bucket] = self.ms.get(bucket, 0.0) + (time.perf_counter() - t0) * 1000.0
            return out

        setattr(obj, name, wrapper)

        def undo():
            if previous is _MISSING:
                obj.__dict__.pop(name, None)
            else:
                setattr(obj, name, previous)

        self._undo.append(undo)

    def restore(self) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo = []


def time_components(deployed, request: dict[str, Any], args: BenchmarkConfig) -> dict[str, list[float]]:
    """Per-iteration timings by bucket."""
    model = model_of(deployed)
    timings: dict[str, list[float]] = {k: [] for k in COMPONENT_ORDER}
    watches = _Stopwatches()

    def _watch():
        watches.wrap(model.embedder, "get_fused_image_text_embedding_from_tensor_images", "embed")
        watches.wrap(llm_module(deployed), "forward", "llm")
        watches.wrap(head_module(deployed), "get_action", "denoise_loop")

    try:
        with torch.inference_mode():
            for i in range(args.warmup + args.num_iterations):
                _watch()
                calibration.infer(deployed, request, seed=args.seed + i)
                parts = dict(watches.ms)
                watches.reset()
                # The whole call is timed with the wrappers off, so their synchronisations do not count.
                watches.restore()
                e2e = _timed(lambda i=i: calibration.infer(deployed, request, seed=args.seed + i))
                if i >= args.warmup:
                    for key in ("embed", "llm", "denoise_loop"):
                        timings[key].append(parts.get(key, float("nan")))
                    timings["e2e"].append(e2e)
    finally:
        watches.restore()
    med = {k: float(np.median(v)) for k, v in timings.items()}
    logger.info(
        "E2E %.1f ms (%.1f Hz) | embed %s (llm %s) | denoise loop %s ms",
        med["e2e"],
        1000 / med["e2e"],
        _fmt(med["embed"]),
        _fmt(med["llm"]),
        _fmt(med["denoise_loop"]),
    )
    return timings


def _fmt(ms: float) -> str:
    return "—" if np.isnan(ms) else f"{ms:.1f}"


def _json_ms(ms: float) -> float | None:
    return None if np.isnan(ms) else ms


def markdown_table(results: dict[str, dict[str, list[float]]], device: str, num_steps: int) -> str:
    lines = [
        f"Device: {device} | {num_steps} denoising steps | median ms",
        "",
        "| Arm | Embed (incl. LLM) | LLM | Denoise loop | E2E | Hz | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    base = None
    for label, data in results.items():
        med = {k: float(np.median(v)) for k, v in data.items()}
        if base is None:
            base = med["e2e"]
        lines.append(
            f"| {label} | {_fmt(med['embed'])} | {_fmt(med['llm'])} | {_fmt(med['denoise_loop'])} | "
            f"{med['e2e']:.1f} | {1000 / med['e2e']:.1f} | {base / med['e2e']:.2f}x |"
        )
    return "\n".join(lines)


def main(args: BenchmarkConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("benchmarking needs a CUDA device")

    deployed = calibration.load_policy(
        args.checkpoint_dir, arm_key=args.arm_key, dataset_key=args.dataset_key, device=args.device
    )
    dataset = calibration.load_dataset(args.dataset_path, episodes=calibration.parse_episodes(args.episodes))
    episode_ids, _lengths = calibration.episode_table(dataset)
    request = calibration.build_observations(dataset, [calibration.SampleId(episode_ids[0], 0)])[0]
    model = model_of(deployed)
    num_steps = int(getattr(model.config, "num_inference_timesteps", 50))
    device = torch.cuda.get_device_name(0)
    logger.info("%s | evo_1 | horizon %d | %d denoising steps", device, model.horizon, num_steps)

    results: dict[str, dict[str, list[float]]] = {}
    if not args.skip_eager:
        logger.info("arm: PyTorch Eager")
        results["PyTorch Eager"] = time_components(deployed, request, args)

    components: dict[str, list[str]] = {}
    for label, engine_dir in _parse_arms(args.arms):
        logger.info("arm: %s (%s)", label, engine_dir)
        installed = install_engines(deployed, engine_dir)
        components[label] = sorted(installed.engines)
        try:
            results[label] = time_components(deployed, request, args)
        finally:
            installed.remove()
        torch.cuda.empty_cache()

    print(markdown_table(results, device, num_steps))

    report = {
        "device": device,
        "checkpoint_dir": args.checkpoint_dir,
        "num_steps": num_steps,
        "num_iterations": args.num_iterations,
        "warmup": args.warmup,
        "arms": {
            label: {
                "components": components.get(label, []),
                "median_ms": {k: _json_ms(float(np.median(v))) for k, v in data.items()},
                "raw_ms": {k: [_json_ms(float(x)) for x in v] for k, v in data.items()},
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
