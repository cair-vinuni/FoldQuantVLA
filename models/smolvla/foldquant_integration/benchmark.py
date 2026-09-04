# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Component latency of the bf16 PyTorch policy and of FoldQuant engine directories.

Upstream LeRobot reports no per-component timing for an inference call, so
this module carries its own timer: one observation from the dataset,
``--warmup`` untimed calls, then ``--num-iterations`` timed ones,
CUDA-synchronised, medians reported. Per iteration it times the pieces of an
inference where they run:

* ``preprocess``   — the checkpoint's processor pipeline (rename, normalise,
  tokenise, host-to-device);
* ``embed_prefix`` — SigLIP on the cameras, the prompt embedding and the state
  token;
* ``prefix_llm``   — the SmolVLM2 prefix pass that fills the KV cache;
* ``denoise_loop`` — every ``denoise_step`` of the Euler loop, summed;
* ``e2e``          — a separate, whole preprocess -> chunk -> postprocess call.

The three model pieces are timed by wrapping the bound methods
``sample_actions`` calls (``embed_prefix``, ``vlm_with_expert.forward`` for the
prefix branch, ``denoise_step``) for the duration of one ``sample_actions``;
the e2e call runs unwrapped. Every arm — PyTorch Eager and each FoldQuant
engine directory — goes through the same loop with
:func:`.runtime.install_engines` swapping the engines in.

Unlike openpi, upstream serves SmolVLA eagerly (``compile_model`` defaults to
False), so the eager arm *is* the deployed reference. ``--compiled`` adds
``torch.compile(sample_actions, mode=config.compile_mode)`` as an extra arm,
timed e2e only since the compiled graph inlines the pieces the component
wrappers would time.

Example::

    python -m foldquant_integration.benchmark --checkpoint <ckpt> --dataset-path <LeRobot LIBERO> \\
        --arms w4a4=exports/smolvla_w4a4/engines w8a8=exports/smolvla_w8a8/engines
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
from ._upstream import LIBERO_CHECKPOINT
from .runtime import install_engines, model_of

logger = logging.getLogger("foldquant.smolvla.benchmark")

COMPONENT_ORDER = ("preprocess", "embed_prefix", "prefix_llm", "denoise_loop", "e2e")


@dataclass
class BenchmarkConfig:
    dataset_path: str
    """Dataset the benchmark observation is read from (first episode, step 0)."""

    checkpoint: str = LIBERO_CHECKPOINT
    arms: list[str] = field(default_factory=list)
    """Engine directories to time, as ``LABEL=DIR`` (or a bare ``DIR``, labelled by its parent's name)."""

    num_iterations: int = 20
    warmup: int = 5
    seed: int = 42
    episodes: str = ""
    """Optional episodes to restrict the dataset load to, as indices or ranges (``0-149``, ``0,3,7``)."""

    skip_eager: bool = False
    """Skip the PyTorch Eager arm (the speedup table then has no baseline)."""

    compiled: bool = False
    """Also time ``torch.compile(sample_actions, config.compile_mode)``, e2e only. Not upstream's
    serving default for SmolVLA — ``compile_model`` ships False — so it is off unless asked for."""

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


_MISSING = object()


class _Stopwatches:
    """Instance-level timing wrappers around bound methods, summed per bucket; ``restore()`` undoes them."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self._undo: list[Callable[[], None]] = []

    def reset(self) -> None:
        self.ms = {}

    def wrap(self, obj, name: str, bucket: str, only: Callable[..., bool] | None = None) -> None:
        previous = obj.__dict__.get(name, _MISSING)
        fn = getattr(obj, name)

        def wrapper(*a, **kw):
            if only is not None and not only(*a, **kw):
                return fn(*a, **kw)
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


def _is_prefix_call(*_a, **kw) -> bool:
    embeds = kw.get("inputs_embeds")
    return embeds is not None and embeds[1] is None


def time_components(
    deployed, observation: dict[str, Any], args: BenchmarkConfig, *, components: bool = True
) -> dict[str, list[float]]:
    """Per-iteration timings by bucket; ``components=False`` times ``preprocess`` and ``e2e`` only."""
    policy = deployed.policy
    model = model_of(deployed)
    device = next(policy.parameters()).device
    timings: dict[str, list[float]] = {k: [] for k in COMPONENT_ORDER}
    state: dict[str, Any] = {}
    watches = _Stopwatches()

    def _preprocess():
        state["batch"] = deployed.preprocessor(dict(observation))

    def _watch():
        watches.wrap(model, "embed_prefix", "embed_prefix")
        watches.wrap(model.vlm_with_expert, "forward", "prefix_llm", only=_is_prefix_call)
        watches.wrap(model, "denoise_step", "denoise_loop")

    def _e2e(noise: torch.Tensor):
        batch = deployed.preprocessor(dict(observation))
        deployed.postprocessor(policy.predict_action_chunk(batch, noise=noise))

    try:
        with torch.inference_mode():
            for i in range(args.warmup + args.num_iterations):
                noise = calibration.action_noise(deployed, args.seed + i).to(device)
                policy.reset()
                tpre = _timed(_preprocess)
                parts: dict[str, float] = {}
                if components:
                    _watch()
                    torch.manual_seed(args.seed + i)
                    policy.predict_action_chunk(dict(state["batch"]), noise=noise)
                    parts = dict(watches.ms)
                    watches.reset()
                    # The whole call is timed with the wrappers off, so their synchronisations do not count.
                    watches.restore()
                policy.reset()
                torch.manual_seed(args.seed + i)
                e2e = _timed(lambda noise=noise: _e2e(noise))
                if i >= args.warmup:
                    timings["preprocess"].append(tpre)
                    timings["embed_prefix"].append(parts.get("embed_prefix", float("nan")))
                    timings["prefix_llm"].append(parts.get("prefix_llm", float("nan")))
                    timings["denoise_loop"].append(parts.get("denoise_loop", float("nan")))
                    timings["e2e"].append(e2e)
    finally:
        watches.restore()
    med = {k: float(np.median(v)) for k, v in timings.items()}
    logger.info(
        "E2E %.1f ms (%.1f Hz) | preprocess %.1f | embed %s | prefix LLM %s | denoise loop %s ms",
        med["e2e"],
        1000 / med["e2e"],
        med["preprocess"],
        _fmt(med["embed_prefix"]),
        _fmt(med["prefix_llm"]),
        _fmt(med["denoise_loop"]),
    )
    return timings


def _fmt(ms: float) -> str:
    return "—" if np.isnan(ms) else f"{ms:.1f}"


class _Compiled:
    """Temporarily serve ``sample_actions`` through ``torch.compile(..., mode)``."""

    def __init__(self, model, mode: str) -> None:
        self._model = model
        self._mode = mode
        self._previous: Any = _MISSING

    def __enter__(self) -> _Compiled:
        self._previous = self._model.__dict__.get("sample_actions", _MISSING)
        self._model.sample_actions = torch.compile(self._model.sample_actions, mode=self._mode)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._previous is _MISSING:
            self._model.__dict__.pop("sample_actions", None)
        else:
            self._model.sample_actions = self._previous
        torch._dynamo.reset()  # noqa: SLF001


def _json_ms(ms: float) -> float | None:
    return None if np.isnan(ms) else ms


def markdown_table(results: dict[str, dict[str, list[float]]], device: str, num_steps: int) -> str:
    lines = [
        f"Device: {device} | {num_steps} denoising steps | median ms",
        "",
        "| Arm | Preprocess | Embed | Prefix LLM | Denoise loop | E2E | Hz | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = None
    for label, data in results.items():
        med = {k: float(np.median(v)) for k, v in data.items()}
        if base is None:
            base = med["e2e"]
        lines.append(
            f"| {label} | {med['preprocess']:.1f} | {_fmt(med['embed_prefix'])} | {_fmt(med['prefix_llm'])} | "
            f"{_fmt(med['denoise_loop'])} | {med['e2e']:.1f} | {1000 / med['e2e']:.1f} | {base / med['e2e']:.2f}x |"
        )
    return "\n".join(lines)


def main(args: BenchmarkConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("benchmarking needs a CUDA device")

    deployed = calibration.load_policy(args.checkpoint, device=args.device, compile=False)
    dataset = calibration.load_dataset(args.dataset_path, episodes=calibration.parse_episodes(args.episodes))
    episode_ids, _lengths, _starts = calibration.episode_table(dataset)
    observation = calibration.build_observations(
        deployed, dataset, [calibration.SampleId(episode_ids[0], 0)]
    )[0]
    model = model_of(deployed)
    num_steps = int(model.config.num_steps)
    device = torch.cuda.get_device_name(0)
    logger.info(
        "%s | smolvla | chunk %d | %d denoising steps",
        device,
        model.config.chunk_size,
        num_steps,
    )

    results: dict[str, dict[str, list[float]]] = {}
    if not args.skip_eager:
        logger.info("arm: PyTorch Eager")
        results["PyTorch Eager"] = time_components(deployed, observation, args)
    if args.compiled:
        mode = str(model.config.compile_mode)
        label = f"PyTorch torch.compile({mode})"
        logger.info("arm: %s — e2e only", label)
        with _Compiled(model, mode):
            results[label] = time_components(deployed, observation, args, components=False)

    components: dict[str, list[str]] = {}
    for label, engine_dir in _parse_arms(args.arms):
        logger.info("arm: %s (%s)", label, engine_dir)
        installed = install_engines(deployed, engine_dir)
        components[label] = sorted(installed.engines)
        try:
            results[label] = time_components(deployed, observation, args)
        finally:
            installed.remove()
        torch.cuda.empty_cache()

    print(markdown_table(results, device, num_steps))

    report = {
        "device": device,
        "checkpoint": args.checkpoint,
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
