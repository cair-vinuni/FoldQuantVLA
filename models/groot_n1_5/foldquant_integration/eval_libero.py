# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""LIBERO success-rate sweep, in process, with a resumable summary.

Upstream evaluates N1.5 on LIBERO through a ZMQ client
(``examples/Libero/eval/run_libero_eval.py``) talking to
``scripts/inference_service.py``, one suite per run, results in a text log.
The reference way to score a FoldQuant arm is exactly that, with
:mod:`.serve` on the server side. This module is the same rollout loop
without the socket, for sweeps that cover several suites and arms
unattended:

* the observation and action conversion is upstream's own ``GR00TPolicy``
  class, subclassed only to hold the in-process policy instead of the ZMQ
  client — the dictionaries the model sees are byte-identical to the served
  path;
* the environment loop (``env.reset``, ``set_init_state``, the
  ``num_steps_wait`` no-op steps, the per-suite ``max_steps``) follows
  ``eval_libero`` in ``run_libero_eval.py``;
* results land in ``<output>/summary.json`` per task, so an interrupted
  sweep resumes where it stopped, and rollout videos are not written.

Upstream defaults to 5 trials per task; ``--n-episodes`` defaults to 20 here
(200 episodes per suite). Needs LIBERO (``libero`` + ``robosuite``) installed
in the environment, as upstream's client does.

Example::

    MUJOCO_GL=egl python -m foldquant_integration.eval_libero \\
        --model-path <ckpt> --embodiment-tag new_embodiment --denoising-steps 8 \\
        --engine-dir exports/n15_w4a4/engines --output results/n15_w4a4
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from . import calibration
from ._upstream import LIBERO_DATA_CONFIG, ensure_libero_on_path, ensure_upstream_on_path
from .runtime import install_engines

logger = logging.getLogger("foldquant.groot_n1_5.eval")

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

#: Episode step budgets, as upstream's ``run_libero_eval.py`` sets them per suite.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 600,
    "libero_10": 1000,
    "libero_90": 400,
}


@dataclass
class EvalConfig:
    model_path: str
    """Checkpoint directory (the PyTorch policy; also the residual half of a FoldQuant arm)."""

    output: str
    """Directory for ``summary.json`` and the per-task log."""

    engine_dir: str | None = None
    """FoldQuant engine directory. Omit for the bf16 PyTorch arm."""

    embodiment_tag: str | None = None
    data_config: str = LIBERO_DATA_CONFIG
    denoising_steps: int | None = None
    """Flow-matching steps (upstream serves the LIBERO checkpoints with 8)."""

    suites: list[str] = field(default_factory=lambda: list(SUITES))
    n_episodes: int = 20
    """Episodes per task (upstream's client defaults to 5); 10 tasks per suite."""

    num_steps_wait: int = 10
    """No-op steps after reset while dropped objects settle, as upstream."""

    resolution: int = 256
    tasks: list[str] | None = None
    """Restrict to these task names."""


def _load_summary(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text())
    return {"tasks": {}}


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    per_suite: dict[str, dict[str, int]] = {}
    for entry in summary["tasks"].values():
        if entry.get("status") != "ok":
            continue
        agg = per_suite.setdefault(entry["suite"], {"successes": 0, "num_episodes": 0})
        agg["successes"] += entry["successes"]
        agg["num_episodes"] += entry["num_episodes"]
    total_s = sum(v["successes"] for v in per_suite.values())
    total_n = sum(v["num_episodes"] for v in per_suite.values())
    summary["per_suite"] = {
        k: {**v, "success_rate": v["successes"] / v["num_episodes"]} for k, v in per_suite.items()
    }
    summary["totals"] = {
        "tasks_completed": sum(1 for e in summary["tasks"].values() if e.get("status") == "ok"),
        "tasks_failed": sum(1 for e in summary["tasks"].values() if e.get("status") != "ok"),
        "successes": total_s,
        "num_episodes": total_n,
        "success_rate": (total_s / total_n) if total_n else None,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(path)


def _make_wrapper(policy):
    """Upstream's ``GR00TPolicy`` LIBERO wrapper around an in-process ``Gr00tPolicy``."""
    ensure_libero_on_path()
    from examples.Libero.eval.run_libero_eval import GR00TPolicy

    class InProcessGR00TPolicy(GR00TPolicy):
        def __init__(self, gr00t_policy):
            # The base class opens a ZMQ client (lazy connect, no traffic); replace it.
            super().__init__(headless=True)
            self.policy = gr00t_policy

    return InProcessGR00TPolicy(policy)


def run_task(wrapper, suite: str, task_id: int, args: EvalConfig) -> dict[str, Any]:
    from libero.libero import benchmark

    from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env

    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, resolution=args.resolution)
    max_steps = MAX_STEPS[suite]
    successes = 0
    steps: list[int] = []
    errors = 0
    try:
        for episode_idx in range(args.n_episodes):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            done = False
            try:
                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, _reward, done, _info = env.step(get_libero_dummy_action())
                        t += 1
                        continue
                    action = wrapper.get_action(obs, task.language)
                    obs, _reward, done, _info = env.step(action.tolist())
                    if done:
                        break
                    t += 1
            except Exception as exc:  # noqa: BLE001 — upstream counts the episode as failed and moves on
                logger.warning("%s/%s episode %d: %s", suite, task.name, episode_idx, exc)
                errors += 1
                done = False
            successes += int(bool(done))
            steps.append(t)
    finally:
        env.close()
    return {
        "suite": suite,
        "task_id": task_id,
        "name": task.name,
        "language": task_description,
        "num_episodes": args.n_episodes,
        "successes": successes,
        "success_rate": successes / args.n_episodes,
        "episode_errors": errors,
        "mean_steps": float(np.mean(steps)) if steps else None,
        "max_steps": max_steps,
    }


def main(args: EvalConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    os.environ.setdefault("MUJOCO_GL", "egl")
    ensure_upstream_on_path()
    from libero.libero import benchmark

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = _load_summary(summary_path)
    summary["arm"] = {
        "model_path": args.model_path,
        "engine_dir": args.engine_dir,
        "embodiment_tag": args.embodiment_tag,
        "data_config": args.data_config,
        "n_episodes": args.n_episodes,
        "num_steps_wait": args.num_steps_wait,
        "resolution": args.resolution,
    }

    benchmark_dict = benchmark.get_benchmark_dict()
    todo = []
    for suite in args.suites:
        for task_id, name in enumerate(benchmark_dict[suite]().get_task_names()):
            key = f"{suite}/{name}"
            if args.tasks and name not in args.tasks:
                continue
            if summary["tasks"].get(key, {}).get("status") == "ok":
                continue
            todo.append((suite, task_id, key))
    logger.info("%d tasks to run (%d already in %s)", len(todo), len(summary["tasks"]), summary_path)
    if not todo:
        _write_summary(summary_path, summary)
        return summary

    policy = calibration.load_policy(
        args.model_path,
        args.embodiment_tag,
        "cuda",
        data_config=args.data_config,
        denoising_steps=args.denoising_steps,
    )
    summary["arm"]["denoising_steps"] = int(policy.denoising_steps)
    summary["arm"]["embodiment_tag"] = policy.embodiment_tag.value
    installed = install_engines(policy, args.engine_dir) if args.engine_dir else None
    if installed is not None:
        logger.info("engines installed: %s", ", ".join(sorted(installed.engines)))
    wrapper = _make_wrapper(policy)

    try:
        for suite, task_id, key in todo:
            t0 = time.time()
            try:
                entry = run_task(wrapper, suite, task_id, args)
                entry["status"] = "ok"
            except Exception as exc:
                logger.exception("%s failed", key)
                entry = {"suite": suite, "task_id": task_id, "status": "error", "error": repr(exc)}
            entry["seconds"] = round(time.time() - t0, 1)
            summary["tasks"][key] = entry
            _write_summary(summary_path, summary)
            if entry["status"] == "ok":
                logger.info("%s: %d/%d in %.0fs", key, entry["successes"], entry["num_episodes"], entry["seconds"])
    finally:
        if installed is not None:
            installed.remove()
    logger.info("totals: %s", summary["totals"])
    return summary


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
