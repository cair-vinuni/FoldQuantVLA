# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""LIBERO success rate for a FoldQuant engine directory, through upstream's own rollout.

Nothing here evaluates anything: the suites, the environment wrapper, the
episode loop, the action queue and the success definition are upstream's
(``lerobot.envs.factory.make_env`` over ``LiberoEnv`` and
``lerobot.scripts.lerobot_eval.eval_policy_all``). This module only adds what a
paper sweep needs around them — the engines installed into the policy for the
duration, one suite at a time, and a resume-safe summary so an interrupted
sweep continues instead of restarting.

Engines pin batch 1, so the vector env is built with one environment per task;
the PyTorch arms could batch, but running them the same way keeps the arms
comparable.

``--engine-dir`` may be omitted to evaluate the bf16 PyTorch policy — the
reference arm — through exactly this driver.

Example::

    MUJOCO_GL=egl python -m foldquant_integration.eval_libero \\
        --engine-dir exports/smolvla_w4a4/engines \\
        --output results/smolvla_w4a4 --n-episodes 50
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tyro

from . import calibration
from ._upstream import LIBERO_CHECKPOINT, LIBERO_SUITES
from .runtime import install_engines

logger = logging.getLogger("foldquant.smolvla.eval")


@dataclass
class EvalConfig:
    output: str
    """Directory for ``summary.json`` (and upstream's rollout videos)."""

    checkpoint: str = LIBERO_CHECKPOINT
    engine_dir: str | None = None
    """FoldQuant engine directory; omitted evaluates the bf16 PyTorch policy."""

    suites: list[str] = field(default_factory=lambda: list(LIBERO_SUITES))
    """LIBERO suites to run, in order."""

    n_episodes: int = 50
    """Episodes per task, as upstream's evaluator counts them."""

    task_ids: list[int] = field(default_factory=list)
    """Restrict to these task indices within each suite (default: every task)."""

    start_seed: int = 7
    """Seed upstream hands the first episode; each episode advances it."""

    videos: bool = False
    """Keep upstream's rollout videos under ``<output>/videos/<suite>``."""

    device: str = "cuda"


def _summary_path(output: Path) -> Path:
    return output / "summary.json"


def _load_summary(output: Path) -> dict[str, Any]:
    path = _summary_path(output)
    if path.is_file():
        return json.loads(path.read_text())
    return {"suites": {}}


def _env_config(suite: str, args: EvalConfig):
    from lerobot.envs.configs import LiberoEnv

    return LiberoEnv(task=suite, task_ids=list(args.task_ids) or None)


def run_suite(deployed, suite: str, args: EvalConfig, output: Path) -> dict[str, Any]:
    """One suite through upstream's evaluator; returns its slice of the summary."""
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all

    env_cfg = _env_config(suite, args)
    envs = make_env(env_cfg, n_envs=1)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=deployed.policy.config)
    videos_dir = (output / "videos" / suite) if args.videos else None
    t0 = time.time()
    try:
        info = eval_policy_all(
            envs=envs,
            policy=deployed.policy,
            env_preprocessor=env_pre,
            env_postprocessor=env_post,
            preprocessor=deployed.preprocessor,
            postprocessor=deployed.postprocessor,
            n_episodes=args.n_episodes,
            videos_dir=videos_dir,
            start_seed=args.start_seed,
        )
    finally:
        for group in envs.values():
            for env in group.values():
                env.close()
    # ``eval_policy_all`` returns {"per_task", "per_group", "overall"}; the
    # suite-wide numbers are under "overall" and its "n_episodes" counts the
    # episodes that actually ran, which is what a caller should trust over the
    # requested count.
    overall = info["overall"]

    def _num(key: str) -> float | None:
        # Upstream returns NaN for an empty accumulator. NaN is not valid JSON and
        # reads as a score; None says "did not run", which is the honest record.
        value = overall.get(key)
        return None if value is None or math.isnan(value) else float(value)

    return {
        "pc_success": _num("pc_success"),
        "avg_sum_reward": _num("avg_sum_reward"),
        "avg_max_reward": _num("avg_max_reward"),
        "n_episodes": overall.get("n_episodes", 0),
        "n_episodes_requested": args.n_episodes,
        "seconds": round(time.time() - t0, 1),
        "per_task": info.get("per_task", []),
    }


def main(args: EvalConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(output)
    summary.setdefault("suites", {})
    summary["checkpoint"] = args.checkpoint
    summary["engine_dir"] = args.engine_dir
    summary["n_episodes"] = args.n_episodes

    deployed = calibration.load_policy(args.checkpoint, device=args.device, compile=False)
    installed = None
    if args.engine_dir:
        installed = install_engines(deployed, args.engine_dir)
        summary["components"] = sorted(installed.engines)
        logger.info("engines installed: %s", ", ".join(summary["components"]))
    else:
        summary["components"] = []
        logger.info("no engine directory: evaluating the bf16 PyTorch policy")

    try:
        for suite in args.suites:
            if suite in summary["suites"]:
                logger.info("%s: already in %s, skipping", suite, _summary_path(output))
                continue
            logger.info("%s: starting (%d episodes per task)", suite, args.n_episodes)
            summary["suites"][suite] = run_suite(deployed, suite, args, output)
            _summary_path(output).write_text(json.dumps(summary, indent=2))
            rate = summary["suites"][suite]["pc_success"]
            logger.info("%s: %s success", suite, "n/a" if rate is None else f"{rate:.1f}%")
    finally:
        if installed is not None:
            installed.remove()

    rates = [s["pc_success"] for s in summary["suites"].values() if s.get("pc_success") is not None]
    summary["pooled_pc_success"] = sum(rates) / len(rates) if rates else None
    _summary_path(output).write_text(json.dumps(summary, indent=2))
    logger.info(
        "pooled over %d suites: %.1f%%",
        len(rates),
        summary["pooled_pc_success"] if summary["pooled_pc_success"] is not None else float("nan"),
    )
    return summary


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
