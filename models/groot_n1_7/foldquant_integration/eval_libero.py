# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""LIBERO success rate of a checkpoint, in PyTorch or on a FoldQuant engine directory.

The rollout itself is upstream's (``gr00t.eval.rollout_policy``): the same
``MultiStepWrapper`` (8-step chunks, 504-step cap, terminate on success), the
same unseeded ``LiberoEnv.reset()``, the same success definition. This tool
only adds what a paper sweep needs around it — the plugin library is loaded
before any engine is deserialised, every task of every requested suite is
visited, and one ``summary.json`` per output directory records per-task
successes so an interrupted run resumes where it stopped.

Videos are not recorded (upstream's ``run_gr00t_sim_policy`` writes a video
of every episode under ``/tmp``; ``run_rollout_gymnasium_policy`` is called
directly with ``video_dir=None`` instead).

Example — the W4A4 arm on all four suites, 20 episodes per task::

    MUJOCO_GL=egl python -m foldquant_integration.eval_libero \\
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
        --engine-dir exports/n17_w4a4/engines --n-envs 1 \\
        --output exports/n17_w4a4/libero

Engines built by :mod:`.build_engines` pin the batch to the captured batch
(1), so TensorRT arms run ``--n-envs 1``. PyTorch arms may batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from foldquant.runtime.plugins import load_plugins
import tyro

from ._upstream import MANIFEST_NAME, ensure_libero_on_path


logger = logging.getLogger("foldquant.groot_n1_7.eval_libero")

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass
class EvalConfig:
    model_path: str
    """Checkpoint directory (the PyTorch policy; also the residual half of a TRT arm)."""

    output: str
    """Directory for ``summary.json`` and the per-task log."""

    engine_dir: Optional[str] = None
    """FoldQuant engine directory. Omit for the bf16 PyTorch arm."""

    suites: List[str] = field(default_factory=lambda: list(SUITES))
    n_episodes: int = 20
    """Episodes per task; 10 tasks per suite."""

    n_envs: int = 1
    max_episode_steps: int = 504
    n_action_steps: int = 8
    trt_mode: str = "n17_full_pipeline"
    tasks: Optional[List[str]] = None
    """Restrict to these task names (``libero_sim/`` prefix optional)."""


def _load_summary(path: Path) -> Dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text())
    return {"tasks": {}}


def _write_summary(path: Path, summary: Dict[str, Any]) -> None:
    per_suite: Dict[str, Dict[str, int]] = {}
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


def list_tasks(suites: List[str]) -> List[Dict[str, str]]:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    tasks = []
    for suite in suites:
        for name in benchmark_dict[suite]().get_task_names():
            tasks.append({"suite": suite, "name": name, "env_name": f"libero_sim/{name}"})
    return tasks


def main(args: EvalConfig) -> Dict[str, Any]:
    # First, before anything reaches a LIBERO import: this module has two
    # such sites (list_tasks and the env registry) and hooking only the
    # later one leaves the first failing with the path correctly set.
    ensure_libero_on_path()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = _load_summary(summary_path)
    summary.update(
        {
            "model_path": args.model_path,
            "engine_dir": args.engine_dir,
            "n_episodes": args.n_episodes,
            "n_envs": args.n_envs,
            "max_episode_steps": args.max_episode_steps,
            "n_action_steps": args.n_action_steps,
        }
    )

    if args.engine_dir:
        manifest = json.loads((Path(args.engine_dir) / MANIFEST_NAME).read_text())
        summary["schemes"] = manifest["schemes"]
        load_plugins(manifest["plugin_libs"])

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.rollout_policy import (
        MultiStepConfig,
        TrtMode,
        VideoConfig,
        WrapperConfigs,
        create_gr00t_sim_policy,
        run_rollout_gymnasium_policy,
    )
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    register_libero_envs()
    tasks = list_tasks(args.suites)
    if args.tasks:
        wanted = {t.removeprefix("libero_sim/") for t in args.tasks}
        tasks = [t for t in tasks if t["name"] in wanted]
    pending = [t for t in tasks if summary["tasks"].get(t["env_name"], {}).get("status") != "ok"]
    logger.info("%d tasks, %d pending", len(tasks), len(pending))
    if not pending:
        _write_summary(summary_path, summary)
        return summary

    policy = create_gr00t_sim_policy(
        args.model_path,
        EmbodimentTag.LIBERO_PANDA,
        trt_engine_path=args.engine_dir or "",
        trt_mode=TrtMode(args.trt_mode),
    )
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=None, max_episode_steps=args.max_episode_steps),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
            terminate_on_success=True,
        ),
    )

    for task in pending:
        t0 = time.time()
        try:
            _, successes, _ = run_rollout_gymnasium_policy(
                env_name=task["env_name"],
                policy=policy,
                wrapper_configs=wrapper_configs,
                n_episodes=args.n_episodes,
                n_envs=args.n_envs,
            )
            successes = [bool(s) for s in successes[: args.n_episodes]]
            entry = {
                "status": "ok",
                "suite": task["suite"],
                "successes": int(sum(successes)),
                "num_episodes": len(successes),
                "episode_successes": successes,
                "seconds": round(time.time() - t0, 1),
            }
            logger.info(
                "%s: %d/%d in %.0fs",
                task["name"],
                entry["successes"],
                entry["num_episodes"],
                entry["seconds"],
            )
        except Exception as exc:  # keep sweeping; the task is retried on resume
            logger.exception("%s failed", task["name"])
            entry = {"status": "failed", "suite": task["suite"], "error": repr(exc)}
        summary["tasks"][task["env_name"]] = entry
        _write_summary(summary_path, summary)

    totals = summary["totals"]
    logger.info(
        "done: %d/%d = %s over %d tasks (%d failed)",
        totals["successes"],
        totals["num_episodes"],
        f"{totals['success_rate']:.4f}" if totals["success_rate"] is not None else "-",
        totals["tasks_completed"],
        totals["tasks_failed"],
    )
    return summary


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
