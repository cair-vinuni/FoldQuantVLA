# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""LIBERO success rate of a checkpoint, in PyTorch or on a FoldQuant engine directory.

The rollout itself is upstream's (``gr00t.eval.rollout_policy``): the same
``MultiStepWrapper`` (8-step chunks, 504-step cap, terminate on success), the
same unseeded ``LiberoEnv.reset()``, the same success definition. This tool
only adds what a paper sweep needs around it: the plugin library is loaded
before any engine is deserialised, every task of every requested suite is
visited, and one ``summary.json`` per output directory records per-task
successes so an interrupted run resumes where it stopped, and only into the same run: the
summary carries a fingerprint of the checkpoint, engines and protocol.

``--protocol p3`` is the paper's closed-loop campaign: episode ``i`` of every
task starts from LIBERO's stored initial state ``i``, ten no-op steps let the
scene settle, 520 environment steps are allowed, eight actions of each chunk
are executed and the episode ends after the chunk that succeeds
(``foldquant.eval_protocol``). ``--protocol upstream`` (the default) is the
release loop above: unseeded random placements, 504 steps.

Videos are not recorded (upstream's ``run_gr00t_sim_policy`` writes a video
of every episode under ``/tmp``; ``run_rollout_gymnasium_policy`` is called
directly with ``video_dir=None`` instead).

Example:  the W4A4 arm on all four suites, 20 episodes per task::

    MUJOCO_GL=egl python -m foldquant_integration.eval_libero \\
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
        --engine-dir exports/n17_w4a4/engines --n-envs 1 \\
        --output exports/n17_w4a4/libero

Engines built by :mod:`.build_engines` pin the batch to the captured batch
(1), so TensorRT arms run ``--n-envs 1``. PyTorch arms may batch.

``--baseline-pack`` rolls out one of the emulated W4A4 comparison arms of
:mod:`.baseline_w4a4` (HoloQ-style / DuQuant-style) in place of an engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from foldquant.eval_protocol import PROTOCOLS, artifact_digest, libero_init_states, prepare_summary, protocol_record
from foldquant.libero_rollout import rollout_task
from foldquant.provenance import public_path
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

    baseline_pack: Optional[str] = None
    """Emulated W4A4 baseline pack (:mod:`.baseline_w4a4`); mutually exclusive with ``--engine-dir``."""

    suites: List[str] = field(default_factory=lambda: list(SUITES))
    n_episodes: Optional[int] = None
    """Episodes per task (default 20); 10 tasks per suite."""

    protocol: str = "upstream"
    """``upstream`` (the release loop, 504 steps, random placements) or ``p3`` (the paper's campaign)."""

    n_envs: int = 1
    max_episode_steps: Optional[int] = None
    """Environment steps per episode; default 504 (upstream) or 520 (p3)."""

    n_action_steps: Optional[int] = None
    """Actions executed per policy call; default 8."""

    seed: Optional[int] = None
    """Simulator seed applied once per task environment; default none (upstream) or 7 (p3)."""

    resume: bool = True
    """Continue an interrupted sweep in ``--output``; refused when it was a different run."""

    trt_mode: str = "n17_full_pipeline"
    tasks: Optional[List[str]] = None
    """Restrict to these task names (``libero_sim/`` prefix optional)."""


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
    if args.protocol not in PROTOCOLS:
        raise SystemExit(f"--protocol {args.protocol}: choose from {sorted(PROTOCOLS)}")
    protocol = PROTOCOLS[args.protocol]
    n_episodes = protocol.resolve(args.n_episodes, "n_episodes", 20)
    max_episode_steps = protocol.resolve(args.max_episode_steps, "max_episode_steps", 504)
    n_action_steps = protocol.resolve(args.n_action_steps, "n_action_steps", 8)
    seed = args.seed if args.seed is not None else protocol.seed
    run = {
        "protocol": protocol_record(protocol, max_episode_steps, n_action_steps, n_episodes, seed),
        "model_path": public_path(args.model_path),
        "model": artifact_digest(args.model_path),
        "engine_dir": public_path(args.engine_dir),
        "engines": artifact_digest(args.engine_dir),
        "baseline_pack": public_path(args.baseline_pack),
        "baseline": artifact_digest(args.baseline_pack),
        "suites": list(args.suites),
        "tasks": sorted(args.tasks) if args.tasks else None,
        "n_envs": args.n_envs,
        "trt_mode": args.trt_mode,
    }
    summary = prepare_summary(summary_path, run, resume=args.resume)
    summary.update(
        {
            "model_path": public_path(args.model_path),
            "engine_dir": public_path(args.engine_dir),
            "baseline_pack": public_path(args.baseline_pack),
            "n_episodes": n_episodes,
            "n_envs": args.n_envs,
            "max_episode_steps": max_episode_steps,
            "n_action_steps": n_action_steps,
            "seed": seed,
        }
    )

    if args.engine_dir and args.baseline_pack:
        raise ValueError("--engine-dir and --baseline-pack are mutually exclusive")
    if args.engine_dir:
        # The float arm comes off upstream's pipeline and carries no FoldQuant manifest;
        # it has no plugin nodes to load libraries for. Reading it unconditionally made
        # eval_libero refuse the one arm that separates TensorRT from quantization.
        manifest_path = Path(args.engine_dir) / MANIFEST_NAME
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            summary["schemes"] = manifest["schemes"]
            load_plugins(manifest["plugin_libs"])
        else:
            summary["schemes"] = {}
            logger.info("%s has no FoldQuant manifest; serving it as a float arm", args.engine_dir)

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
    if args.baseline_pack:
        # The sim wrapper holds the Gr00tPolicy as ``.policy``; the pack replaces the
        # in-scope Linears of its model with the emulation and the step context feeds
        # the per-step DiT tables. Same device as the policy was loaded on.
        from .baselines import apply_pack, install_dit_step_context

        gr00t_policy = policy.policy
        device = next(gr00t_policy.model.parameters()).device
        apply_pack(gr00t_policy.model, args.baseline_pack, backend="fake")
        gr00t_policy.model.to(device=device)
        install_dit_step_context(gr00t_policy.model)
        manifest = gr00t_policy.model.baseline_manifest
        summary["schemes"] = {
            "baseline": manifest.get("method"),
            "rotation_mode": manifest.get("rotation_mode"),
            "llm": f"w4a4-{manifest.get('llm_activation_granularity')}",
            "dit": f"w4a4-{manifest.get('dit_activation_granularity')}",
            "execution": "emulated",
        }
        logger.info("rolling out the EMULATED %s W4A4 baseline from %s", manifest.get("method"), args.baseline_pack)
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=None, max_episode_steps=max_episode_steps),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )
    if protocol.fixed_init_states:
        import gymnasium as gym
        from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        if args.n_envs != 1:
            logger.info("--protocol %s rolls out one environment at a time; ignoring --n-envs %d", protocol.name, args.n_envs)

    for task in pending:
        t0 = time.time()
        try:
            if protocol.fixed_init_states:
                suite_obj = benchmark_dict[task["suite"]]()
                task_id = suite_obj.get_task_names().index(task["name"])
                episodes = rollout_task(
                    policy,
                    lambda name=task["env_name"]: gym.make(name),
                    MultiStepWrapper,
                    libero_init_states(suite_obj, task_id),
                    n_episodes=n_episodes,
                    max_episode_steps=max_episode_steps,
                    n_action_steps=n_action_steps,
                    settle_steps=protocol.settle_steps,
                    seed=seed,
                )
                successes = [bool(e["success"]) for e in episodes]
            else:
                _, successes, _ = run_rollout_gymnasium_policy(
                    env_name=task["env_name"],
                    policy=policy,
                    wrapper_configs=wrapper_configs,
                    n_episodes=n_episodes,
                    n_envs=args.n_envs,
                )
                successes = [bool(s) for s in successes[:n_episodes]]
                episodes = None
            entry = {
                "status": "ok",
                "suite": task["suite"],
                "successes": int(sum(successes)),
                "num_episodes": len(successes),
                "episode_successes": successes,
                "seconds": round(time.time() - t0, 1),
            }
            if episodes is not None:
                entry["episodes"] = episodes
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
