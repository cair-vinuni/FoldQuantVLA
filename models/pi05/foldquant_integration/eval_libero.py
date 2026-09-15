# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""LIBERO success-rate sweep through upstream's own client, with a resumable summary.

Upstream evaluates Pi0.5 on LIBERO with a websocket client
(``examples/libero/main.py``, which runs in its own Python 3.8 environment
with LIBERO and robosuite) talking to ``scripts/serve_policy.py``, one suite
per run, results in the client's log. This module runs exactly that pairing
for several suites unattended: it starts :mod:`.serve` for the arm, waits for
the port, runs the unmodified upstream client on each suite with the
client environment's interpreter (``--client-python``), reads the final
``Total success rate`` / ``Total episodes`` lines off the client log, and
records them in ``<output>/summary.json`` per suite — an interrupted sweep
resumes where it stopped.

The client saves a replay video of every episode; they are directed to
``<output>/videos/<suite>/``.

Example::

    MUJOCO_GL=egl python -m foldquant_integration.eval_libero \\
        --checkpoint-dir <ckpt> --engine-dir exports/pi05_w4a4/engines \\
        --client-python examples/libero/.venv/bin/python --output results/pi05_w4a4
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import logging
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any

from foldquant.provenance import public_path
import tyro

from ._upstream import LIBERO_TRAIN_CONFIG
from ._upstream import UPSTREAM_ROOT

logger = logging.getLogger("foldquant.pi05.eval")

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

_RATE = re.compile(r"Total success rate: ([0-9.]+)")
_EPISODES = re.compile(r"Total episodes: (\d+)")


@dataclass
class EvalConfig:
    checkpoint_dir: str
    client_python: str
    """Interpreter of the LIBERO client environment (``examples/libero/README.md``, "Without Docker")."""

    engine_dir: str | None = None
    """FoldQuant engine directory; omit to score the bf16 PyTorch policy."""

    output: str = "results/pi05"
    """Directory for ``summary.json``, the per-suite client logs and the replay videos."""

    config: str = LIBERO_TRAIN_CONFIG
    suites: list[str] = field(default_factory=lambda: list(SUITES))
    num_trials_per_task: int = 50
    """Rollouts per task (upstream's default; 10 tasks per suite)."""

    replan_steps: int = 5
    seed: int = 7
    """The client's seed (object positions follow it even from a fixed initial state)."""

    port: int = 8000
    server_timeout_s: float = 900.0
    """How long to wait for the server to accept connections (engine + plugin loading)."""


def _wait_for_port(port: int, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited early with code {proc.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(2.0)
    raise TimeoutError(f"policy server did not open port {port} within {timeout_s:.0f}s")


def _start_server(args: EvalConfig, log_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "foldquant_integration.serve",
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--config",
        args.config,
        "--port",
        str(args.port),
    ]
    if args.engine_dir:
        cmd += ["--engine-dir", args.engine_dir]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(UPSTREAM_ROOT), env.get("PYTHONPATH", "")) if p)
    with open(log_path, "w") as log:
        # Popen inherits the descriptor; the parent's handle can close once the child holds it.
        proc = subprocess.Popen(cmd, cwd=str(UPSTREAM_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    logger.info("policy server pid %d (log %s)", proc.pid, log_path)
    _wait_for_port(args.port, proc, args.server_timeout_s)
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _run_suite(args: EvalConfig, suite: str, out: Path) -> dict[str, Any]:
    log_path = out / f"{suite}.log"
    cmd = [
        args.client_python,
        str(UPSTREAM_ROOT / "examples" / "libero" / "main.py"),
        "--args.host",
        "127.0.0.1",
        "--args.port",
        str(args.port),
        "--args.task-suite-name",
        suite,
        "--args.num-trials-per-task",
        str(args.num_trials_per_task),
        "--args.replan-steps",
        str(args.replan_steps),
        "--args.seed",
        str(args.seed),
        "--args.video-out-path",
        str(out / "videos" / suite),
    ]
    env = dict(os.environ)
    libero = UPSTREAM_ROOT / "third_party" / "libero"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(libero), env.get("PYTHONPATH", "")) if p)
    t0 = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.call(cmd, cwd=str(UPSTREAM_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    text = log_path.read_text()
    rate = _RATE.findall(text)
    episodes = _EPISODES.findall(text)
    if rc != 0 or not rate or not episodes:
        raise RuntimeError(f"{suite}: client exited {rc} without a final result; see {log_path}")
    n = int(episodes[-1])
    sr = float(rate[-1])
    return {
        "success_rate": sr,
        "successes": round(sr * n),
        "episodes": n,
        "seconds": round(time.time() - t0, 1),
        "log": log_path.name,
    }


def main(args: EvalConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    unknown = [s for s in args.suites if s not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suites {unknown}; choose from {list(SUITES)}")
    if not Path(args.client_python).is_file():
        raise SystemExit(f"--client-python {args.client_python} is not a file")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary: dict[str, Any] = (
        json.loads(summary_path.read_text())
        if summary_path.is_file()
        else {
            "checkpoint_dir": public_path(args.checkpoint_dir),
            "engine_dir": public_path(args.engine_dir),
            "config": args.config,
            "num_trials_per_task": args.num_trials_per_task,
            "replan_steps": args.replan_steps,
            "seed": args.seed,
            "suites": {},
        }
    )
    todo = [s for s in args.suites if s not in summary["suites"]]
    if not todo:
        logger.info("all requested suites already in %s", summary_path)
    else:
        server = _start_server(args, out / "server.log")
        try:
            for suite in todo:
                logger.info("suite %s", suite)
                summary["suites"][suite] = _run_suite(args, suite, out)
                summary_path.write_text(json.dumps(summary, indent=2))
                r = summary["suites"][suite]
                logger.info("%s: %d/%d = %.3f", suite, r["successes"], r["episodes"], r["success_rate"])
        finally:
            _stop_server(server)

    done = [s for s in args.suites if s in summary["suites"]]
    total_s = sum(summary["suites"][s]["successes"] for s in done)
    total_n = sum(summary["suites"][s]["episodes"] for s in done)
    summary["overall"] = {
        "suites": done,
        "successes": total_s,
        "episodes": total_n,
        "success_rate": (total_s / total_n) if total_n else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("overall %d/%d = %s", total_s, total_n, summary["overall"]["success_rate"])
    return summary


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
