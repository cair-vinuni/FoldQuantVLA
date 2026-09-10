# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""LIBERO success rate for a FoldQuant engine directory, through upstream's own client.

Nothing here evaluates anything: the suites, the environment, the episode loop,
the action horizon, the step budgets and the success definition are upstream's
``LIBERO_evaluation/libero_client_4tasks.py``, run unmodified against
:mod:`.serve`. This module only starts that pair, reads the client's own log
and records the result.

Unlike the other families' drivers this one cannot run a single suite: the
client takes no arguments — its suite list, episode count, horizon and per-suite
step budgets are class attributes of ``Args`` — so it walks all four suites in
one process, as upstream intends, and the summary is written per suite from the
markers the client logs (``Start task suite <name>`` and each
``Task N Summary: k/m Successful``). An interrupted sweep therefore resumes at
the whole run, not at a suite; a finished run is not repeated.

The client needs its own environment (upstream's README builds a Python 3.8
LIBERO env), named with ``--client-python``.

Example::

    MUJOCO_GL=egl python -m foldquant_integration.eval_libero \\
        --checkpoint-dir <ckpt> --engine-dir exports/evo1_w4a4/engines \\
        --client-python <libero venv>/bin/python --output results/evo1_w4a4
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from foldquant.provenance import public_path

from ._upstream import LIBERO_ARM_KEY, LIBERO_CHECKPOINT, LIBERO_CLIENT, LIBERO_DATASET_KEY, UPSTREAM_ROOT

logger = logging.getLogger("foldquant.evo_1.eval")

_SUITE = re.compile(r"Start task suite (\S+?)=")
_TASK = re.compile(r"Task (\d+) Summary: (\d+)/(\d+) Successful")


@dataclass
class EvalConfig:
    client_python: str
    """Interpreter of the LIBERO client environment (upstream's README builds it)."""

    output: str
    """Directory for ``summary.json`` and the server / client logs."""

    checkpoint_dir: str = LIBERO_CHECKPOINT
    engine_dir: str | None = None
    """FoldQuant engine directory; omitted evaluates the bf16 PyTorch model."""

    port: int = 9000
    arm_key: str = LIBERO_ARM_KEY
    dataset_key: str = LIBERO_DATASET_KEY
    server_timeout_s: float = 900.0
    """How long to wait for the served policy to accept connections."""

    device: str = "cuda"


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_port(port: int, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"the server exited with code {proc.returncode} before accepting connections")
        if _port_open(port):
            return
        time.sleep(2.0)
    proc.kill()
    raise SystemExit(f"the server did not accept connections within {timeout_s:.0f}s")


def _start_server(args: EvalConfig, log_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "foldquant_integration.serve",
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--port",
        str(args.port),
        "--arm-key",
        args.arm_key,
        "--dataset-key",
        args.dataset_key,
        "--device",
        args.device,
    ]
    if args.engine_dir:
        cmd += ["--engine-dir", args.engine_dir]
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(UPSTREAM_ROOT))
    logger.info("starting server: %s", " ".join(cmd))
    with log_path.open("wb") as log:
        # Popen inherits the descriptor; the parent's handle can close once the child holds it.
        return subprocess.Popen(cmd, cwd=str(UPSTREAM_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()


def parse_client_log(text: str) -> dict[str, Any]:
    """Per-suite successes out of the client's own log.

    The client prints a suite banner and then one summary line per task; the
    suites are counted in the order they appear, which is upstream's
    ``Args.task_suites``.
    """
    suites: dict[str, dict[str, int]] = {}
    current: str | None = None
    for line in text.splitlines():
        found = _SUITE.search(line)
        if found:
            current = found.group(1)
            suites.setdefault(current, {"successes": 0, "episodes": 0, "tasks": 0})
            continue
        task = _TASK.search(line)
        if task and current is not None:
            entry = suites[current]
            entry["successes"] += int(task.group(2))
            entry["episodes"] += int(task.group(3))
            entry["tasks"] += 1
    for entry in suites.values():
        entry["success_rate"] = (entry["successes"] / entry["episodes"]) if entry["episodes"] else None
    return suites


def main(args: EvalConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not Path(args.client_python).is_file():
        raise SystemExit(f"--client-python {args.client_python} is not a file")
    if not LIBERO_CLIENT.is_file():
        raise SystemExit(f"upstream's client is missing: {LIBERO_CLIENT}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary.get("complete"):
            logger.info("%s already holds a finished sweep; nothing to do", summary_path)
            return summary

    if _port_open(args.port):
        raise SystemExit(f"port {args.port} is already in use; another server is running")

    server = _start_server(args, out / "server.log")
    client_log = out / "client.log"
    try:
        _wait_for_port(args.port, server, args.server_timeout_s)
        logger.info("server ready; running upstream's client over all four suites")
        env = dict(os.environ)
        env.setdefault("MUJOCO_GL", "egl")
        t0 = time.time()
        with client_log.open("wb") as log:
            code = subprocess.call(
                [args.client_python, str(LIBERO_CLIENT)],
                cwd=str(LIBERO_CLIENT.parent),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        seconds = round(time.time() - t0, 1)
    finally:
        _stop_server(server)

    text = client_log.read_text(errors="replace")
    suites = parse_client_log(text)
    total_success = sum(s["successes"] for s in suites.values())
    total_episodes = sum(s["episodes"] for s in suites.values())
    summary = {
        "checkpoint_dir": public_path(args.checkpoint_dir),
        "engine_dir": public_path(args.engine_dir),
        "client_returncode": code,
        "seconds": seconds,
        "suites": suites,
        "pooled_success_rate": (total_success / total_episodes) if total_episodes else None,
        "complete": bool(suites) and code == 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    for name, entry in suites.items():
        logger.info("%s: %d/%d", name, entry["successes"], entry["episodes"])
    logger.info("pooled: %s (client exit %d) -> %s", summary["pooled_success_rate"], code, summary_path)
    return summary


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
