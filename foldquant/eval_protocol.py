# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Closed-loop evaluation protocols and run fingerprints shared by the families.

``PROTOCOLS`` names the LIBERO rollout settings a family evaluator can select
with ``--protocol``. ``p3`` is the paper's closed-loop campaign: every task is
rolled out from LIBERO's stored initial states (episode ``i`` starts from
state ``i``), ten no-op steps with the gripper open let dropped objects
settle, 520 environment steps are allowed per episode, eight actions of each
chunk are executed and the episode ends after the chunk in which the task
succeeds. ``upstream`` leaves every setting to the family's own release loop.

A run fingerprint ties a ``summary.json`` to what produced it, so an
interrupted sweep resumes only into the same run. Torch-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "LIBERO_DUMMY_ACTION",
    "LIBERO_SUITES",
    "PROTOCOLS",
    "Protocol",
    "artifact_digest",
    "libero_init_states",
    "prepare_summary",
    "run_fingerprint",
]

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

#: LIBERO's no-op action: zero motion, gripper open (last channel -1).
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclass(frozen=True)
class Protocol:
    name: str
    #: environment steps per episode after the settle steps; None = family default
    max_episode_steps: Optional[int]
    #: actions executed per policy call; None = family default
    n_action_steps: Optional[int]
    #: episodes per task; None = family default
    n_episodes: Optional[int]
    #: no-op steps after the initial state is set
    settle_steps: int
    #: True: episode i starts from LIBERO's stored initial state i
    fixed_init_states: bool

    def resolve(self, value: Optional[int], field: str, family_default: int) -> int:
        """*value* if given on the command line, else this protocol's, else the family's."""
        if value is not None:
            return int(value)
        own = getattr(self, field)
        return int(family_default if own is None else own)


PROTOCOLS: Dict[str, Protocol] = {
    "upstream": Protocol("upstream", None, None, None, 0, False),
    "p3": Protocol("p3", 520, 8, 20, 10, True),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_digest(path: Any, max_hashed_bytes: int = 4 << 20) -> Optional[Dict[str, Any]]:
    """A cheap identity for a checkpoint or engine directory.

    Lists every regular file under *path* with its size, and hashes the small
    files (configs, manifests, processor tables: up to *max_hashed_bytes*).
    Weight and engine files are identified by name and size only, which is
    enough to tell two builds apart without reading gigabytes.
    """
    if not path:
        return None
    root = Path(path)
    if not root.exists():
        return {"path": str(root), "missing": True}
    if root.is_file():
        files = [root]
        base = root.parent
    else:
        files = sorted(p for p in root.rglob("*") if p.is_file())
        base = root
    entries = []
    for p in files:
        size = p.stat().st_size
        rel = p.relative_to(base).as_posix()
        if size <= max_hashed_bytes and p.suffix.lower() in {".json", ".yaml", ".yml", ".txt", ".sha256", ".cff"}:
            entries.append([rel, size, _sha256(p.read_bytes())])
        else:
            entries.append([rel, size])
    return {"files": len(entries), "digest": _sha256(json.dumps(entries).encode())}


def run_fingerprint(run: Dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of *run* (keys sorted, no whitespace)."""
    return _sha256(json.dumps(run, sort_keys=True, separators=(",", ":"), default=str).encode())


def prepare_summary(path: Path, run: Dict[str, Any], resume: bool = True) -> Dict[str, Any]:
    """The summary to write into *path*, resuming an earlier one only if it is the same run.

    *run* holds everything that decides an outcome: the protocol, the artifact
    digests, the suites and tasks, the episode count, the step cap and the
    chunk length. Its fingerprint is stored beside it. An existing summary with
    a different fingerprint (or none, written before fingerprints existed) is
    refused rather than silently continued, since its finished tasks would be
    credited to the new run.
    """
    fingerprint = run_fingerprint(run)
    summary: Dict[str, Any] = {"tasks": {}}
    if path.is_file():
        if not resume:
            pass
        else:
            old = json.loads(path.read_text())
            if old.get("fingerprint") != fingerprint:
                diff = _diff_runs(old.get("run") or {}, run)
                raise SystemExit(
                    f"{path} belongs to a different run; refusing to resume into it.\n"
                    f"  differing fields: {diff}\n"
                    f"  Pass --no-resume to discard it, or write this run to another --output."
                )
            summary = old
    summary["run"] = run
    summary["fingerprint"] = fingerprint
    summary.setdefault("tasks", {})
    return summary


def _diff_runs(old: Dict[str, Any], new: Dict[str, Any]) -> str:
    if not old:
        return "the existing summary carries no fingerprint (written by an earlier version)"
    keys = sorted(set(old) | set(new))
    changed = [f"{k}: {old.get(k)!r} -> {new.get(k)!r}" for k in keys if old.get(k) != new.get(k)]
    return "; ".join(changed) or "(identical fields, different encoding)"


def libero_init_states(task_suite: Any, task_id: int) -> Any:
    """``task_suite.get_task_init_states(task_id)`` on any torch version.

    LIBERO stores each task's initial states as a pickled NumPy array and reads
    it back with a bare ``torch.load``; torch >= 2.6 defaults ``weights_only``
    to True and refuses that pickle. The files come from the LIBERO checkout
    itself, so they are loaded the way LIBERO was written to load them.
    """
    import os

    import torch
    from libero.libero import get_libero_path

    task = task_suite.get_task(task_id)
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    return torch.load(path, weights_only=False)


def protocol_record(protocol: Protocol, max_episode_steps: int, n_action_steps: int, n_episodes: int) -> Dict[str, Any]:
    """The resolved protocol, as written into a summary."""
    rec = asdict(protocol)
    rec.update(max_episode_steps=max_episode_steps, n_action_steps=n_action_steps, n_episodes=n_episodes)
    return rec
