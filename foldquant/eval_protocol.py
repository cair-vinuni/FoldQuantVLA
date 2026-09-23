# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Closed-loop evaluation protocols and run fingerprints shared by the families.

``PROTOCOLS`` names the LIBERO rollout settings a family evaluator can select
with ``--protocol``. ``p3`` is the paper's closed-loop campaign: every task is
rolled out from LIBERO's stored initial states (episode ``i`` starts from
state ``i``) in a simulator seeded with 7, ten no-op steps with the gripper
open let dropped objects settle, 520 environment steps are allowed per
episode, the family's executed prefix of each chunk is applied (eight actions
on GR00T N1.7 and N1.6, one on N1.5, five on π₀.₅) and the episode ends after
the chunk in which the task succeeds. ``upstream`` leaves every setting to the family's own release loop.

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
    #: simulator seed applied to each task's environment; None = family default
    seed: Optional[int] = None

    def resolve(self, value: Optional[int], field: str, family_default: int) -> int:
        """*value* if given on the command line, else this protocol's, else the family's."""
        if value is not None:
            return int(value)
        own = getattr(self, field)
        return int(family_default if own is None else own)


PROTOCOLS: Dict[str, Protocol] = {
    "upstream": Protocol("upstream", None, None, None, 0, False, None),
    "p3": Protocol("p3", 520, 8, 20, 10, True, 7),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


#: Where full-content digests are remembered, keyed by a directory's file
#: listing (relative names, sizes, mtimes). Override with FOLDQUANT_CACHE_DIR.
def _cache_dir() -> Path:
    import os

    return Path(os.environ.get("FOLDQUANT_CACHE_DIR", Path.home() / ".cache" / "foldquant")) / "artifact_digests"


def _listing(root: Path) -> list:
    if root.is_file():
        files, base = [root], root.parent
    else:
        files, base = sorted(p for p in root.rglob("*") if p.is_file()), root
    return [[p.relative_to(base).as_posix(), p.stat().st_size, p.stat().st_mtime_ns] for p in files], files


def _stream_hash(files: list, base: Path) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(base).as_posix().encode())
        h.update(b"\0")
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def artifact_digest(path: Any, content: bool = True) -> Optional[Dict[str, Any]]:
    """An identity for a checkpoint, engine or dataset directory.

    With ``content=True`` the digest is the SHA-256 over the relative name and
    the complete bytes of every regular file under *path*, so any change to
    any file changes it. Reading a large checkpoint once is the cost of that
    guarantee; the result is remembered under ``~/.cache/foldquant`` keyed by
    the directory's listing (names, sizes, mtimes), so an unchanged directory
    is not re-read. With ``content=False`` only the listing is digested, which
    identifies a dataset by its files without reading them.
    """
    if not path:
        return None
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return {"path": str(root), "missing": True}
    listing, files = _listing(root)
    listing_key = _sha256(json.dumps([str(root), listing]).encode())
    if not content:
        return {"files": len(files), "digest": listing_key, "content": False}
    cache = _cache_dir() / listing_key
    if cache.is_file():
        return {"files": len(files), "digest": cache.read_text().strip(), "content": True}
    digest = _stream_hash(files, root.parent if root.is_file() else root)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(digest)
    except OSError:
        pass
    return {"files": len(files), "digest": digest, "content": True}


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


def _cli() -> None:
    """``python -m foldquant.eval_protocol digest PATH [--listing]``: print an artifact's digest."""
    import argparse

    ap = argparse.ArgumentParser(description="print the content (or listing) digest of an artifact directory")
    ap.add_argument("command", choices=["digest"])
    ap.add_argument("path")
    ap.add_argument("--listing", action="store_true", help="digest the file listing only, without reading the files")
    args = ap.parse_args()
    d = artifact_digest(args.path, content=not args.listing)
    if d is None or d.get("missing"):
        raise SystemExit(f"{args.path}: not found")
    print(d["digest"])


def protocol_record(
    protocol: Protocol, max_episode_steps: int, n_action_steps: int, n_episodes: int, seed: Optional[int] = None
) -> Dict[str, Any]:
    """The resolved protocol, as written into a summary: *n_action_steps* is the
    number of actions the family executes per policy call."""
    rec = asdict(protocol)
    rec.update(max_episode_steps=max_episode_steps, n_action_steps=n_action_steps, n_episodes=n_episodes, seed=seed)
    return rec


if __name__ == "__main__":
    _cli()
