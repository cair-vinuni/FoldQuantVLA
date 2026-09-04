# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Calibration observations through the upstream Evo-1 server path.

A folded scheme fits SmoothQuant scales and GPTQ Hessians from what it sees, so
FoldQuant samples ``num_samples`` ``(episode, step)`` pairs spread over a
LeRobot dataset and hands each one to upstream's own request handler in exactly
the shape its LIBERO client sends over the wire: three 448x448 images (agent
view, wrist, and the zero-filled third slot the client pads with), the
``[1, 1, 0]`` image mask, the 8-d proprioceptive state, the task prompt and the
24-d action mask whose first seven entries are the LIBERO action.

Nothing here re-implements a transform. ``Evo1_server`` guards its entry point
with ``if __name__ == "__main__"``, so its ``load_model_and_normalizer``,
``decode_image_from_list`` and ``infer_from_json_dict`` are imported and called
as the server calls them; this module only builds the JSON dictionaries and
seeds the sampler. The dataset itself is read by :mod:`._dataset`, which parses
the v3 parquet index directly — the ``lerobot`` package cannot be installed
beside Evo-1's pinned torch, and reading a frame is not a transform.

Seeding is by the global RNG on purpose: upstream draws the flow-matching start
inside ``FlowmatchingActionHead.get_action`` (``torch.rand``) with no way to
pass it in, so ``torch.manual_seed`` immediately before each request is what
makes an observation's replay identical across calibration passes and across
the PyTorch / engine verification runs.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import _dataset
from ._upstream import UPSTREAM_ROOT

logger = logging.getLogger(__name__)

#: Side the **server** resizes every frame to (``decode_image_from_list``).
SERVER_IMAGE_SIZE = 448
#: The client's masks: two real cameras and one zero-filled slot; seven of the
#: 24 action channels are LIBERO's.
IMAGE_MASK = [1, 1, 0]
ACTION_MASK = [1] * 7 + [0] * 17


@dataclass(frozen=True)
class SampleId:
    """One calibration observation: which episode (its ``episode_index``), which step."""

    episode: int
    step: int


@dataclass
class CalibrationSet:
    """A LeRobot dataset reader plus the episodes this run may draw from."""

    frames: Any
    episodes: list[int]


@dataclass
class Deployed:
    """A loaded Evo-1 model with the normalizer the checkpoint ships."""

    model: Any
    normalizer: Any
    arm_key: str
    dataset_key: str


def _upstream_on_path() -> None:
    """Put ``Evo_1/`` on ``sys.path``, which is what upstream's own modules assume.

    ``scripts/Evo1.py`` appends its parent before importing ``config`` and
    ``model.*``; importing those modules from outside the tree needs the same
    entry, and adding it twice is harmless.
    """
    root = str((UPSTREAM_ROOT / "Evo_1").resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_norm_keys(checkpoint_dir: str, arm_key: str = "", dataset_key: str = "") -> tuple[str, str]:
    """The ``(arm_key, dataset_key)`` pair to index this checkpoint's ``norm_stats.json``.

    Upstream's server carries both as literals for the checkpoint its authors
    served; a different release keys its stats differently, and
    ``Normalizer._get_stats_for`` ignores the dataset key entirely when the arm
    entry already holds ``observation.state``. So an empty argument is read off
    the file: unambiguous when there is one key, and an error naming the
    choices when there is more than one.
    """
    stats = json.loads((Path(checkpoint_dir) / "norm_stats.json").read_text())
    arm = arm_key
    if not arm:
        if len(stats) != 1:
            raise ValueError(f"norm_stats.json holds arms {sorted(stats)}; name one with --arm-key")
        arm = next(iter(stats))
    if arm not in stats:
        raise ValueError(f"arm key {arm!r} is not in norm_stats.json (has {sorted(stats)})")
    entry = stats[arm]
    if "observation.state" in entry or "action" in entry:
        return arm, dataset_key  # this arm holds its stats directly; the dataset key is unused
    dataset = dataset_key
    if not dataset:
        if len(entry) != 1:
            raise ValueError(f"arm {arm!r} holds datasets {sorted(entry)}; name one with --dataset-key")
        dataset = next(iter(entry))
    if dataset not in entry:
        raise ValueError(f"dataset key {dataset!r} is not under arm {arm!r} (has {sorted(entry)})")
    return arm, dataset


def load_policy(checkpoint_dir: str, *, arm_key: str = "", dataset_key: str = "", device: str = "cuda") -> Deployed:
    """The upstream model and normalizer, as ``Evo1_server.py`` builds them.

    The checkpoint directory is upstream's: ``config.json``, ``norm_stats.json``
    and the DeepSpeed ``mp_rank_00_model_states.pt``. Upstream pins
    ``num_inference_timesteps`` to 50 inside the loader, so the calibration and
    the served policy walk the same Euler schedule.
    """
    _upstream_on_path()
    from scripts.Evo1_server import load_model_and_normalizer

    arm, dataset = resolve_norm_keys(checkpoint_dir, arm_key, dataset_key)
    logger.info("normalizer keys: arm %r, dataset %r", arm, dataset)
    model, normalizer = load_model_and_normalizer(str(checkpoint_dir))
    if device != "cuda":
        model = model.to(device)
    return Deployed(model=model.eval(), normalizer=normalizer, arm_key=arm, dataset_key=dataset)


def parse_episodes(value: str) -> list[int] | None:
    """``"0-149"`` / ``"0,3,7"`` / ``"0-9,20-29"`` -> episode indices, ``""`` -> None."""
    text = value.strip()
    if not text:
        return None
    out: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part.lstrip("-"):
            first, _, last = part.partition("-")
            out.extend(range(int(first), int(last) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def load_dataset(dataset_path: str, *, episodes: Sequence[int] | None = None) -> CalibrationSet:
    """A LeRobot v3 dataset, local directory or hub id, restricted to *episodes*.

    Read through :mod:`._dataset` rather than the ``lerobot`` package: every
    release of it that still supports this environment's Python would upgrade
    the torch Evo-1 pins (see that module).
    """
    frames = _dataset.LeRobotFrames(_dataset.ensure_local(dataset_path))
    present = frames.available_episodes()
    if not present:
        raise ValueError(f"{dataset_path} has no data files on disk for any episode")
    wanted = present if episodes is None else [e for e in present if int(e) in {int(x) for x in episodes}]
    if not wanted:
        raise ValueError(f"none of the requested episodes are present; the dataset holds {present[:8]}...")
    if episodes is not None and len(wanted) != len({int(x) for x in episodes}):
        logger.warning(
            "%d of the %d requested episodes are present; sampling uses those", len(wanted), len(set(episodes))
        )
    return CalibrationSet(frames=frames, episodes=sorted(int(e) for e in wanted))


def episode_table(dataset: CalibrationSet) -> tuple[list[int], list[int]]:
    """``(episode_index, length)`` per selected episode."""
    lengths = {e.episode_index: e.length for e in dataset.frames.episodes()}
    ids = list(dataset.episodes)
    return ids, [int(lengths[e]) for e in ids]


def plan_samples(
    episode_ids: Sequence[int],
    episode_lengths: Sequence[int],
    num_samples: int,
    *,
    seed: int,
    exclude_episodes: Sequence[int] = (),
    heldout: bool = False,
) -> list[SampleId]:
    """Choose ``(episode, step)`` pairs spread across episodes.

    Episodes are visited round-robin in a seeded shuffled order so the sample
    covers as many episodes (tasks, scenes) as the budget allows; the step
    inside each episode is drawn uniformly. ``exclude_episodes`` keeps a
    verification split disjoint from the calibration split at the episode level.
    """
    if len(episode_ids) != len(episode_lengths):
        raise ValueError("episode_ids and episode_lengths differ in length")
    rng = np.random.default_rng(seed + (1_000_003 if heldout else 0))
    excluded = {int(e) for e in exclude_episodes}
    candidates = [i for i in range(len(episode_ids)) if int(episode_ids[i]) not in excluded]
    if not candidates:
        raise ValueError("no episodes left to sample from after exclusions")
    rng.shuffle(candidates)
    samples: list[SampleId] = []
    while len(samples) < num_samples:
        for i in candidates:
            if len(samples) >= num_samples:
                break
            n = int(episode_lengths[i])
            samples.append(SampleId(int(episode_ids[i]), int(rng.integers(0, n))))
    return samples


def camera_keys(dataset: CalibrationSet) -> list[str]:
    """The dataset's camera streams, in name order — the two the client sends."""
    keys = sorted(k for k in dataset.frames.features if k.startswith("observation.images."))
    if len(keys) < 2:
        raise ValueError(f"Evo-1's client sends two real cameras; the dataset has {keys}")
    return keys[:2]


def _client_frame(frame: Any) -> list:
    """A stored frame -> the JSON list the client puts on the wire.

    ``encode_image_array`` is ``img.astype(uint8).tolist()`` and nothing else:
    the client sends the environment's frame at its own resolution, in its own
    channel order, and the **server** resizes to 448 and applies ``BGR2RGB``.
    That colour swap is upstream's own — the frames it swaps came from MuJoCo
    as RGB — and the checkpoint was evaluated through it, so this reproduces
    the wire exactly and lets the server do both steps rather than pre-empting
    them (resizing twice would also resample the frame a second time).
    """
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
    if np.issubdtype(array.dtype, np.floating):
        array = np.rint(array * 255.0).clip(0, 255).astype(np.uint8)
    return array.astype(np.uint8).tolist()


def client_request(dataset: CalibrationSet, record: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """One frame as the JSON dictionary ``libero_client_4tasks.py`` sends."""
    first = _client_frame(record[keys[0]])
    # The client's third slot is a zero frame of the same shape it sends the
    # others in, masked off by IMAGE_MASK; the server still decodes it.
    zeros = np.zeros((len(first), len(first[0]), 3), dtype=np.uint8)
    return {
        "image": [first, _client_frame(record[keys[1]]), zeros.tolist()],
        "state": np.asarray(record["observation.state"], dtype=np.float32).tolist(),
        "prompt": str(record.get("task", "")),
        "image_mask": list(IMAGE_MASK),
        "action_mask": list(ACTION_MASK),
    }


def build_observations(dataset: CalibrationSet, samples: Sequence[SampleId]) -> list[dict[str, Any]]:
    """The client-format request per sample, read in dataset order to keep the reader's cache warm."""
    keys = camera_keys(dataset)
    requests: dict[SampleId, dict[str, Any]] = {}
    for s in sorted(set(samples), key=lambda s: (s.episode, s.step)):
        requests[s] = client_request(dataset, dataset.frames.frame(s.episode, s.step), keys)
    logger.info("built %d observations from %d episodes", len(samples), len({s.episode for s in samples}))
    return [requests[s] for s in samples]


def sample_observations(
    dataset: CalibrationSet,
    num_samples: int,
    *,
    seed: int,
    exclude_episodes: Sequence[int] = (),
    heldout: bool = False,
) -> tuple[list[SampleId], list[dict[str, Any]]]:
    """:func:`plan_samples` + :func:`build_observations`."""
    ids, lengths = episode_table(dataset)
    samples = plan_samples(ids, lengths, num_samples, seed=seed, exclude_episodes=exclude_episodes, heldout=heldout)
    return samples, build_observations(dataset, samples)


def infer(deployed: Deployed, request: dict[str, Any], *, seed: int) -> np.ndarray:
    """Upstream's ``infer_from_json_dict`` on *request* under a seeded global RNG.

    Returns the denormalised action chunk the client would receive.
    """
    _upstream_on_path()
    from scripts.Evo1_server import infer_from_json_dict

    torch.manual_seed(seed)
    actions = infer_from_json_dict(
        dict(request), deployed.model, deployed.normalizer, deployed.arm_key, deployed.dataset_key
    )
    return np.asarray(actions, dtype=np.float32)


def make_forward_loop(
    deployed: Deployed, observations: Sequence[dict[str, Any]], *, seed: int
) -> Callable[[Any], None]:
    """The ``forward_loop(module)`` every FoldQuant export takes.

    It replays the full model — the module argument is ignored on purpose: the
    LLM and action-head hooks fire wherever they sit, and the head is reached
    only through the whole Euler loop. Observation ``i`` always runs under seed
    ``seed + i`` so every calibration pass of an export (scales, then Hessians)
    fits the same activations.
    """

    def _loop(_module: Any) -> None:
        with torch.inference_mode():
            for i, request in enumerate(observations):
                infer(deployed, request, seed=seed + i)

    return _loop
