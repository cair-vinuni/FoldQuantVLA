# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Calibration observations through the upstream SmolVLA policy path.

A folded scheme fits SmoothQuant scales and GPTQ Hessians from what it sees,
so FoldQuant samples ``num_samples`` ``(episode, step)`` pairs spread over a
LeRobot dataset and pushes each one through the pipeline upstream's evaluator
uses: the checkpoint's own preprocessor (rename -> normalise -> tokenise ->
device) and then ``predict_action_chunk``. Upstream's LIBERO rollout builds
its batch the same way (``lerobot_eval.eval_policy_all``), so the calibration
pass and the evaluated pass see the same tensors.

The dataset is opened with the ``lerobot`` package this tree *is*; frames come
out of its video decoder already in the ``observation.images.*`` key space the
processor expects, so nothing here re-implements a transform. Only two things
are added around upstream's path:

* the episode-balanced sample plan (shared by every family's integration), and
* explicit, seeded flow-matching noise. Upstream draws it inside
  ``sample_actions`` from the global RNG; passing it makes an observation's
  replay identical across calibration passes and across the PyTorch / engine
  verification runs.

A dataset whose camera keys differ from the checkpoint's (``image`` /
``wrist_image`` against ``image`` / ``image2``) is mapped onto the policy's
visual features in declaration order, which is what upstream's ``rename_map``
does for the same mismatch at evaluation time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleId:
    """One calibration observation: which episode (its ``episode_index``), which step."""

    episode: int
    step: int


@dataclass
class Deployed:
    """A loaded SmolVLA policy with the processors the checkpoint ships.

    Upstream keeps the policy and its processor pipeline as separate objects
    (unlike openpi's ``Policy``, which owns its transforms), and the evaluator
    calls them in that order; the tools here pass this triple around so they
    drive exactly that path.
    """

    policy: Any
    preprocessor: Any
    postprocessor: Any


def load_policy(checkpoint: str, *, device: str = "cuda", compile: bool = False) -> Deployed:
    """The upstream ``SmolVLAPolicy`` and its processors, as ``lerobot_eval`` builds them.

    *checkpoint* is a local directory or a hub id; the configuration, the
    normalisation statistics and the processor pipeline all travel inside it.

    ``compile=False`` (the default) clears ``compile_model`` on the loaded
    config. Calibration needs an eager model: the export hooks keep activations
    across forward passes, and a CUDA-graph output is overwritten by the next
    replay. Upstream's own default is eager, so this only matters for a
    checkpoint whose config turns compilation on.
    """
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = SmolVLAConfig.from_pretrained(checkpoint)
    if not compile:
        config.compile_model = False
    config.device = device
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return Deployed(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor)


def parse_episodes(value: str) -> list[int] | None:
    """``"0-149"`` / ``"0,3,7"`` / ``"0-9,20-29"`` -> episode indices, ``""`` -> None.

    The release is split into data files holding a few episodes each and only
    the files a request touches are fetched, so asking for a range is how a
    calibration run bounds what it downloads. What actually loads can be less
    than what was asked for; :func:`episode_table` reports the truth.
    """
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


def load_dataset(dataset_path: str, *, episodes: Sequence[int] | None = None):
    """A LeRobot dataset, local directory or hub id.

    A path that exists on disk is opened as ``root`` (``repo_id`` is then only
    a label and no hub access happens); anything else is treated as a hub
    ``repo_id``. ``episodes`` restricts the load to those episode indices,
    which is how a calibration run avoids pulling a whole LIBERO release.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_path).expanduser()
    episode_list = None if episodes is None else [int(e) for e in episodes]
    if (root / "meta" / "info.json").is_file():
        return LeRobotDataset(repo_id=root.resolve().name, root=root.resolve(), episodes=episode_list)
    return LeRobotDataset(repo_id=dataset_path, episodes=episode_list)


def episode_table(dataset) -> tuple[list[int], list[int], list[int]]:
    """``(episode_index, length, first_index)`` per **loaded** episode, in dataset order.

    ``LeRobotDataset.__getitem__`` indexes the episode-filtered frames, while
    the metadata's ``dataset_from_index`` / ``dataset_to_index`` are slices of
    the *whole* release — the two agree only for an unfiltered load. A filtered
    load also need not contain every episode that was asked for (the release is
    split into data files, and only the files that were fetched contribute), so
    the table is read off the frames actually present rather than off the
    metadata's intentions.
    """
    frames = getattr(dataset, "hf_dataset", None)
    if getattr(dataset, "episodes", None) is None or frames is None:
        episodes = dataset.meta.episodes
        ids = [int(e) for e in episodes["episode_index"]]
        starts = [int(v) for v in episodes["dataset_from_index"]]
        ends = [int(v) for v in episodes["dataset_to_index"]]
        lengths = [end - start for start, end in zip(starts, ends, strict=True)]
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        return [ids[i] for i in order], [lengths[i] for i in order], [starts[i] for i in order]

    column = np.asarray(frames.data.column("episode_index").to_numpy())
    if column.size == 0:
        raise ValueError("the dataset loaded no frames for the requested episodes")
    # Contiguous runs: one per episode, in the order the reader concatenated them.
    boundaries = np.flatnonzero(np.diff(column)) + 1
    starts = [0, *boundaries.tolist()]
    ends = [*boundaries.tolist(), int(column.size)]
    ids = [int(column[s]) for s in starts]
    lengths = [e - s for s, e in zip(starts, ends, strict=True)]
    requested = {int(e) for e in dataset.episodes}
    if len(requested) != len(ids):
        logger.warning(
            "%d of the %d requested episodes are loaded (%s); sampling uses the loaded ones",
            len(ids),
            len(requested),
            ", ".join(str(i) for i in ids[:8]) + ("..." if len(ids) > 8 else ""),
        )
    return ids, lengths, starts


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
    verification split disjoint from the calibration split at the episode
    level — a held-out *step* of a calibrated episode is not held out.
    ``heldout`` only changes the stream so the two splits never coincide.
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


def image_key_map(deployed: Deployed, dataset) -> dict[str, str]:
    """``{dataset key: policy key}`` for the camera streams.

    Matching names pass through untouched. A dataset that names its cameras
    differently is mapped onto the policy's visual features in declaration
    order — the same correspondence upstream's ``--rename_map`` states by hand
    when evaluating a checkpoint on a differently-named LIBERO conversion.
    """
    from lerobot.configs.types import FeatureType

    policy_keys = [
        k for k, ft in deployed.policy.config.input_features.items() if ft.type is FeatureType.VISUAL
    ]
    dataset_keys = [k for k in dataset.meta.features if k.startswith("observation.images.")]
    if set(policy_keys) <= set(dataset_keys):
        return {k: k for k in policy_keys}
    if len(dataset_keys) < len(policy_keys):
        raise ValueError(
            f"the dataset has {len(dataset_keys)} camera streams {dataset_keys} but the checkpoint expects "
            f"{len(policy_keys)} {policy_keys}"
        )
    mapping = dict(zip(sorted(dataset_keys), policy_keys, strict=False))
    logger.warning("camera keys renamed by declaration order: %s", mapping)
    return mapping


def prompt_of(dataset, item: dict[str, Any]) -> str:
    """The task text of a dataset item (``task`` when materialised, else through ``task_index``)."""
    if isinstance(item.get("task"), str):
        return item["task"]
    return str(dataset.meta.tasks.index[int(item["task_index"])])


def policy_batch(dataset, item: dict[str, Any], keys: dict[str, str]) -> dict[str, Any]:
    """One dataset item as the batch upstream's evaluator hands the preprocessor.

    Every tensor gets the leading batch axis the policy works in; the frames
    stay exactly as the dataset decoded them (float CHW in [0, 1]) because that
    is what the processor pipeline normalises.
    """
    from lerobot.utils.constants import OBS_STATE

    batch: dict[str, Any] = {
        policy_key: item[dataset_key].unsqueeze(0) for dataset_key, policy_key in keys.items()
    }
    batch[OBS_STATE] = torch.as_tensor(item[OBS_STATE], dtype=torch.float32).unsqueeze(0)
    batch["task"] = [prompt_of(dataset, item)]
    return batch


def build_observations(deployed: Deployed, dataset, samples: Sequence[SampleId]) -> list[dict[str, Any]]:
    """The policy-format batch per sample.

    Items are read in dataset order to keep each episode's video decoder warm;
    the returned list follows *samples*.
    """
    ids, lengths, starts = episode_table(dataset)
    start_of = dict(zip(ids, starts, strict=True))
    length_of = dict(zip(ids, lengths, strict=True))
    keys = image_key_map(deployed, dataset)
    observations: dict[SampleId, dict[str, Any]] = {}
    for s in sorted(set(samples), key=lambda s: (s.episode, s.step)):
        if s.episode not in start_of:
            raise KeyError(f"episode {s.episode} is not in the dataset")
        if not 0 <= s.step < length_of[s.episode]:
            raise IndexError(f"step {s.step} outside episode {s.episode} (length {length_of[s.episode]})")
        observations[s] = policy_batch(dataset, dataset[start_of[s.episode] + s.step], keys)
    logger.info("built %d observations from %d episodes", len(samples), len({s.episode for s in samples}))
    return [observations[s] for s in samples]


def sample_observations(
    deployed: Deployed,
    dataset,
    num_samples: int,
    *,
    seed: int,
    exclude_episodes: Sequence[int] = (),
    heldout: bool = False,
) -> tuple[list[SampleId], list[dict[str, Any]]]:
    """:func:`plan_samples` + :func:`build_observations`."""
    ids, lengths, _starts = episode_table(dataset)
    samples = plan_samples(
        ids, lengths, num_samples, seed=seed, exclude_episodes=exclude_episodes, heldout=heldout
    )
    return samples, build_observations(deployed, dataset, samples)


def action_noise(deployed: Deployed, seed: int) -> torch.Tensor:
    """The flow-matching starting noise for one chunk, drawn from a seeded generator."""
    config = deployed.policy.config
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((1, int(config.chunk_size), int(config.max_action_dim)), dtype=np.float32)
    return torch.from_numpy(noise)


def infer(deployed: Deployed, observation: dict[str, Any], *, seed: int) -> torch.Tensor:
    """Upstream's inference path on a copy of *observation* with seeded noise; returns the action chunk.

    ``preprocessor -> predict_action_chunk -> postprocessor`` is the sequence
    ``eval_policy_all`` runs per step, so what comes back here is what the
    LIBERO rollout would execute.
    """
    policy = deployed.policy
    policy.reset()
    torch.manual_seed(seed)
    batch = deployed.preprocessor(dict(observation))
    device = next(policy.parameters()).device
    actions = policy.predict_action_chunk(batch, noise=action_noise(deployed, seed).to(device))
    return deployed.postprocessor(actions).detach().float().cpu()


def make_forward_loop(
    deployed: Deployed, observations: Sequence[dict[str, Any]], *, seed: int
) -> Callable[[Any], None]:
    """The ``forward_loop(module)`` every FoldQuant export takes.

    It replays the full policy — the module argument is ignored on purpose:
    the LLM and expert hooks fire wherever they sit in the graph, and the
    expert is reached only through the whole Euler loop. Observation ``i``
    always runs under seed ``seed + i`` so every calibration pass of an export
    (scales, then Hessians) and every export of the same arm fits the same
    activations.
    """

    def _loop(_module: Any) -> None:
        with torch.inference_mode():
            for i, obs in enumerate(observations):
                infer(deployed, obs, seed=seed + i)

    return _loop
