# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Calibration observations through the upstream GR00T data path.

The upstream ONNX export captures ONE observation (trajectory 0, step 0). A
folded scheme fits SmoothQuant scales and GPTQ Hessians from what it sees, so
FoldQuant samples ``num_samples`` ``(episode, step)`` pairs spread over the
whole dataset instead, decodes each episode once, and turns every step into
exactly the observation dict ``Gr00tPolicy.get_action`` takes, using
upstream's own ``extract_step_data`` / ``parse_observation_gr00t`` so the
calibration pass and the deployed pass agree byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleId:
    """One calibration observation: which episode, which step."""

    episode: int
    step: int


def resolve_embodiment(model_path: str, embodiment_tag: Optional[str]):
    """``EmbodimentTag`` for the run: explicit, or the single one the checkpoint's processor config names.

    Mirrors the auto-detection of upstream's ``build_trt_pipeline.py`` so a
    fine-tuned checkpoint with one embodiment needs no ``--embodiment-tag``.
    """
    from gr00t.data.embodiment_tags import EmbodimentTag

    if embodiment_tag is not None:
        return EmbodimentTag.resolve(embodiment_tag)
    config_file = Path(model_path) / "processor_config.json"
    if not config_file.is_file():
        raise ValueError(
            f"cannot auto-detect the embodiment: {config_file} not found; pass --embodiment-tag"
        )
    modality_configs = (
        json.loads(config_file.read_text()).get("processor_kwargs", {}).get("modality_configs", {})
    )
    if len(modality_configs) != 1:
        raise ValueError(
            f"processor_config.json names {sorted(modality_configs)} embodiments; pass --embodiment-tag explicitly"
        )
    return EmbodimentTag.resolve(next(iter(modality_configs)))


def load_policy(model_path: str, embodiment_tag: Optional[str], device: Any = "cuda"):
    """The upstream ``Gr00tPolicy``."""
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    tag = resolve_embodiment(model_path, embodiment_tag)
    return Gr00tPolicy(embodiment_tag=tag, model_path=model_path, device=device)


def load_dataset(policy, dataset_path: str, video_backend: str = "torchcodec"):
    """The upstream episode loader, configured from the policy's own modality config."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    return LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=video_backend,
        video_backend_kwargs=None,
    )


def plan_samples(
    episode_lengths: Sequence[int],
    num_samples: int,
    *,
    seed: int,
    exclude_episodes: Sequence[int] = (),
    heldout: bool = False,
) -> List[SampleId]:
    """Choose ``(episode, step)`` pairs spread across episodes.

    Episodes are visited round-robin in a seeded shuffled order so the sample
    covers as many episodes (tasks, scenes) as the budget allows; the step
    inside each episode is drawn uniformly. ``exclude_episodes`` keeps a
    verification split disjoint from the calibration split at the episode
    level. A held-out *step* of a calibrated episode is not held out.
    ``heldout`` only changes the stream so the two splits never coincide.
    """
    rng = np.random.default_rng(seed + (1_000_003 if heldout else 0))
    candidates = [i for i in range(len(episode_lengths)) if i not in set(exclude_episodes)]
    if not candidates:
        raise ValueError("no episodes left to sample from after exclusions")
    rng.shuffle(candidates)
    samples: List[SampleId] = []
    while len(samples) < num_samples:
        for ep in candidates:
            if len(samples) >= num_samples:
                break
            n = int(episode_lengths[ep])
            samples.append(SampleId(ep, int(rng.integers(0, n))))
    return samples


def build_observations(policy, dataset, samples: Sequence[SampleId]) -> List[Dict[str, Any]]:
    """Decode each referenced episode once and build the ``get_action`` observation per sample."""
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.eval.open_loop_eval import parse_observation_gr00t

    modality_configs = policy.get_modality_config()
    by_episode: Dict[int, List[SampleId]] = {}
    for s in samples:
        by_episode.setdefault(s.episode, []).append(s)

    observations: Dict[SampleId, Dict[str, Any]] = {}
    for ep, ids in sorted(by_episode.items()):
        traj = dataset[ep]
        for s in ids:
            # allow_padding: the action modality's delta indices run past the
            # episode end for late steps; the policy never reads actions, but
            # extract_step_data iterates every configured modality.
            point = extract_step_data(
                traj,
                s.step,
                modality_configs=modality_configs,
                embodiment_tag=policy.embodiment_tag,
                allow_padding=True,
            )
            obs: Dict[str, Any] = {}
            for key, value in point.states.items():
                obs[f"state.{key}"] = value
            for key, value in point.images.items():
                obs[f"video.{key}"] = np.array(value)
            for key in modality_configs["language"].modality_keys:
                obs[key] = point.text
            observations[s] = parse_observation_gr00t(obs, modality_configs)
    logger.info("built %d observations from %d episodes", len(samples), len(by_episode))
    return [observations[s] for s in samples]


def sample_observations(
    policy,
    dataset,
    num_samples: int,
    *,
    seed: int,
    exclude_episodes: Sequence[int] = (),
    heldout: bool = False,
) -> Tuple[List[SampleId], List[Dict[str, Any]]]:
    """:func:`plan_samples` + :func:`build_observations`."""
    samples = plan_samples(
        dataset.episode_lengths,
        num_samples,
        seed=seed,
        exclude_episodes=exclude_episodes,
        heldout=heldout,
    )
    return samples, build_observations(policy, dataset, samples)


def make_forward_loop(
    policy, observations: Sequence[Dict[str, Any]], *, seed: int
) -> Callable[[Any], None]:
    """The ``forward_loop(module)`` every FoldQuant export takes.

    It replays the full policy; the module argument is ignored on purpose:
    the LLM and DiT hooks fire wherever they sit in the graph, and the DiT is
    reached only through the whole denoising loop.

    ``torch.manual_seed(seed + i)`` precedes observation ``i`` every time the
    loop runs. The DiT's action tokens start from ``torch.randn`` noise inside
    ``get_action``, so without it each export (and each calibration pass
    within one export) fits on a different activation set: the Hessians of
    the GPTQ pass would be taken on inputs the SmoothQuant scales never saw,
    and two exports of the same arm would round the DiT differently.
    """

    def _loop(_module: Any) -> None:
        with torch.inference_mode():
            for i, obs in enumerate(observations):
                torch.manual_seed(seed + i)
                policy.get_action(obs)

    return _loop
