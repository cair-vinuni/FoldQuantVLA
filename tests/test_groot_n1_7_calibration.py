# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""The GR00T N1.7 calibration plan and forward loop (no upstream checkpoint needed)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "groot_n1_7"))

from foldquant_integration import calibration  # noqa: E402


class _NoisyPolicy:
    """Stands in for ``Gr00tPolicy``: ``get_action`` draws flow-matching noise like upstream does."""

    def __init__(self) -> None:
        self.draws: list = []

    def get_action(self, obs: dict) -> dict:
        self.draws.append(torch.randn(4))
        return {"action": self.draws[-1]}


def test_the_forward_loop_replays_the_same_denoising_noise_every_pass() -> None:
    policy = _NoisyPolicy()
    loop = calibration.make_forward_loop(policy, [{"i": 0}, {"i": 1}, {"i": 2}], seed=7)
    loop(None)  # the SmoothQuant pass
    torch.randn(1000)  # anything the export does in between
    loop(None)  # the GPTQ pass
    first, second = policy.draws[:3], policy.draws[3:]
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert not torch.equal(first[0], first[1])  # per-observation seeds, not one draw repeated


def test_a_different_seed_gives_a_different_calibration_set() -> None:
    p1, p2 = _NoisyPolicy(), _NoisyPolicy()
    calibration.make_forward_loop(p1, [{}], seed=0)(None)
    calibration.make_forward_loop(p2, [{}], seed=1)(None)
    assert not torch.equal(p1.draws[0], p2.draws[0])


def test_heldout_plan_is_episode_disjoint_from_the_calibration_plan() -> None:
    lengths = [50] * 20
    calib = calibration.plan_samples(lengths, 12, seed=0)
    held = calibration.plan_samples(lengths, 6, seed=0, exclude_episodes=[s.episode for s in calib], heldout=True)
    assert not {s.episode for s in calib} & {s.episode for s in held}
    assert all(0 <= s.step < 50 for s in calib + held)
    assert calibration.plan_samples(lengths, 12, seed=0) == calib  # the plan itself is seeded
