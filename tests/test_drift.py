# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""``worst_channel`` names where an action error landed, under either layout."""

from __future__ import annotations

import pytest
import torch

from foldquant.drift import worst_channel

GROOT_LABELS = ["gripper", "pitch", "roll", "x", "y", "yaw", "z"]


def test_channel_major_names_the_channel():
    """GR00T concatenates whole chunks per named channel: index // steps."""
    delta = torch.zeros(7 * 16)
    delta[0 * 16 + 3] = 2.0  # 'gripper', step 3
    got = worst_channel(delta, order="channel_major", width=16, labels=GROOT_LABELS)
    assert got == {"max_abs": 2.0, "channel": 0, "label": "gripper", "step": 3}


def test_step_major_is_indexed_oppositely():
    """A flattened (steps, width) chunk: index % width."""
    delta = torch.zeros(50 * 7)
    delta[11 * 7 + 6] = 1.5
    got = worst_channel(delta, order="step_major", width=7)
    assert got["channel"] == 6 and got["step"] == 11 and got["label"] is None


def test_the_two_layouts_disagree_on_the_same_buffer():
    """The reason the caller must state the order: the same spike names two channels."""
    delta = torch.zeros(4 * 8)
    delta[9] = 1.0
    cm = worst_channel(delta, order="channel_major", width=8, labels=list("abcd"))
    sm = worst_channel(delta, order="step_major", width=8)
    assert (cm["channel"], cm["step"]) == (1, 1)
    assert (sm["channel"], sm["step"]) == (1, 1)
    # ... but with a non-square layout they diverge, which is the real hazard
    delta2 = torch.zeros(2 * 8)
    delta2[9] = 1.0
    cm2 = worst_channel(delta2, order="channel_major", width=8, labels=list("ab"))
    sm2 = worst_channel(delta2, order="step_major", width=2)
    assert cm2["channel"] == 1 and sm2["channel"] == 1 and cm2["step"] != sm2["step"]


def test_labels_may_be_absent_and_width_inferred():
    delta = torch.zeros(3 * 5)
    delta[7] = -4.0
    got = worst_channel(delta, order="channel_major", labels=["a", "b", "c"])
    assert got["max_abs"] == 4.0 and got["label"] == "b"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"order": "sideways", "width": 4}, "order must be"),
        ({"order": "step_major"}, "step_major needs width"),
        ({"order": "channel_major"}, "channel_major needs width"),
    ],
)
def test_refuses_an_ambiguous_request(kwargs, message):
    with pytest.raises(ValueError, match=message):
        worst_channel(torch.zeros(8), **kwargs)


def test_refuses_an_empty_delta():
    with pytest.raises(ValueError, match="empty delta"):
        worst_channel(torch.zeros(0), order="step_major", width=4)
