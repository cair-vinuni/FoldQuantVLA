# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Locate the largest action error, so a drift cell can name it.

A low action cosine with a max-abs near the width of a channel's range is
almost always one channel saturating rather than the chunk degrading, and the
two call for different responses. The summary carries the magnitude already;
what it cannot say is *where*, which leaves the reader inferring "gripper flip"
from a number. These helpers answer that from the same tensors the cosine is
computed on.

Two layouts occur across the families and they index oppositely:

* **channel-major** — the action arrives as a dict of named channels, each a
  whole chunk, concatenated in key order (GR00T). Element ``i`` belongs to
  channel ``i // steps``.
* **step-major** — the action is a ``(steps, width)`` chunk flattened (pi).
  Element ``i`` belongs to channel ``i % width``.

Getting that backwards names the wrong channel while still looking plausible,
which is why the caller states the layout rather than the helper guessing it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

__all__ = ["worst_channel"]


def worst_channel(
    delta: Any,
    *,
    order: str,
    width: Optional[int] = None,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Where the largest ``|delta|`` sits, as ``{max_abs, channel, label, step}``.

    Args:
        delta: 1-D engine-minus-reference over one flattened action chunk.
        order: ``"channel_major"`` or ``"step_major"`` — see the module
            docstring; the two index oppositely.
        width: channels per step (step-major), or steps per channel
            (channel-major). Inferred from *labels* when omitted.
        labels: channel names in concatenation order, when the family has them.

    Returns ``label: None`` for a family whose channels are unnamed.
    """
    import torch

    flat = torch.as_tensor(delta).flatten().abs()
    if flat.numel() == 0:
        raise ValueError("worst_channel: empty delta")
    index = int(torch.argmax(flat))
    total = int(flat.numel())

    if order == "channel_major":
        per = int(width) if width else (total // len(labels) if labels else 0)
        if per <= 0:
            raise ValueError("channel_major needs width (steps per channel) or labels")
        channel, step = divmod(index, per)
    elif order == "step_major":
        if not width:
            raise ValueError("step_major needs width (channels per step)")
        step, channel = divmod(index, int(width))
    else:
        raise ValueError(f"order must be 'channel_major' or 'step_major', got {order!r}")

    label = None
    if labels is not None and 0 <= channel < len(labels):
        label = str(labels[channel])
    return {
        "max_abs": float(flat[index]),
        "channel": int(channel),
        "label": label,
        "step": int(step),
    }
