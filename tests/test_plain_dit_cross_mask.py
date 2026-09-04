# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""A DiT without a text/image alternation schedule attends the whole encoder.

``AlternateVLDiT`` (N1.6/N1.7) splits cross-attention between the text-only and
the image tokens on a schedule its config carries. A plain ``DiT`` (N1.5) has no
such split: its forward is called with no mask and every cross block attends the
full encoder sequence. The plugin emitters previously refused a plain DiT rather
than guess a schedule, which left N1.5 with no FoldQuant arm at all; they now
give those blocks an all-zero additive mask, which is what "no mask" means once
the graph has to name a tensor.
"""

from __future__ import annotations

import inspect

from foldquant.dit_common import (
    ATTEND_ALL_MASK,
    emit_attend_all_mask,
    resolve_attend_n,
)


class _PlainDiT:
    """A DiT that declares no schedule, on the module or its config."""

    class config:  # noqa: D106 - stand-in
        hidden_size = 1024


class _AlternatingDiT:
    class config:  # noqa: D106 - stand-in
        attend_text_every_n_blocks = 2


def test_a_plain_dit_reports_no_schedule_instead_of_raising() -> None:
    assert resolve_attend_n(_PlainDiT()) is None


def test_an_alternating_dit_still_reports_its_schedule() -> None:
    assert resolve_attend_n(_AlternatingDiT()) == 2


def test_the_schedule_is_read_off_the_module_when_the_config_lacks_it() -> None:
    class OnModule:
        attend_text_every_n_blocks = 3

    assert resolve_attend_n(OnModule()) == 3


def test_the_attend_all_mask_is_zeros_and_broadcastable() -> None:
    """Zeros = attend everywhere. Any non-zero entry would mask a real token."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    nodes: list = []
    inits: list = []
    emit_attend_all_mask(nodes, inits)

    assert nodes == [], "the mask is a constant; it needs no graph nodes"
    assert len(inits) == 1
    tensor = inits[0]
    assert tensor.name == ATTEND_ALL_MASK
    assert tensor.data_type == onnx.TensorProto.BFLOAT16
    assert list(tensor.dims) == [1, 1, 1, 1], "must broadcast over (B, H, S_q, S_kv)"
    assert not np.any(numpy_helper.to_array(tensor)), "a non-zero entry would mask a real encoder token"


def test_both_emitters_route_a_plain_dit_to_the_attend_all_mask() -> None:
    """Neither half of a text/image split may be selected when there is no split.

    Picking one silently would produce a graph that builds and runs at full speed
    while attending to the wrong tokens — the failure the old hard refusal existed
    to prevent.
    """
    from foldquant import dit_int4, dit_int8

    for module in (dit_int8, dit_int4):
        src = inspect.getsource(module)
        assert "elif attend_n is None:" in src, f"{module.__name__} does not handle a scheduleless DiT"
        assert f"attn_mask_name = {ATTEND_ALL_MASK}" in src or "attn_mask_name = ATTEND_ALL_MASK" in src
        assert "if attend_n is None:\n        emit_attend_all_mask(nodes, inits)" in src, (
            f"{module.__name__} selects the mask but never emits it"
        )
