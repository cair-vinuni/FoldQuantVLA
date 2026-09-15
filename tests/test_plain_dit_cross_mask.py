# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

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

import torch

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


class _MasklessDiT(torch.nn.Module):
    """Upstream N1.5's ``DiT``: called with no masks, and its forward declares none."""

    class config:  # noqa: D106 - stand-in
        hidden_size = 8

    def forward(self, hidden_states, encoder_hidden_states, timestep, encoder_attention_mask=None):
        return hidden_states


class _MaskedPlainDiT(_MasklessDiT):
    """A plain DiT that declares the masks and ignores them."""

    def forward(  # type: ignore[override]
        self, hidden_states, encoder_hidden_states, timestep, image_mask=None, backbone_attention_mask=None
    ):
        return hidden_states


def _loop_without_masks(module):
    module(
        hidden_states=torch.zeros(1, 3, 8),
        encoder_hidden_states=torch.zeros(1, 5, 8),
        timestep=torch.zeros(1, dtype=torch.long),
    )


def test_capture_synthesises_attend_all_masks_for_a_maskless_plain_dit() -> None:
    """The 5-tuple contract holds even when the forward has no mask parameters.

    The masks it records are all-True: that is what "called with no mask" means
    for a DiT that attends the whole encoder sequence.
    """
    from foldquant.calibrate import capture_dit_inputs

    (sample,) = capture_dit_inputs(_MasklessDiT(), _loop_without_masks)
    sa, vl, ts, image_mask, backbone_mask = sample
    assert vl.shape == (1, 5, 8)
    for mask in (image_mask, backbone_mask):
        assert mask.dtype == torch.bool and tuple(mask.shape) == (1, 5)
        assert bool(mask.all()), "attend everything, as the unmasked forward does"


def test_capture_keeps_masks_a_plain_dit_was_actually_given() -> None:
    from foldquant.calibrate import capture_dit_inputs

    given = torch.tensor([[True, False, True, True, False]])

    def loop(module):
        module(
            hidden_states=torch.zeros(1, 3, 8),
            encoder_hidden_states=torch.zeros(1, 5, 8),
            timestep=torch.zeros(1, dtype=torch.long),
            image_mask=given,
            backbone_attention_mask=given,
        )

    (sample,) = capture_dit_inputs(_MaskedPlainDiT(), loop)
    assert torch.equal(sample[3], given) and torch.equal(sample[4], given)


def test_mask_acceptance_is_read_off_the_forward_signature() -> None:
    from foldquant.calibrate import dit_accepts_masks

    assert not dit_accepts_masks(_MasklessDiT())
    assert dit_accepts_masks(_MaskedPlainDiT())


def test_replay_inputs_take_the_module_dtype_where_autocast_used_to() -> None:
    """N1.5 runs its action head under bf16 autocast, so the captured encoder states
    arrive fp32 out of the ``vlln`` LayerNorm while the DiT holds bf16 weights. The
    replay has no autocast; it hands the module inputs in the module's own dtype —
    the engine's declared input dtype — and leaves timestep/masks integral."""
    from foldquant.calibrate import dit_inputs_for

    module = _MasklessDiT().to(torch.bfloat16)
    module.weight = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
    sample = (
        torch.zeros(1, 3, 8),
        torch.zeros(1, 5, 8),
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, 5, dtype=torch.bool),
        torch.ones(1, 5, dtype=torch.bool),
    )
    sa, vl, ts, image_mask, backbone_mask = dit_inputs_for(module, sample)
    assert sa.dtype == vl.dtype == torch.bfloat16
    assert ts.dtype == torch.long
    assert image_mask.dtype == backbone_mask.dtype == torch.bool
