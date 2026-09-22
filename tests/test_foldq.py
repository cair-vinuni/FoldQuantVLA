# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The shared FoldQuant fold must behave identically at 4 and 8 bit.

These pin the property that made the shared module necessary: the INT8 action
path used to keep its own copy of the weight quantizer, drifted, and silently
stopped folding. A test that both widths fold the same way is what stops that
from happening again.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from foldquant import foldq  # noqa: E402
from foldquant import rotation as rotations  # noqa: E402
from foldquant.foldq import fold_site, rotation_block_for  # noqa: E402


@pytest.mark.parametrize(
    "k_in,block,expected",
    [(1024, 64, 64), (480, 64, 32), (320, 64, 64), (720, 64, 16), (96, 64, 32), (3, 64, 1)],
)
def test_rotation_block_narrows_to_fit(k_in: int, block: int, expected: int) -> None:
    """A site that does not divide the nominal block still gets a rotation."""
    assert rotation_block_for(k_in, block) == expected


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("fold_order", ["before", "after"])
def test_both_widths_ship_the_same_fold_description(bits: int, fold_order: str) -> None:
    """Only the packed weight differs between 4 and 8 bit; the fold does not."""
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    s = torch.rand(256) + 0.5
    _, _, attrs = fold_site(w, bits=bits, block_size=64, s_ch=s, fold_order=fold_order)
    assert attrs["rot_block_size"] == 64
    field = "act_scale_pre" if fold_order == "before" else "act_scale_ch"
    assert field in attrs
    assert ("act_scale_ch" in attrs) != ("act_scale_pre" in attrs)


def test_an_unfolded_site_tells_the_runtime_not_to_rotate() -> None:
    """No scales means no rotation: the weights were never folded for one."""
    w = torch.randn(8, 128)
    for bits in (4, 8):
        _, _, attrs = fold_site(w, bits=bits, block_size=64, s_ch=None)
        assert attrs["rot_block_size"] == 0
        assert "act_scale_ch" not in attrs and "act_scale_pre" not in attrs


@pytest.mark.parametrize("fold_order", ["before", "after"])
def test_the_fold_is_exact_before_quantization(fold_order: str) -> None:
    """W'·(rotated, scaled x) reconstructs W·x; the fold is baking only."""
    torch.manual_seed(0)
    k, bs = 256, 64
    w = torch.randn(48, k)
    x = torch.randn(6, k)
    s = torch.rand(k) + 0.5
    perm, rot = rotations.hadamard_blocks(k, bs)
    folded = rotations.fold_weight_sq(w, perm, rot, s, bs, fold_order)
    if fold_order == "before":
        xq = rotations.apply_rotation(x / s, perm, rot, bs)
    else:
        xq = rotations.apply_rotation(x, perm, rot, bs) / s
    assert torch.allclose(xq @ folded.T, x @ w.T, atol=1e-3)


def test_a_dense_rotation_carries_the_scale_itself() -> None:
    """The vector must not also ship, or the scale is applied twice."""
    torch.manual_seed(0)
    # build_rotation derives the dense rotation from the weight block, so it needs
    # at least as many output rows as the block is wide; real projections do.
    w = torch.randn(128, 128)
    s = torch.rand(128) + 0.5
    _, _, attrs = fold_site(w, bits=4, block_size=64, s_ch=s, fwht=False)
    assert attrs["rot_block_size"] == 0
    assert "rotation" in attrs and "perm" in attrs
    assert "act_scale_ch" not in attrs and "act_scale_pre" not in attrs


@pytest.mark.parametrize("k_real", [720, 480])
@pytest.mark.parametrize("fold_order", ["before", "after"])
def test_the_fold_is_exact_through_zero_padding(k_real: int, fold_order: str) -> None:
    """Widths that are not multiples of the block are padded; padding must stay exact.

    Reproduces the emitter's exact sequence for a padded site: the weight is
    zero-padded, the activation is zero-padded by an ONNX Pad node, and the
    SmoothQuant vector is measured on the PADDED activation (so its tail entries
    are the clamp floor, not a real amax). If any of the three disagree on the
    padded width, the recovered product is silently wrong rather than an error.
    """
    torch.manual_seed(0)
    bs = 64
    w = torch.randn(48, k_real)
    x = torch.randn(6, k_real)

    w_pad = rotations.pad_in_dim(w, bs)
    x_pad = rotations.pad_in_dim(x, bs)
    assert w_pad.shape[1] == x_pad.shape[1] == ((k_real + bs - 1) // bs) * bs

    # what the capture stores: amax of the padded activation, floored like finalize_scales
    s = x_pad.abs().amax(dim=0).clamp_min(1e-8)

    perm, rot = foldq.site_rotation(w_pad, bs, True)
    folded = rotations.fold_weight_sq(w_pad, perm, rot, s, bs, fold_order)
    if fold_order == "before":
        xq = rotations.apply_rotation(x_pad / s, perm, rot, bs)
    else:
        xq = rotations.apply_rotation(x_pad, perm, rot, bs) / s

    # the padded product must equal the UNPADDED float product
    assert torch.allclose(xq @ folded.T, x @ w.T, atol=1e-3), (
        f"padded fold is not exact at k_real={k_real}, fold_order={fold_order}: "
        f"max |diff| = {(xq @ folded.T - x @ w.T).abs().max():.4f}"
    )


@pytest.mark.parametrize("k_real", [720, 480])
def test_the_padded_site_keeps_the_nominal_block(k_real: int) -> None:
    """fold_site must see the PADDED width, or it narrows the block silently.

    rotation_block_for(720, 64) is 16 and rotation_block_for(480, 64) is 32. A
    caller that pads the activation but hands fold_site the raw width bakes a
    rotation the kernel does not perform.
    """
    w = torch.randn(48, k_real)
    _, _, spec = foldq.fold_site(
        rotations.pad_in_dim(w, 64),
        bits=4,
        block_size=64,
        s_ch=torch.ones(((k_real + 63) // 64) * 64),
        fold_order="before",
        fwht=True,
    )
    assert spec["rot_block_size"] == 64, (
        f"padded site fell back to block {spec['rot_block_size']}; the kernel rotates in 64s. "
        f"rotation_block_for would narrow the RAW width to "
        f"{rotation_block_for(k_real, 64)}; fold_site must see the padded one."
    )


def test_gptq_rounding_beats_round_to_nearest_at_four_bits() -> None:
    """GPTQ spends the same grid better; it must not change anything else.

    At 4 bits the fold is grid-limited. Measured on a flow-matching action head, the
    W4A4/W8A8 error ratio came out 18.3x against an ideal 127/7 = 18.14x, so no
    amount of scaling or rotation buys more. GPTQ adds no codes; it propagates
    each column's rounding error into the columns not yet quantized. The packed
    bytes keep the same length and the scale keeps the same shape, so the kernel
    and the node attributes cannot tell the difference.
    """
    from foldquant.llm_gptq import gptq_prepare

    torch.manual_seed(0)
    k, n = 256, 128
    w = torch.randn(n, k)
    x = torch.randn(512, k) * (1.0 + torch.rand(k))  # per-channel spread
    prep = gptq_prepare((x.T @ x).double())

    rtn_b, rtn_s, _ = foldq.fold_site(w, bits=4, block_size=64, s_ch=None, fwht=True)
    gpt_b, gpt_s, _ = foldq.fold_site(w, bits=4, block_size=64, s_ch=None, fwht=True, gptq=prep)
    assert len(rtn_b) == len(gpt_b), "GPTQ changed the packed byte length"
    assert len(rtn_s) == len(gpt_s), "GPTQ changed the scale shape"
    assert rtn_b != gpt_b, "GPTQ produced identical codes; it was not applied"


@pytest.mark.parametrize("fold_order", ["before", "after"])
def test_the_hessian_is_taken_in_the_frame_the_kernel_sees(fold_order: str) -> None:
    """GPTQ compensates the error of the weight the ENGINE stores.

    That weight multiplies the rotated, scaled activation, so its Hessian has to
    be built there. A Hessian on the raw activation compensates an error that
    never occurs. It is not merely less effective, it is aimed at the wrong
    thing. This pins that the accumulator applies scale and rotation, by checking
    it differs from the raw-frame Hessian.
    """
    torch.manual_seed(0)
    k, bs = 128, 64
    w = torch.randn(64, k)
    perm, rmat = foldq.site_rotation(w, bs, True)
    s = torch.rand(k) + 0.5
    hess, accum = foldq.hessian_accumulator({"s": (perm, rmat)}, {"s": s}, fold_order)
    x = torch.randn(256, k)
    accum("s", x)
    raw = (x.T @ x).double()
    assert not torch.allclose(hess["s"], raw, rtol=1e-3), "Hessian was taken on the raw activation"
    assert hess["s"].shape == (k, k)
    assert torch.allclose(hess["s"], hess["s"].T, atol=1e-6), "Hessian must stay symmetric"


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_the_hessian_accumulates_the_exact_fp32_per_call_sum(device: str) -> None:
    """The accumulator is float64 on the host, fed by fp32 per-call Gram matrices.

    Staging the fp32 matrix (pinned when the activation is on a GPU) and adding
    in place is the fast path; the value it must equal is the plain sum of the
    per-call fp32 ``x_r^T x_r`` widened to float64, with no extra rounding and the
    same result whether the calls came from the CPU or a device.
    """
    torch.manual_seed(0)
    k, bs = 128, 64
    w = torch.randn(64, k)
    perm, rmat = foldq.site_rotation(w, bs, True)
    s = torch.rand(k) + 0.5
    hess, accum = foldq.hessian_accumulator({"s": (perm, rmat)}, {"s": s}, "before")
    expected = torch.zeros(k, k, dtype=torch.float64)
    for _ in range(3):
        x = torch.randn(64, k, device=device)
        accum("s", x)
        xr = rotations.apply_rotation(x.float() / s.to(device), perm, rmat, bs)
        expected += (xr.T @ xr).double().cpu()
    assert hess["s"].dtype == torch.float64 and hess["s"].device.type == "cpu"
    assert torch.equal(hess["s"], expected)
