# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""The emulated W4A4 baseline arms of GR00T N1.7 (no upstream checkpoint needed)."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "groot_n1_7"))

from foldquant_integration.baselines import builder, calibration, context, packing, runtime  # noqa: E402


@pytest.mark.parametrize("rotation_mode", ["svd_hadamard", "svd"])
def test_permute_rotate_preserves_the_product_before_rounding(rotation_mode: str) -> None:
    torch.manual_seed(0)
    weight = torch.randn(512, 256)
    inputs = torch.randn(64, 256)
    perm, rotations, rotated_weight = packing.build_svd_hadamard_transform(
        weight, block_size=64, seed=3, rotation_mode=rotation_mode
    )
    assert rotations.shape == (4, 64, 64)
    eye = torch.eye(64).expand(4, 64, 64)
    assert torch.allclose(rotations.transpose(1, 2) @ rotations, eye, atol=1e-4)
    transformed = packing.apply_input_transform(inputs, perm, rotations)
    folded = packing.apply_weight_transform(weight, perm, rotations)
    assert torch.allclose(folded, rotated_weight, atol=1e-4)
    assert torch.allclose(transformed @ folded.T, inputs @ weight.T, atol=1e-3)


def test_svd_mode_is_the_singular_vectors_alone() -> None:
    torch.manual_seed(1)
    weight = torch.randn(128, 64)
    perm, rotations, _ = packing.build_svd_hadamard_transform(
        weight, block_size=64, seed=0, rotation_mode="svd"
    )
    block = weight.index_select(1, perm)
    u, _, _ = torch.linalg.svd(block.T, full_matrices=True)
    # Singular vectors are unique up to sign per column.
    agreement = (rotations[0] * u).sum(dim=0).abs()
    assert torch.allclose(agreement, torch.ones(64), atol=1e-3)


def _record(in_features: int, out_features: int, granularity: str, scale_rows: int | None) -> dict:
    torch.manual_seed(2)
    weight = torch.randn(out_features, in_features)
    perm, rotations, rotated = packing.build_svd_hadamard_transform(weight, block_size=64, seed=0)
    weight_q, weight_scale = packing.symmetric_quantize_per_output_channel(rotated)
    packed, logical = packing.pack_signed_int4(weight_q, pad_to=64)
    record = {
        "scope": "llm" if granularity != "static-per-step-per-channel" else "dit",
        "module_kind": "linear",
        "projection": "mlp.down_proj",
        "solver": "rtn",
        "weight_packed": packed,
        "weight_shape": [out_features, logical],
        "padded_in_features": packed.shape[1] * 2,
        "weight_scale": weight_scale,
        "transform": "svd-hadamard",
        "permutation": perm,
        "rotation_blocks": rotations.half(),
        "activation_granularity": granularity,
        "activation_scale": None
        if scale_rows is None
        else torch.full((scale_rows, in_features), 0.05),
        "bias": None,
    }
    record["sha256"] = runtime.tensor_record_sha256(record)
    return record, weight


def test_static_per_channel_layer_uses_one_frozen_table_and_int4_codes() -> None:
    record, weight = _record(128, 32, "static-per-channel", 1)
    inputs = torch.randn(64, 128) * 3
    # Calibrate the frozen table the way the collector does: per-channel q99.9 / 7
    # of the permuted + rotated activation.
    transformed_cal = packing.apply_input_transform(
        inputs, record["permutation"], record["rotation_blocks"].float()
    )
    record["activation_scale"] = (
        calibration._per_channel_quantile(transformed_cal, 0.999) / packing.SIGNED_QMAX
    ).unsqueeze(0)
    record["sha256"] = runtime.tensor_record_sha256(record)
    layer = runtime.BaselineLinear(nn.Linear(128, 32, bias=False), name="t", record=record)
    assert tuple(layer.activation_scale.shape) == (1, 128)
    transformed = layer._transform_and_pad(inputs)
    codes, scale = layer._activation_codes_and_scale(transformed)
    assert codes.dtype == torch.int8
    assert int(codes.min()) >= -7 and int(codes.max()) <= 7
    assert torch.equal(scale, layer.activation_scale[0])
    # No denoising-step context is needed for the frozen table.
    assert context.get_dit_quant_step() is None
    out = layer(inputs)
    assert out.shape == (64, 32)
    assert torch.isfinite(out).all()
    reference = inputs @ weight.T
    cos = torch.nn.functional.cosine_similarity(out.flatten(), reference.flatten(), dim=0)
    assert cos > 0.9


def test_static_per_channel_record_rejects_a_wrong_table_shape() -> None:
    record, _ = _record(128, 32, "static-per-channel", 4)
    with pytest.raises(ValueError, match=r"\[1, 128\]"):
        runtime.BaselineLinear(nn.Linear(128, 32, bias=False), name="t", record=record)


def test_per_step_layer_needs_the_step_context() -> None:
    record, _ = _record(128, 32, "static-per-step-per-channel", 4)
    layer = runtime.BaselineLinear(nn.Linear(128, 32, bias=False), name="t", record=record)
    inputs = torch.randn(2, 128)
    with pytest.raises(RuntimeError, match="denoising-step context"):
        layer(inputs)
    with context.set_dit_quant_step(2, total_steps=4):
        assert layer(inputs).shape == (2, 32)


def test_per_channel_quantile_matches_a_single_call() -> None:
    torch.manual_seed(3)
    rows = torch.randn(300, 5000)
    chunked = calibration._per_channel_quantile(rows, 0.999)
    whole = torch.quantile(rows.abs(), 0.999, dim=0).clamp_min(1e-6)
    assert torch.allclose(chunked, whole)


class _Config:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def to_dict(self) -> dict:
        return {"dtype": self.dtype, "model_type": "Gr00tN1d7"}


class _Model(nn.Module):
    def __init__(self, dtype: str) -> None:
        super().__init__()
        self.config = _Config(dtype)


def test_builder_rejects_a_calibration_from_another_config(tmp_path) -> None:
    calibrated_on = _Model("bfloat16")
    artifact = {
        "manifest": {
            "format_version": calibration.CALIBRATION_FORMAT_VERSION,
            "algorithm": "foldquant-baseline-calibration",
            "config_sha256": runtime.model_config_sha256(calibrated_on),
            "scopes": ["llm", "dit"],
        },
        "layers": {},
    }
    path = tmp_path / "calibration.pt"
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="config hash"):
        builder.build_pack(
            _Model("float32"),
            calibration_path=path,
            output_path=tmp_path / "pack.pt",
            checkpoint="ckpt",
            source_revision="test",
        )


def test_dit_step_context_counts_calls_and_wraps() -> None:
    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_inference_timesteps = 4
            self.model = nn.Identity()

    head = Head()
    seen = []
    head.model.register_forward_hook(lambda *_: seen.append(context.get_dit_quant_step()))
    ctx = context.DiTStepContext(head)
    for _ in range(6):
        head.model(torch.zeros(1))
    assert seen == [0, 1, 2, 3, 0, 1]
    assert context.get_dit_quant_step() is None
    ctx.remove()
