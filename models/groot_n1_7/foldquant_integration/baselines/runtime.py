# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Ported from the authors' Isaac-GR00T fork for the FoldQuant release.

"""Validated fake-quant packed-INT4 runtime for the GR00T-N1.7 W4A4 baselines.

Only the emulated (``fake``) backend is shipped: activations and weights are
rounded to signed INT4 codes and dequantised before a BF16 ``F.linear``. It
measures the rounding of a recipe, not its speed.

Three activation-scaling modes are supported, selected per layer record:

* ``dynamic-per-token`` — one scale per token from that token's ``max|x|``
  (HoloQ-style LLM);
* ``static-per-step-per-channel`` — a calibrated ``[steps, in]`` table indexed
  by the active denoising step (HoloQ-style DiT);
* ``static-per-channel`` — one calibrated ``[1, in]`` table, frozen, used at
  every call (DuQuant-style / Omega-QVLA ``PercentileCalibrator``: per-channel
  q99.9 over the calibration tokens, running max across observations).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .context import get_dit_quant_step, get_dit_quant_total_steps
from .packing import (
    ACTIVATION_BITS,
    SIGNED_QMAX,
    SUPPORTED_PACK_FORMAT_VERSIONS,
    WEIGHT_BITS,
    apply_input_transform,
    pack_signed_int4,
    unpack_signed_int4,
)
from .scope import ScopeSummary, discover_n1d7_targets, normalize_scopes, validate_n1d7_scope


SUPPORTED_BACKENDS = frozenset({"fake"})
ACTIVATION_GRANULARITIES = frozenset(
    {"dynamic-per-token", "static-per-step-per-channel", "static-per-channel"}
)
PACK_ALGORITHMS = frozenset({"holoq-vla", "foldquant-baseline-w4a4"})


def model_config_sha256(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        raise ValueError("Model does not expose a serializable Hugging Face config")
    payload = json.dumps(config.to_dict(), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_record_sha256(record: dict[str, Any]) -> str:
    """Hash the tensor-bearing part of one layer record deterministically."""

    digest = hashlib.sha256()
    for key in sorted(record):
        if key == "sha256":
            continue
        value = record[key]
        digest.update(key.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _module_dimensions(module: nn.Module) -> tuple[int, int, torch.Tensor | None]:
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features, module.bias
    if isinstance(module, nn.Conv3d):
        return module.weight[0].numel(), module.out_channels, module.bias
    raise TypeError(f"Unsupported quantized module type {type(module).__name__}")


class _BaselineOperator(nn.Module):
    """Shared quantized matrix core used by Linear and lowered patch Conv3d."""

    def __init__(
        self,
        module: nn.Module,
        *,
        name: str,
        record: dict[str, Any],
        backend: str,
    ) -> None:
        super().__init__()
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported baseline backend {backend!r}")
        self.name = name
        self.backend = backend
        self.in_features, self.out_features, original_bias = _module_dimensions(module)
        self.scope = str(record["scope"])
        self.solver = str(record["solver"])
        self.module_kind = str(record.get("module_kind", "linear"))
        self.transform = str(record.get("transform", "svd-hadamard"))
        self.activation_granularity = str(
            record.get(
                "activation_granularity",
                "static-per-step-per-channel" if self.scope == "dit" else "dynamic-per-token",
            )
        )
        if self.activation_granularity not in ACTIVATION_GRANULARITIES:
            raise ValueError(
                f"{name}: unsupported activation granularity {self.activation_granularity!r}"
            )

        if "weight_packed" in record:
            weight_packed = record["weight_packed"]
            if weight_packed.dtype != torch.uint8 or weight_packed.ndim != 2:
                raise ValueError(f"{name}: weight_packed must be a rank-2 uint8 tensor")
            weight_shape = tuple(int(item) for item in record["weight_shape"])
            if weight_shape != (self.out_features, self.in_features):
                raise ValueError(
                    f"{name}: logical weight shape {weight_shape} != "
                    f"{(self.out_features, self.in_features)}"
                )
            padded_in_features = int(record.get("padded_in_features", weight_packed.shape[1] * 2))
        elif "weight_q" in record:
            weight_q = record["weight_q"]
            if weight_q.dtype != torch.int8 or tuple(weight_q.shape) != (
                self.out_features,
                self.in_features,
            ):
                raise ValueError(f"{name}: invalid legacy weight_q tensor")
            weight_packed, _ = pack_signed_int4(weight_q)
            padded_in_features = weight_packed.shape[1] * 2
        else:
            raise ValueError(f"{name}: record has no quantized weight")
        if padded_in_features != weight_packed.shape[1] * 2:
            raise ValueError(f"{name}: packed storage does not match padded_in_features")
        self.padded_in_features = padded_in_features
        self.register_buffer("weight_packed", weight_packed.detach().cpu().contiguous())

        weight_scale = record["weight_scale"]
        if tuple(weight_scale.shape) != (self.out_features, 1):
            raise ValueError(f"{name}: invalid weight_scale shape {tuple(weight_scale.shape)}")
        self.register_buffer("weight_scale", weight_scale.detach().float().cpu())
        bias_value = record["bias"] if "bias" in record else original_bias
        if bias_value is None:
            self.bias = None
        else:
            if tuple(bias_value.shape) != (self.out_features,):
                raise ValueError(f"{name}: invalid bias shape {tuple(bias_value.shape)}")
            self.register_buffer("bias", bias_value.detach().clone())

        permutation = record.get("permutation")
        rotations = record.get("rotation_blocks")
        if self.transform == "identity":
            self.permutation = None
            self.rotation_blocks = None
        else:
            if not isinstance(permutation, torch.Tensor) or tuple(permutation.shape) != (
                self.in_features,
            ):
                raise ValueError(f"{name}: invalid permutation")
            if not isinstance(rotations, torch.Tensor) or rotations.ndim != 3:
                raise ValueError(f"{name}: invalid rotation_blocks")
            if rotations.shape[0] * rotations.shape[1] != self.in_features:
                raise ValueError(f"{name}: rotation blocks do not cover all input channels")
            self.register_buffer("permutation", permutation.detach().long().cpu())
            self.register_buffer("rotation_blocks", rotations.detach().float().cpu())

        activation_scale = record.get("activation_scale")
        if self.activation_granularity == "dynamic-per-token":
            if activation_scale is not None:
                raise ValueError(f"{name}: dynamic activation records must not store a scale table")
            self.activation_scale = None
        elif self.activation_granularity == "static-per-step-per-channel":
            if self.scope != "dit" or not isinstance(activation_scale, torch.Tensor):
                raise ValueError(f"{name}: static per-step activation tables are DiT-only")
            if activation_scale.ndim != 2 or activation_scale.shape[1] != self.in_features:
                raise ValueError(f"{name}: invalid DiT activation scale table")
            self.register_buffer("activation_scale", activation_scale.detach().float().cpu())
        else:  # static-per-channel
            if not isinstance(activation_scale, torch.Tensor):
                raise ValueError(f"{name}: static per-channel records need an activation scale")
            if tuple(activation_scale.shape) != (1, self.in_features):
                raise ValueError(
                    f"{name}: static per-channel scale must be [1, {self.in_features}], "
                    f"got {tuple(activation_scale.shape)}"
                )
            if not bool(torch.isfinite(activation_scale).all()) or bool(
                (activation_scale <= 0).any()
            ):
                raise ValueError(f"{name}: static per-channel scale must be finite and positive")
            self.register_buffer("activation_scale", activation_scale.detach().float().cpu())

    def _apply(self, fn):
        super()._apply(fn)
        # Scales are numerical metadata: preserve FP32 after model.to(dtype=bf16)
        # while retaining the requested device.
        self.weight_scale = self.weight_scale.float()
        if isinstance(getattr(self, "activation_scale", None), torch.Tensor):
            self.activation_scale = self.activation_scale.float()
        if isinstance(getattr(self, "bias", None), torch.Tensor):
            self.bias = self.bias.float()
        return self

    def _transform_and_pad(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"{self.name}: input width {inputs.shape[-1]} != {self.in_features}")
        if self.transform == "identity":
            transformed = inputs
        else:
            transformed = apply_input_transform(inputs, self.permutation, self.rotation_blocks)
        if self.padded_in_features > self.in_features:
            transformed = F.pad(transformed, (0, self.padded_in_features - self.in_features))
        return transformed

    def _pad_scale(self, scale: torch.Tensor) -> torch.Tensor:
        if self.padded_in_features > self.in_features:
            scale = F.pad(scale, (0, self.padded_in_features - self.in_features), value=1.0)
        return scale

    def _activation_codes_and_scale(
        self, transformed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.activation_granularity == "dynamic-per-token":
            scale = transformed.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / SIGNED_QMAX
        elif self.activation_granularity == "static-per-channel":
            scale = self._pad_scale(
                self.activation_scale[0].to(transformed.device, transformed.dtype)
            )
        else:
            step = get_dit_quant_step()
            total_steps = get_dit_quant_total_steps()
            if step is None or total_steps is None:
                raise RuntimeError(
                    f"{self.name}: DiT W4A4 inference requires an active denoising-step context"
                )
            table_steps = self.activation_scale.shape[0]
            if total_steps != table_steps:
                raise RuntimeError(
                    f"{self.name}: runtime uses {total_steps} denoising steps but pack has "
                    f"{table_steps}"
                )
            scale = self._pad_scale(
                self.activation_scale[step].to(transformed.device, transformed.dtype)
            )
        codes = torch.round(transformed / scale).clamp(-SIGNED_QMAX, SIGNED_QMAX).to(torch.int8)
        return codes, scale

    def _fake_forward(self, transformed: torch.Tensor) -> torch.Tensor:
        codes, activation_scale = self._activation_codes_and_scale(transformed)
        activation = codes.to(transformed.dtype) * activation_scale
        weight_codes = unpack_signed_int4(
            self.weight_packed, logical_width=self.padded_in_features
        ).to(device=transformed.device, dtype=transformed.dtype)
        weight = weight_codes * self.weight_scale.to(transformed.device, transformed.dtype)
        bias = None if self.bias is None else self.bias.to(transformed.device, transformed.dtype)
        return F.linear(activation, weight, bias)

    def _forward_matrix(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._fake_forward(self._transform_and_pad(inputs))


class BaselineLinear(_BaselineOperator):
    """Emulated W4A4 Linear driven by one packed layer record."""

    def __init__(
        self, linear: nn.Linear, *, name: str, record: dict[str, Any], backend: str = "fake"
    ) -> None:
        if not isinstance(linear, nn.Linear):
            raise TypeError("BaselineLinear can only replace nn.Linear")
        super().__init__(linear, name=name, record=record, backend=backend)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._forward_matrix(inputs)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scope={self.scope}, {self.activation_granularity}, W4A4 (emulated)"
        )


class BaselinePatchConv3d(_BaselineOperator):
    """Qwen3-VL non-overlapping patch Conv3d lowered to the W4A4 matrix core."""

    def __init__(
        self, conv: nn.Conv3d, *, name: str, record: dict[str, Any], backend: str = "fake"
    ) -> None:
        if (
            conv.groups != 1
            or tuple(conv.padding) != (0, 0, 0)
            or tuple(conv.dilation) != (1, 1, 1)
        ):
            raise ValueError(f"{name}: only the Qwen3-VL patch Conv3d can be lowered")
        if tuple(conv.kernel_size) != tuple(conv.stride):
            raise ValueError(f"{name}: patch Conv3d kernel and stride must match")
        self.kernel_size = tuple(conv.kernel_size)
        self.in_channels = conv.in_channels
        super().__init__(conv, name=name, record=record, backend=backend)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected = (self.in_channels, *self.kernel_size)
        if tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                f"{self.name}: patch lowering expects per-patch input [B, {expected}], "
                f"got {tuple(inputs.shape)}"
            )
        output = self._forward_matrix(inputs.flatten(1))
        return output[:, :, None, None, None]


# The fork's class names, kept so its tests and notes read unchanged.
HoloQLinear = BaselineLinear
HoloQPatchConv3d = BaselinePatchConv3d


def _parent_and_attribute(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pack(path: str | Path) -> dict[str, Any]:
    """Load a pack and verify its ``.sha256`` sidecar."""

    pack_path = Path(path)
    if not pack_path.is_file():
        raise FileNotFoundError(f"baseline pack not found: {pack_path}")
    checksum_path = pack_path.with_suffix(pack_path.suffix + ".sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(f"baseline pack checksum sidecar not found: {checksum_path}")
    expected_checksum = checksum_path.read_text(encoding="ascii").strip().split()[0]
    if file_sha256(pack_path) != expected_checksum:
        raise ValueError(f"baseline pack file checksum mismatch: {pack_path}")
    try:
        pack = torch.load(pack_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        pack = torch.load(pack_path, map_location="cpu")
    if not isinstance(pack, dict):
        raise ValueError("baseline pack root must be a dictionary")
    return pack


load_holoq_pack = load_pack


def apply_pack(
    model: nn.Module,
    path: str | Path,
    *,
    expected_suite: str | None = None,
    backend: str = "fake",
) -> ScopeSummary:
    """Validate and atomically replace every declared W4A4 target with its emulation.

    ``model`` is the ``Gr00tN1d7`` module (``Gr00tPolicy.model``). The replacement
    modules are created on CPU; move the model to its device afterwards.
    """

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(SUPPORTED_BACKENDS)}")
    pack = load_pack(path)
    manifest = pack.get("manifest")
    records = pack.get("layers")
    if not isinstance(manifest, dict) or not isinstance(records, dict):
        raise ValueError("Pack must contain dictionary fields 'manifest' and 'layers'")
    version = manifest.get("format_version")
    if version not in SUPPORTED_PACK_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported pack format {version}; supported={sorted(SUPPORTED_PACK_FORMAT_VERSIONS)}"
        )
    compatible_backends = manifest.get("runtime_compatible_backends")
    if compatible_backends is not None and backend not in compatible_backends:
        raise ValueError(
            f"Pack declares backends {compatible_backends}, not requested backend {backend!r}"
        )
    if manifest.get("algorithm") not in PACK_ALGORITHMS:
        raise ValueError(f"Unsupported pack algorithm {manifest.get('algorithm')!r}")
    if manifest.get("model_type") != "Gr00tN1d7":
        raise ValueError(f"Pack model_type must be Gr00tN1d7, got {manifest.get('model_type')}")
    if not str(manifest.get("suite", "")).strip():
        raise ValueError("Pack carries no suite label")
    if expected_suite is not None and manifest.get("suite") != expected_suite:
        raise ValueError(
            f"Pack suite {manifest.get('suite')!r} does not match requested {expected_suite!r}"
        )
    if (
        manifest.get("weight_bits") != WEIGHT_BITS
        or manifest.get("activation_bits") != ACTIVATION_BITS
    ):
        raise ValueError("Only uniform W4A4 packs are supported")

    scopes = normalize_scopes(manifest.get("scopes", ("llm", "dit")))
    include_vit_mergers = bool(manifest.get("include_vit_mergers", False))
    include_vit_patch_embed = bool(manifest.get("include_vit_patch_embed", False))
    targets = discover_n1d7_targets(
        model,
        scopes=scopes,
        include_vit_mergers=include_vit_mergers,
        include_vit_patch_embed=include_vit_patch_embed,
    )
    summary = validate_n1d7_scope(
        targets,
        expected_llm_layers=int(manifest.get("llm_layers", 0)),
        expected_dit_layers=int(manifest.get("dit_layers", 0)),
        expected_vit_layers=int(manifest.get("vit_layers", 0)),
        scopes=scopes,
    )
    target_names = {target.name for target in targets}
    missing = sorted(target_names - set(records))
    extra = sorted(set(records) - target_names)
    if missing or extra:
        raise ValueError(
            "Pack coverage does not exactly match the model: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    if manifest.get("config_sha256") != model_config_sha256(model):
        raise ValueError("Pack config hash does not match the loaded checkpoint")
    if "dit" in scopes:
        model_steps = int(model.action_head.num_inference_timesteps)
        if model_steps != int(manifest["num_inference_timesteps"]):
            raise ValueError("Checkpoint and pack denoising-step counts differ")

    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    for target in targets:
        record = records[target.name]
        if record.get("sha256") != tensor_record_sha256(record):
            raise ValueError(f"Layer record checksum mismatch: {target.name}")
        if record.get("scope") != target.scope:
            raise ValueError(f"Layer scope mismatch for {target.name}")
        if record.get("module_kind", "linear") != target.module_kind:
            raise ValueError(f"Module kind mismatch for {target.name}")
        if target.module_kind == "conv3d_patch":
            replacement: nn.Module = BaselinePatchConv3d(
                target.module, name=target.name, record=record, backend=backend
            )
        else:
            replacement = BaselineLinear(
                target.module, name=target.name, record=record, backend=backend
            )
        parent, attribute = _parent_and_attribute(model, target.name)
        replacements.append((parent, attribute, replacement))
    for parent, attribute, replacement in replacements:
        setattr(parent, attribute, replacement)

    model.baseline_manifest = manifest
    model.baseline_backend = backend
    return summary


apply_holoq_pack = apply_pack
