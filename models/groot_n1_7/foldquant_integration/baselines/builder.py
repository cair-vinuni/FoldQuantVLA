# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Ported from the authors' Isaac-GR00T fork for the FoldQuant release.

"""Build a packed W4A4 baseline artifact from a calibration artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .calibration import CALIBRATION_ALGORITHMS, CALIBRATION_FORMAT_VERSION
from .packing import (
    ACTIVATION_BITS,
    PACK_FORMAT_VERSION,
    WEIGHT_BITS,
    apply_weight_transform,
    gptq_quantize_blockwise,
    pack_signed_int4,
    symmetric_quantize_per_output_channel,
)
from .runtime import (
    ACTIVATION_GRANULARITIES,
    file_sha256,
    model_config_sha256,
    tensor_record_sha256,
)
from .scope import discover_n1d7_targets, normalize_scopes, validate_n1d7_scope


def _load_artifact(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location="cpu")


def _weight_matrix(module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return module.weight.detach().float()
    if isinstance(module, nn.Conv3d):
        return module.weight.detach().float().flatten(1)
    raise TypeError(f"Unsupported quantization target {type(module).__name__}")


def build_pack(
    model: nn.Module,
    *,
    calibration_path: str | Path,
    output_path: str | Path,
    checkpoint: str,
    source_revision: str,
    checkpoint_revision: str = "",
    scopes: Iterable[str] | str | None = None,
    include_vit_mergers: bool | None = None,
    include_vit_patch_embed: bool | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    """Quantize every in-scope weight in the calibrated coordinates and pack it.

    Every layer record carries its own transform, solver, activation granularity
    and (when static) activation-scale table, so the runtime needs nothing but
    the pack. The model must be the checkpoint the calibration was collected on
    (``config_sha256`` guard), loaded in bf16 like the serving policy.
    """

    calibration_path = Path(calibration_path)
    calibration = _load_artifact(calibration_path)
    calibration_manifest = calibration.get("manifest", {})
    stats = calibration.get("layers", {})
    if calibration_manifest.get("format_version") not in {1, CALIBRATION_FORMAT_VERSION}:
        raise ValueError("Unsupported calibration artifact format")
    if calibration_manifest.get("algorithm") not in CALIBRATION_ALGORITHMS:
        raise ValueError("Not a W4A4 baseline calibration artifact")
    config_hash = model_config_sha256(model)
    if calibration_manifest.get("config_sha256") != config_hash:
        raise ValueError("Calibration config hash does not match the loaded checkpoint")

    selected_scopes = normalize_scopes(scopes or calibration_manifest.get("scopes"))
    if include_vit_mergers is None:
        include_vit_mergers = bool(calibration_manifest.get("include_vit_mergers", False))
    if include_vit_patch_embed is None:
        include_vit_patch_embed = bool(calibration_manifest.get("include_vit_patch_embed", False))
    targets = discover_n1d7_targets(
        model,
        scopes=selected_scopes,
        include_vit_mergers=include_vit_mergers,
        include_vit_patch_embed=include_vit_patch_embed,
    )
    summary = validate_n1d7_scope(
        targets,
        expected_llm_layers=int(calibration_manifest.get("llm_layers", 0)),
        expected_dit_layers=int(calibration_manifest.get("dit_layers", 0)),
        expected_vit_layers=int(calibration_manifest.get("vit_layers", 0)),
        scopes=selected_scopes,
    )
    target_names = {target.name for target in targets}
    if set(stats) != target_names:
        missing = sorted(target_names - set(stats))
        extra = sorted(set(stats) - target_names)
        raise ValueError(
            f"Calibration coverage mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    llm_granularity = str(calibration_manifest.get("llm_activation_granularity", "dynamic-per-token"))
    dit_granularity = str(
        calibration_manifest.get("dit_activation_granularity", "static-per-step-per-channel")
    )
    for granularity in (llm_granularity, dit_granularity):
        if granularity not in ACTIVATION_GRANULARITIES:
            raise ValueError(f"Unsupported activation granularity {granularity!r} in calibration")

    gptq_block_size = int(calibration_manifest["gptq_block_size"])
    damping = float(calibration_manifest["gptq_damping"])
    records: dict[str, dict[str, Any]] = {}
    for target in targets:
        stat = stats[target.name]
        weight = _weight_matrix(target.module)
        transform = str(stat.get("transform", "svd-hadamard"))
        permutation = stat.get("permutation")
        rotations = stat.get("rotation_blocks")
        if transform == "identity":
            rotated_weight = weight
        else:
            rotations = rotations.float()
            rotated_weight = apply_weight_transform(weight, permutation, rotations)

        if target.module_kind == "conv3d_patch":
            weight_q, weight_scale = symmetric_quantize_per_output_channel(rotated_weight)
            solver = "rtn"
        elif target.scope in {"llm", "vit"}:
            weight_q, weight_scale = gptq_quantize_blockwise(
                rotated_weight,
                stat["hessian_blocks"],
                block_size=gptq_block_size,
                damping=damping,
            )
            solver = "gptq"
        else:
            weight_q, weight_scale = symmetric_quantize_per_output_channel(rotated_weight)
            solver = "rtn"

        if target.module_kind == "conv3d_patch":
            activation_granularity = "dynamic-per-token"
        elif target.scope == "dit":
            activation_granularity = dit_granularity
        else:
            activation_granularity = llm_granularity
        recorded = stat.get("activation_granularity")
        if recorded is not None and recorded != activation_granularity:
            raise ValueError(
                f"{target.name}: calibration recorded {recorded!r} activations, "
                f"manifest says {activation_granularity!r}"
            )
        activation_scale = stat.get("activation_scale")
        if activation_granularity == "dynamic-per-token":
            activation_scale = None
        else:
            if not isinstance(activation_scale, torch.Tensor):
                raise ValueError(f"{target.name}: static activation scaling needs a calibrated table")
            activation_scale = activation_scale.float()
            expected_rows = (
                int(calibration_manifest["num_inference_timesteps"])
                if activation_granularity == "static-per-step-per-channel"
                else 1
            )
            if tuple(activation_scale.shape) != (expected_rows, weight.shape[1]):
                raise ValueError(
                    f"{target.name}: activation scale must be [{expected_rows}, {weight.shape[1]}], "
                    f"got {tuple(activation_scale.shape)}"
                )

        weight_packed, logical_width = pack_signed_int4(weight_q.cpu(), pad_to=64)
        record = {
            "scope": target.scope,
            "module_kind": target.module_kind,
            "projection": target.projection,
            "solver": solver,
            "weight_packed": weight_packed,
            "weight_shape": [int(weight_q.shape[0]), int(logical_width)],
            "padded_in_features": int(weight_packed.shape[1] * 2),
            "weight_storage": "signed-int4-low-high-nibble",
            "weight_scale": weight_scale.float().cpu(),
            "transform": transform,
            "permutation": None if permutation is None else permutation.long().cpu(),
            "rotation_blocks": None if rotations is None else rotations.half().cpu(),
            "activation_granularity": activation_granularity,
            "activation_scale": activation_scale,
            "bias": (
                None
                if getattr(target.module, "bias", None) is None
                else target.module.bias.detach().half().cpu()
            ),
        }
        record["sha256"] = tensor_record_sha256(record)
        records[target.name] = record

    manifest: dict[str, Any] = {
        "format_version": PACK_FORMAT_VERSION,
        "algorithm": "foldquant-baseline-w4a4",
        "method": calibration_manifest.get("method"),
        "model_type": "Gr00tN1d7",
        "suite": calibration_manifest["suite"],
        "checkpoint": checkpoint,
        "checkpoint_revision": checkpoint_revision,
        "source_revision": source_revision,
        "config_sha256": config_hash,
        "calibration_sha256": file_sha256(calibration_path),
        "calibration_run_id": calibration_manifest["run_id"],
        "weight_bits": WEIGHT_BITS,
        "activation_bits": ACTIVATION_BITS,
        "signed_range": [-7, 7],
        "weight_storage": "two-signed-int4-values-per-byte",
        "execution": "emulated: INT4 codes dequantised before a BF16 F.linear; no INT4 kernel",
        "runtime_compatible_backends": ["fake"],
        "scopes": list(selected_scopes),
        "include_vit_mergers": include_vit_mergers,
        "include_vit_patch_embed": include_vit_patch_embed,
        "high_precision_ops": [
            "embedding",
            "layernorm",
            "rotary",
            "attention_qk_av",
            "softmax",
            "adaln_modulation",
            "vl_self_attention",
            "cross_attention_kv",
        ],
        "num_inference_timesteps": int(calibration_manifest["num_inference_timesteps"]),
        "llm_layers": summary.llm_layers,
        "dit_layers": summary.dit_layers,
        "vit_layers": summary.vit_layers,
        "llm_linears": summary.llm_linears,
        "dit_linears": summary.dit_linears,
        "vit_linears": summary.vit_linears,
        "vit_patch_convs": summary.vit_patch_convs,
        "total_linears": summary.total_linears,
        "total_quantized_modules": summary.total_modules,
        "rotation": calibration_manifest["rotation"],
        "rotation_mode": calibration_manifest.get("rotation_mode", "svd_hadamard"),
        "rotation_block_size": calibration_manifest["rotation_block_size"],
        "permutation": calibration_manifest["permutation"],
        "llm_solver": "gptq" if "llm" in selected_scopes else None,
        "vit_solver": "gptq-linear-rtn-patch" if "vit" in selected_scopes else None,
        "gptq_block_size": gptq_block_size,
        "gptq_damping": damping,
        "dit_solver": "rtn" if "dit" in selected_scopes else None,
        "llm_activation_granularity": llm_granularity,
        "dit_activation_granularity": dit_granularity,
        "activation_percentile": calibration_manifest["activation_percentile"],
        "seed": calibration_manifest["seed"],
    }
    for key in ("num_calib", "calibration_seed", "dataset", "calibration_source"):
        if key in calibration_manifest:
            manifest[key] = calibration_manifest[key]
    if extra_manifest:
        manifest.update(extra_manifest)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest": manifest, "layers": records}, output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        file_sha256(output_path) + "\n", encoding="ascii"
    )
    return output_path


#: The fork's function name.
build_holoq_pack = build_pack
