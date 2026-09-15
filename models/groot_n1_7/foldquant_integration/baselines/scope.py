# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Ported from the authors' Isaac-GR00T fork for the FoldQuant release.

"""Strict GR00T-N1.7 LLM, DiT, and Qwen3-VL vision W4A4 scope discovery."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from torch import nn


SUPPORTED_SCOPES = frozenset({"llm", "dit", "vit"})
DEFAULT_SCOPES = ("llm", "dit")

_LLM_RE = re.compile(
    r"(?:^|.*\.)backbone\.model(?:\.model)?\.language_model\.layers\."
    r"(?P<layer>\d+)\."
    r"(?P<projection>self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
_DIT_RE = re.compile(
    r"(?:^|.*\.)action_head\.model\.transformer_blocks\."
    r"(?P<layer>\d+)\."
    r"(?P<projection>attn1\.(?:to_q|to_k|to_v|to_out\.0)|"
    r"ff\.net\.(?:0\.proj|2))$"
)
_VIT_BLOCK_RE = re.compile(
    r"(?:^|.*\.)backbone\.model(?:\.model)?\.visual\.blocks\."
    r"(?P<layer>\d+)\."
    r"(?P<projection>attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$"
)
_VIT_MERGER_RE = re.compile(
    r"(?:^|.*\.)backbone\.model(?:\.model)?\.visual\."
    r"(?P<merger>merger|deepstack_merger_list\.(?P<layer>\d+))\."
    r"(?P<projection>linear_fc1|linear_fc2)$"
)
_VIT_PATCH_RE = re.compile(
    r"(?:^|.*\.)backbone\.model(?:\.model)?\.visual\.patch_embed\."
    r"(?P<projection>proj)$"
)

_LLM_PROJECTIONS = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}
_DIT_PROJECTIONS = {
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
}
_VIT_PROJECTIONS = {"attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"}


def normalize_scopes(scopes: Iterable[str] | str | None) -> tuple[str, ...]:
    if scopes is None:
        normalized = DEFAULT_SCOPES
    elif isinstance(scopes, str):
        normalized = tuple(item.strip().lower() for item in scopes.split(",") if item.strip())
    else:
        normalized = tuple(str(item).strip().lower() for item in scopes if str(item).strip())
    if not normalized:
        raise ValueError("At least one W4A4 scope must be selected")
    unknown = sorted(set(normalized) - SUPPORTED_SCOPES)
    if unknown:
        raise ValueError(
            f"Unsupported W4A4 scopes {unknown}; choose from {sorted(SUPPORTED_SCOPES)}"
        )
    return tuple(scope for scope in ("llm", "dit", "vit") if scope in normalized)


@dataclass(frozen=True)
class QuantTarget:
    name: str
    module: nn.Module
    scope: str
    layer_index: int
    projection: str
    module_kind: str = "linear"


@dataclass(frozen=True)
class ScopeSummary:
    llm_layers: int = 0
    dit_layers: int = 0
    vit_layers: int = 0
    llm_linears: int = 0
    dit_linears: int = 0
    vit_linears: int = 0
    vit_patch_convs: int = 0

    @property
    def total_linears(self) -> int:
        return self.llm_linears + self.dit_linears + self.vit_linears

    @property
    def total_modules(self) -> int:
        return self.total_linears + self.vit_patch_convs


def discover_n1d7_targets(
    model: nn.Module,
    *,
    scopes: Iterable[str] | str | None = None,
    include_vit_mergers: bool = True,
    include_vit_patch_embed: bool = False,
) -> list[QuantTarget]:
    """Discover an explicit, reproducible W4A4 scope.

    Vision LayerNorm, embeddings, RoPE, QK/AV attention matmuls, and softmax stay
    high precision because they are not weighted Linear/patch-projection operators.
    Qwen3-VL's non-overlapping patch Conv3d can optionally be lowered to GEMM.
    """

    selected = set(normalize_scopes(scopes))
    targets: list[QuantTarget] = []
    for name, module in model.named_modules():
        match = None
        scope = ""
        module_kind = "linear"
        if isinstance(module, nn.Linear):
            if "llm" in selected:
                match = _LLM_RE.fullmatch(name)
                scope = "llm"
            if match is None and "dit" in selected:
                match = _DIT_RE.fullmatch(name)
                scope = "dit"
            if match is None and "vit" in selected:
                match = _VIT_BLOCK_RE.fullmatch(name)
                scope = "vit"
            if match is None and "vit" in selected and include_vit_mergers:
                match = _VIT_MERGER_RE.fullmatch(name)
                scope = "vit"
        elif isinstance(module, nn.Conv3d) and "vit" in selected and include_vit_patch_embed:
            match = _VIT_PATCH_RE.fullmatch(name)
            scope = "vit"
            module_kind = "conv3d_patch"
        if match is None:
            continue
        layer = match.groupdict().get("layer")
        targets.append(
            QuantTarget(
                name=name,
                module=module,
                scope=scope,
                layer_index=-1 if layer is None else int(layer),
                projection=match.group("projection"),
                module_kind=module_kind,
            )
        )
    return sorted(targets, key=lambda target: target.name)


def _validate_layer_grid(
    targets: list[QuantTarget], scope: str, expected_projections: set[str], *, required: bool
) -> int:
    by_layer: dict[int, set[str]] = {}
    for target in targets:
        if (
            target.scope == scope
            and target.layer_index >= 0
            and (scope != "vit" or target.projection in _VIT_PROJECTIONS)
        ):
            by_layer.setdefault(target.layer_index, set()).add(target.projection)
    if not by_layer:
        if required:
            raise ValueError(f"No GR00T-N1.7 {scope.upper()} quantization targets were found")
        return 0
    layer_indices = sorted(by_layer)
    expected_indices = list(range(layer_indices[-1] + 1))
    if layer_indices != expected_indices:
        raise ValueError(
            f"{scope.upper()} layer indices are not contiguous from zero: {layer_indices}"
        )
    errors = []
    for index, projections in by_layer.items():
        missing = sorted(expected_projections - projections)
        extra = sorted(projections - expected_projections)
        if missing or extra:
            errors.append(f"layer {index}: missing={missing}, extra={extra}")
    if errors:
        raise ValueError(f"Invalid {scope.upper()} projection grid: " + "; ".join(errors))
    return len(by_layer)


def validate_n1d7_scope(
    targets: list[QuantTarget],
    *,
    expected_llm_layers: int | None = None,
    expected_dit_layers: int | None = None,
    expected_vit_layers: int | None = None,
    scopes: Iterable[str] | str | None = None,
) -> ScopeSummary:
    """Validate complete projection grids for every requested transformer scope."""

    selected = set(normalize_scopes(scopes)) if scopes is not None else {t.scope for t in targets}
    names = [target.name for target in targets]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate quantization target names were discovered")
    llm_layers = _validate_layer_grid(targets, "llm", _LLM_PROJECTIONS, required="llm" in selected)
    dit_layers = _validate_layer_grid(targets, "dit", _DIT_PROJECTIONS, required="dit" in selected)
    vit_layers = _validate_layer_grid(targets, "vit", _VIT_PROJECTIONS, required="vit" in selected)
    for label, actual, expected in (
        ("LLM", llm_layers, expected_llm_layers),
        ("DiT", dit_layers, expected_dit_layers),
        ("ViT", vit_layers, expected_vit_layers),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"Expected {expected} {label} layers, discovered {actual}")
    summary = ScopeSummary(
        llm_layers=llm_layers,
        dit_layers=dit_layers,
        vit_layers=vit_layers,
        llm_linears=sum(t.scope == "llm" and t.module_kind == "linear" for t in targets),
        dit_linears=sum(t.scope == "dit" and t.module_kind == "linear" for t in targets),
        vit_linears=sum(t.scope == "vit" and t.module_kind == "linear" for t in targets),
        vit_patch_convs=sum(t.scope == "vit" and t.module_kind == "conv3d_patch" for t in targets),
    )
    if summary.llm_linears != 7 * llm_layers:
        raise ValueError(f"Expected {7 * llm_layers} LLM linears, found {summary.llm_linears}")
    if summary.dit_linears != 6 * dit_layers:
        raise ValueError(f"Expected {6 * dit_layers} DiT linears, found {summary.dit_linears}")
    minimum_vit_linears = 4 * vit_layers
    if summary.vit_linears < minimum_vit_linears:
        raise ValueError(
            f"Expected at least {minimum_vit_linears} ViT block linears, found {summary.vit_linears}"
        )
    return summary
