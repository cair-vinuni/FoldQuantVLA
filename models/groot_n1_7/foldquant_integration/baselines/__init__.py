# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Ported from the authors' Isaac-GR00T fork for the FoldQuant release.

"""Emulated W4A4 baselines for GR00T-N1.7: HoloQ-style and DuQuant-style.

Both arms are post-training fake quantization of the 112 LLM + 192 DiT
projection Linears (zigzag permutation + block-64 rotation, GPTQ / RTN
weights, INT4 activations), differing only in the rotation and in how the
activation scale is chosen — see :data:`.calibration.METHODS`. Nothing here
runs an INT4 kernel; the arms exist so a closed-loop comparison against the
FoldQuant engines can be reproduced from the release alone.
"""

from .builder import build_pack
from .calibration import METHODS, BaselineCalibrationCollector
from .context import (
    DiTStepContext,
    get_dit_quant_step,
    get_dit_quant_total_steps,
    install_dit_step_context,
    set_dit_quant_step,
)
from .runtime import BaselineLinear, BaselinePatchConv3d, apply_pack, load_pack
from .scope import discover_n1d7_targets, normalize_scopes, validate_n1d7_scope


__all__ = [
    "METHODS",
    "BaselineCalibrationCollector",
    "BaselineLinear",
    "BaselinePatchConv3d",
    "DiTStepContext",
    "apply_pack",
    "build_pack",
    "discover_n1d7_targets",
    "get_dit_quant_step",
    "get_dit_quant_total_steps",
    "install_dit_step_context",
    "load_pack",
    "normalize_scopes",
    "set_dit_quant_step",
    "validate_n1d7_scope",
]
