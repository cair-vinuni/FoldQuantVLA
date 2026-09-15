# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Custom TensorRT plugins: registry, device-matched resolution, and the CMake build.

Import-light on purpose: nothing here pulls in ``torch`` or ``tensorrt`` at
module scope, so the registry can be consulted (and the build CLI run) in an
environment that has neither.
"""

from .locator import INT4_GROUPWISE_LIB, INT4_PER_ROW_LIB, INT8_PER_ROW_LIB, KNOWN_PLUGIN_LIBS, MissingPluginError
from .registry import TENSORRT_KERNELS, KernelSpec, kernel_spec

__all__ = [
    "INT4_GROUPWISE_LIB",
    "INT4_PER_ROW_LIB",
    "INT8_PER_ROW_LIB",
    "KNOWN_PLUGIN_LIBS",
    "KernelSpec",
    "MissingPluginError",
    "TENSORRT_KERNELS",
    "kernel_spec",
]
