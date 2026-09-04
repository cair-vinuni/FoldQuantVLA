# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Make the FoldQuant TensorRT plugins available to this process.

TensorRT resolves plugin creators (namespace / name / version) when the ONNX
parser sees a plugin node and again when an engine is deserialized — so the
``.so`` must be ``dlopen``-ed before either. Two call sites, two policies:

* :func:`prepare_plugins` — at BUILD time: resolve the device-matched binary,
  rebuilding from the vendored sources into the cache when this device matches
  nothing committed, then load.
* :func:`load_plugins` — at RUN time: resolve and load only; a missing binary is
  a deployment error (:class:`foldquant.kernels.locator.MissingPluginError`),
  never a silent mid-inference rebuild.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from ..kernels import locator

logger = logging.getLogger(__name__)

__all__ = ["load_plugins", "prepare_plugins", "warn_tensor_core_fit"]


def warn_tensor_core_fit(libnames: Iterable[str]) -> None:
    """Log a warning for every lib whose tensor-core path this GPU lacks.

    A W4A4 engine on a device without s4 tensor cores (Hopper) is correct but
    ~15x slower, and the mismatch is otherwise silent. Best effort: a probe
    failure logs at debug and never blocks a build.
    """
    try:
        import torch

        from ..kernels.registry import tensor_core_support_warning

        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        for lib in libnames:
            msg = tensor_core_support_warning(lib, cap)
            if msg:
                logger.warning("%s", msg)
    except Exception:
        logger.debug("tensor-core support probe failed; continuing.", exc_info=True)


def prepare_plugins(libnames: Iterable[str]) -> List[Path]:
    """Resolve (rebuilding if needed) and load *libnames*; returns the loaded paths."""
    names = list(dict.fromkeys(libnames))
    if not names:
        return []
    from ..kernels.build import ensure_plugins_for_current_device

    warn_tensor_core_fit(names)
    paths = ensure_plugins_for_current_device(names)
    locator.load_plugin_libs(paths)
    return paths


def load_plugins(libnames: Iterable[str]) -> List[Path]:
    """Resolve and load *libnames* without rebuilding; raises ``MissingPluginError`` on a miss."""
    names = list(dict.fromkeys(libnames))
    if not names:
        return []
    return locator.load_required_plugins(names)
