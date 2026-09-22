# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Save ONNX graphs without appending to stale external weights.

Plugin exports can overwrite a float graph at the same path. Because
``onnx.save_model`` appends to an existing sidecar, remove it before saving;
otherwise unused float weights inflate the file and can exceed parser limits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["save_plugin_onnx"]


def save_plugin_onnx(model: Any, output_path: Path | str, **save_kwargs: Any) -> Path:
    """Save *model* to *output_path* with a freshly written external-data sidecar.

    Args:
        model: The ``onnx.ModelProto`` to write.
        output_path: Destination ``.onnx`` path. Any sidecar left by a previous
            write of this path is removed first.
        **save_kwargs: Extra ``onnx.save_model`` options (e.g. ``size_threshold``,
            ``convert_attribute``). The external-data options are fixed.

    Returns:
        The saved ONNX ``Path``.
    """
    import onnx

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sidecar = out_path.with_name(out_path.name + ".data")
    # Truncate rather than append: see the module docstring.
    sidecar.unlink(missing_ok=True)

    onnx.save_model(
        model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=sidecar.name,
        **save_kwargs,
    )
    return out_path
