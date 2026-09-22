# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Writing a plugin-node graph over the float graph the export driver produced.

Every FoldQuant scheme overwrites its module's ONNX **in place**: the
driver runs ``torch.onnx.export`` first, then hands the scheme that path (see
``foldquant/export.py``). The float graph's
external-data sidecar is still sitting next to it at that point, and
``onnx.save_model`` *appends* to an existing sidecar rather than truncating it.

Appending is silently destructive: the new tensors land behind the dead float
weights. On GR00T's DiT that pushed 18 MB of live tensors past the 2 GiB mark
(behind 2.18 GB of float weights) and TensorRT's parser rejected the model with
``Trying to access weights or a null tensor!``, although the graph was valid.

Constructs ONNX only: no ``tensorrt`` import, no ``.so`` load, no
``foldquant.runtime`` import.
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
