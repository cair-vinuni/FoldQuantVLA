# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""groot_n1_6 adapter for the FoldQuant fake-quant pipeline (:mod:`foldquant.quantized`).

Run from ``models/groot_n1_6``::

    python -m foldquant_integration.quantize ... --output-dir exports/<arm>/quantized
    python -m foldquant_integration.export --model-path <dir> --output-dir exports/<arm>/onnx
    python -m foldquant_integration.build_engines --onnx-dir exports/<arm>/onnx --engine-dir exports/<arm>/engines
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from foldquant import quantized as _pipeline

from ._upstream import EXPORT_METADATA_NAME

FAMILY = "groot_n1_6"


def load_policy(model_path: str, embodiment_tag: Optional[str], device: Any = "cuda", **load_kwargs: Any) -> Any:
    from . import calibration

    return calibration.load_policy(model_path, embodiment_tag, device)


def module_paths(policy: Any) -> Dict[str, Any]:
    """The quantizable modules, at the paths ``quantize`` reads them from."""

    return {
        "llm": policy.model.backbone.model.language_model,
        "dit": policy.model.action_head.model,
    }


def build_engines(
    onnx_dir: Path,
    engine_dir: Path,
    *,
    max_batch: int = 1,
    workspace_mb: int = 8192,
) -> Any:
    from .build_engines import BuildConfig, build

    return build(BuildConfig(engine_dir=str(engine_dir), onnx_dir=str(onnx_dir), max_batch=max_batch, workspace_mb=workspace_mb))

def checkpoint_root(policy: Any) -> Any:
    """The module whose parameter names are the checkpoint's tensor keys."""
    return policy.model


def save_arm_state(
    directory: Path,
    *,
    policy: Any,
    model_path: str,
    results: List[Any],
    export_metadata: Dict[str, Any],
    export_manifest: Dict[str, Any],
    base_model_id: Optional[str] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    """Called by ``quantize``."""
    return _pipeline.save_arm_state(
        directory,
        family=FAMILY,
        model_path=model_path,
        results=results,
        export_manifest=export_manifest,
        extra_files={EXPORT_METADATA_NAME: export_metadata},
        base_model_id=base_model_id,
        load_kwargs=load_kwargs,
        modules=module_paths(policy),
        checkpoint_root=checkpoint_root(policy),
    )


if __name__ == "__main__":
    _pipeline.main()
