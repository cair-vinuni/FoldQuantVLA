# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""groot_n1_6 adapter for the FoldQuant fake-quant pipeline (:mod:`foldquant.fakequant`).

Run from ``models/groot_n1_6``::

    python -m foldquant_integration.export_foldquant ... --save-fakequant exports/<arm>/fakequant
    python -m foldquant.fakequant convert --fakequant-dir <dir> --model-path <ckpt> --output-dir exports/<arm>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from foldquant import fakequant as _pipeline

from ._upstream import EXPORT_METADATA_NAME

FAMILY = "groot_n1_6"


def load_policy(model_path: str, embodiment_tag: Optional[str], device: Any = "cuda", **load_kwargs: Any) -> Any:
    from . import calibration

    return calibration.load_policy(model_path, embodiment_tag, device)


def module_paths(policy: Any) -> Dict[str, Any]:
    """The quantizable modules, at the paths ``export_foldquant`` reads them from."""

    return {
        "llm": policy.model.backbone.model.language_model,
        "dit": policy.model.action_head.model,
    }


def build_engines(
    onnx_dir: Path,
    engine_dir: Path,
    *,
    float_onnx_dir: Optional[str] = None,
    float_engine_dir: Optional[str] = None,
    max_batch: int = 1,
    workspace_mb: int = 8192,
) -> Any:
    from .build_engines import BuildConfig, build

    if float_engine_dir:
        raise SystemExit("groot_n1_6 builds every engine it installs from ONNX; --float-engine-dir does not apply")
    return build(BuildConfig(engine_dir=str(engine_dir), onnx_dir=str(onnx_dir), float_onnx_dir=float_onnx_dir,
                             max_batch=max_batch, workspace_mb=workspace_mb))


def save_arm_state(
    directory: Path,
    *,
    model_path: str,
    results: List[Any],
    export_metadata: Dict[str, Any],
    export_manifest: Dict[str, Any],
    base_model_id: Optional[str] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
    bundle: bool = True,
    copy_base: bool = False,
) -> Path:
    """Called by ``export_foldquant --save-fakequant``."""
    return _pipeline.save_arm_state(
        directory,
        family=FAMILY,
        model_path=model_path,
        results=results,
        export_manifest=export_manifest,
        extra_files={EXPORT_METADATA_NAME: export_metadata},
        base_model_id=base_model_id,
        load_kwargs=load_kwargs,
        bundle=bundle,
        copy_base=copy_base,
    )


if __name__ == "__main__":
    _pipeline.main()
