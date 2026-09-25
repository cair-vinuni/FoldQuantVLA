# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""GR00T N1.7 adapter for the FoldQuant fake-quant pipeline (:mod:`foldquant.fakequant`).

The pipeline itself lives in ``foldquant``: fake-quant state -> real-quant
plugin ONNX -> TensorRT engines. This module tells it what is GR00T N1.7
specific: how to load the policy, where its quantized modules sit, and how the
engine directory is assembled (the FoldQuant graphs plus upstream's float
components). Run the pipeline from ``models/groot_n1_7``::

    python -m foldquant_integration.export_foldquant ... --save-fakequant exports/w4a4/fakequant
    python -m foldquant.fakequant push --fakequant-dir exports/w4a4/fakequant --repo-id <org>/<name>
    python -m foldquant.fakequant convert --fakequant-dir <dir> --model-path <ckpt> \\
        --output-dir exports/w4a4 --float-onnx-dir exports/float/onnx

``python -m foldquant_integration.fakequant`` is the same command line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from foldquant import fakequant as _pipeline

from ._upstream import EXPORT_METADATA_NAME

FAMILY = "groot_n1_7"


def _tag_value(embodiment_tag: Optional[str]) -> Optional[str]:
    """``EmbodimentTag.LIBERO_PANDA`` (an enum's ``str``) -> its value; anything else unchanged."""
    if embodiment_tag and embodiment_tag.startswith("EmbodimentTag."):
        from gr00t.data.embodiment_tags import EmbodimentTag

        return EmbodimentTag[embodiment_tag.split(".", 1)[1]].value
    return embodiment_tag


def load_policy(model_path: str, embodiment_tag: Optional[str], device: Any = "cuda") -> Any:
    from . import calibration

    return calibration.load_policy(model_path, _tag_value(embodiment_tag), device)


def module_paths(policy: Any) -> Dict[str, Any]:
    """The quantizable modules, at the paths ``export_foldquant`` reads them from."""
    return {
        "llm": policy.model.backbone.model.model.language_model,
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
) -> Dict[str, str]:
    """A complete ``n17_full_pipeline`` engine directory: FoldQuant graphs + float components."""
    from .build_engines import BuildConfig, build

    if not float_onnx_dir and not float_engine_dir:
        raise SystemExit(
            "GR00T N1.7 engines need the float components too: pass --float-onnx-dir (upstream's "
            "build_trt_pipeline.py export) or --float-engine-dir (its engines)"
        )
    return build(
        BuildConfig(
            onnx_dir=str(onnx_dir),
            engine_dir=str(engine_dir),
            float_onnx_dir=float_onnx_dir,
            float_engine_dir=float_engine_dir,
            max_batch=max_batch,
            workspace_mb=workspace_mb,
        )
    )


def save_arm_state(
    directory: Path,
    *,
    model_path: str,
    results: List[Any],
    export_metadata: Dict[str, Any],
    export_manifest: Dict[str, Any],
    base_model_id: Optional[str] = None,
    bundle: bool = True,
    copy_base: bool = False,
) -> Path:
    """Called by ``export_foldquant --save-fakequant``."""
    path = _pipeline.save_arm_state(
        directory,
        family=FAMILY,
        model_path=model_path,
        results=results,
        export_manifest=export_manifest,
        extra_files={EXPORT_METADATA_NAME: export_metadata},
        base_model_id=base_model_id,
        bundle=bundle,
        copy_base=copy_base,
    )
    # The export manifest records the tag as the enum's str; the state records a value
    # the policy loader accepts, so `to-onnx` needs no --embodiment-tag.
    manifest_path = Path(directory) / "foldquant_quant.json"
    import json

    data = json.loads(manifest_path.read_text())
    data["embodiment_tag"] = _tag_value(data.get("embodiment_tag"))
    manifest_path.write_text(json.dumps(data, indent=2))
    return path


def install(policy: Any, fakequant_dir: Union[str, Path], model_path: Optional[str] = None, *, check_checkpoint: bool = True) -> Any:
    """Install a fake-quant arm on a loaded ``Gr00tPolicy``; returns ``(handle, state)``."""
    return _pipeline.install_on_policy(policy, fakequant_dir, model_path, check_checkpoint=check_checkpoint)


def load_state_for(fakequant_dir: Union[str, Path], model_path: Optional[str] = None, *, check_checkpoint: bool = True) -> Any:
    """Read an N1.7 quant state and check it belongs to *model_path*."""
    from foldquant.quant_state import load_state

    state = load_state(fakequant_dir)
    if state.manifest.get("family") != FAMILY:
        raise ValueError(f"{fakequant_dir} is a {state.manifest.get('family')!r} quant state, not {FAMILY!r}")
    if check_checkpoint:
        _pipeline.verify_base_checkpoint(state, _pipeline.resolve_model_path(state, fakequant_dir, model_path))
    return state


if __name__ == "__main__":
    _pipeline.main()
