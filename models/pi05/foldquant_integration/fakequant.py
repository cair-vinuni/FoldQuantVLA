# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""π₀.₅ adapter for the FoldQuant fake-quant pipeline (:mod:`foldquant.fakequant`).

The state converts to the real-quant ONNX graphs and engines like any family.
The PyTorch fake-quant covers both quantized modules: the Gemma prefix LLM and
the action expert (:mod:`foldquant.expert_fake_quant`, reached through
:class:`.runtime.Pi05ExpertView`, whose submodules are the policy's own).
``serve``, ``eval_libero`` and ``verify`` run a fake-quant model given as
``--checkpoint-dir``, or a state given with ``--fakequant-dir``.
Run from ``models/pi05``::

    python -m foldquant_integration.export_foldquant ... --save-fakequant exports/<arm>/fakequant
    python -m foldquant.fakequant convert --fakequant-dir <dir> --model-path <checkpoint dir> \
        --output-dir exports/<arm>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foldquant import fakequant as _pipeline

from ._upstream import EXPORT_METADATA_NAME

FAMILY = "pi05"


def load_policy(model_path: str, embodiment_tag: str | None, device: Any = "cuda", **load_kwargs: Any) -> Any:
    """``embodiment_tag`` does not apply; ``config_name`` comes from the state."""
    from . import calibration

    return calibration.load_policy(model_path, device=device, compile=False, **load_kwargs)


def module_paths(policy: Any) -> dict[str, Any]:
    from .runtime import expert_view, llm_module

    return {"llm": llm_module(policy), "expert": expert_view(policy)}


def build_engines(
    onnx_dir: Path,
    engine_dir: Path,
    *,
    float_onnx_dir: str | None = None,
    float_engine_dir: str | None = None,
    max_batch: int = 1,
    workspace_mb: int = 8192,
) -> Any:
    from .build_engines import BuildConfig, build

    if float_onnx_dir or float_engine_dir or max_batch != 1:
        raise SystemExit("pi05 engines pin batch 1 and take no float graphs")
    return build(BuildConfig(onnx_dir=str(onnx_dir), engine_dir=str(engine_dir), workspace_mb=workspace_mb))


def save_arm_state(
    directory: Path,
    *,
    model_path: str,
    results: list,
    export_metadata: dict,
    export_manifest: dict,
    base_model_id: str | None = None,
    load_kwargs: dict | None = None,
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
