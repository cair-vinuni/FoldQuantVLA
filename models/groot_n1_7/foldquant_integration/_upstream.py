# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Locate the upstream Isaac GR00T tree this integration sits in."""

from __future__ import annotations

from pathlib import Path
import sys


#: ``models/groot_n1_7`` — the trimmed upstream ``n1.7-release`` checkout.
UPSTREAM_ROOT = Path(__file__).resolve().parents[1]
#: Upstream's ONNX export / TensorRT build / verification tools (plain scripts,
#: not a package — importable only with this directory on ``sys.path``).
DEPLOYMENT_DIR = UPSTREAM_ROOT / "scripts" / "deployment"

#: Engine file each upstream pipeline component is loaded from
#: (``trt_model_forward._setup_n17_full_pipeline``), and the ONNX file the
#: upstream export writes for it. Order is the upstream build order.
PIPELINE_COMPONENTS = (
    ("vit", ("vit_fp32.onnx", "vit_bf16.onnx"), "vit_bf16.engine"),
    ("llm", ("llm_bf16.onnx",), "llm_bf16.engine"),
    ("vl_self_attention", ("vl_self_attention.onnx",), "vl_self_attention.engine"),
    ("state_encoder", ("state_encoder.onnx",), "state_encoder.engine"),
    ("action_encoder", ("action_encoder.onnx",), "action_encoder.engine"),
    ("dit", ("dit_bf16.onnx",), "dit_bf16.engine"),
    ("action_decoder", ("action_decoder.onnx",), "action_decoder.engine"),
)

#: Names of the FoldQuant export manifest and of upstream's shape-hint file.
MANIFEST_NAME = "foldquant_export.json"
EXPORT_METADATA_NAME = "export_metadata.json"


def ensure_deployment_on_path() -> None:
    """Make ``scripts/deployment`` importable (idempotent)."""
    d = str(DEPLOYMENT_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)


#: The LIBERO checkout the release pins as a submodule. Its ``libero/`` package
#: has no ``__init__.py``, so an editable install leaves nothing importable and
#: ``import libero`` fails even after ``pip install -e`` reports success; the
#: directory has to be on ``sys.path`` instead.
LIBERO_DIR = UPSTREAM_ROOT / "external_dependencies" / "LIBERO"


def ensure_libero_on_path() -> None:
    """Make the pinned LIBERO checkout importable (idempotent).

    Raises with the submodule command rather than letting the rollout fail on a
    bare ModuleNotFoundError three frames deep in upstream's env registry.
    """
    if not (LIBERO_DIR / "libero").is_dir():
        raise RuntimeError(
            f"{LIBERO_DIR} is empty — initialise the pinned LIBERO first:\n"
            f"    git submodule update --init {LIBERO_DIR.relative_to(UPSTREAM_ROOT.parent.parent)}"
        )
    d = str(LIBERO_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
