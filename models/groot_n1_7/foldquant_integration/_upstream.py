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
