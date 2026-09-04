# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Locate the upstream Isaac GR00T tree this integration sits in."""

from __future__ import annotations

from pathlib import Path
import sys


#: ``models/groot_n1_6`` — the trimmed upstream ``n1.6.1-release`` checkout.
UPSTREAM_ROOT = Path(__file__).resolve().parents[1]
#: Upstream's DiT export / TensorRT build / timing tools (plain scripts, not a
#: package — importable only with this directory on ``sys.path``).
DEPLOYMENT_DIR = UPSTREAM_ROOT / "scripts" / "deployment"

#: The two modules FoldQuant replaces, the graph each is exported to and the
#: engine it is served from. N1.6's upstream deployment path compiles the DiT
#: only (``export_onnx_n1d6.py`` writes ``dit_model.onnx``); the LLM engine is
#: FoldQuant's own.
COMPONENTS = (
    ("llm", "llm_bf16.onnx", "llm_bf16.engine"),
    ("dit", "dit_bf16.onnx", "dit_bf16.engine"),
)
#: Name of the float DiT graph upstream's ``export_onnx_n1d6.py`` writes.
UPSTREAM_DIT_ONNX = "dit_model.onnx"

#: Names of the FoldQuant export manifest, the shape-hint file the export
#: writes beside it, and the record :mod:`.build_engines` leaves in the engine
#: directory.
MANIFEST_NAME = "foldquant_export.json"
EXPORT_METADATA_NAME = "export_metadata.json"
ENGINES_RECORD_NAME = "foldquant_engines.json"


def ensure_deployment_on_path() -> None:
    """Make ``scripts/deployment`` importable (idempotent)."""
    d = str(DEPLOYMENT_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
