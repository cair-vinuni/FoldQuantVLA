# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Locate the upstream Isaac GR00T tree this integration sits in."""

from __future__ import annotations

import sys
from pathlib import Path

#: ``models/groot_n1_5`` — the trimmed upstream ``n1.5-release`` checkout.
UPSTREAM_ROOT = Path(__file__).resolve().parents[1]
#: Upstream's fp16 / FP8 TensorRT path (plain scripts, not a package). Its
#: DiT graph takes three inputs (``sa_embs``, ``vl_embs``, ``timesteps_tensor``)
#: and its LLM graph is fp16, so neither is interchangeable with a FoldQuant
#: engine; the directory is referenced for documentation only.
DEPLOYMENT_DIR = UPSTREAM_ROOT / "deployment_scripts"

#: The data config upstream's LIBERO fine-tunes were trained and served with
#: (``examples/Libero/README.md``). N1.5 keeps the modality config and the
#: transforms in code, not in the checkpoint, so every tool takes
#: ``--data-config`` and defaults to this one.
LIBERO_DATA_CONFIG = "examples.Libero.custom_data_config:LiberoDataConfig"

#: The two modules FoldQuant replaces, the graph each is exported to and the
#: engine it is served from.
COMPONENTS = (
    ("llm", "llm_bf16.onnx", "llm_bf16.engine"),
    ("dit", "dit_bf16.onnx", "dit_bf16.engine"),
)

#: Names of the FoldQuant export manifest, the shape-hint file the export
#: writes beside it, and the record :mod:`.build_engines` leaves in the engine
#: directory.
MANIFEST_NAME = "foldquant_export.json"
EXPORT_METADATA_NAME = "export_metadata.json"
ENGINES_RECORD_NAME = "foldquant_engines.json"


def ensure_upstream_on_path() -> None:
    """Make ``examples.Libero.*`` importable (idempotent).

    Upstream addresses its LIBERO data config and evaluation helpers as
    ``examples.Libero...`` — a package that exists only relative to the
    repository root, which upstream's own scripts assume is the working
    directory.
    """
    root = str(UPSTREAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
