# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Locate the upstream LeRobot tree this integration sits in."""

from __future__ import annotations

from pathlib import Path

#: ``models/smolvla`` — the trimmed upstream LeRobot checkout.
UPSTREAM_ROOT = Path(__file__).resolve().parents[1]

#: The released LIBERO checkpoint the paper arms are built from. SmolVLA keeps
#: its whole configuration (model variant, normalisation stats, processor
#: pipeline) inside the checkpoint, so unlike openpi there is no separate train
#: config to name — a path or hub id is the only handle a tool needs.
LIBERO_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"

#: The four LIBERO suites, in the order the protocol reports them.
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

#: The two modules FoldQuant replaces, the graph each is exported to and the
#: engine it is served from. ``llm`` is the SmolVLM2 text decoder run once per
#: observation over the image + prompt + state prefix (its KV cache is the only
#: thing that crosses the engine boundary); ``expert`` is the 16-layer action
#: expert run once per Euler step.
COMPONENTS = (
    ("llm", "llm_bf16.onnx", "llm_bf16.engine"),
    ("expert", "expert_bf16.onnx", "expert_bf16.engine"),
)

#: Names of the FoldQuant export manifest, the shape-hint file the export
#: writes beside it, and the record :mod:`.build_engines` leaves in the engine
#: directory.
MANIFEST_NAME = "foldquant_export.json"
EXPORT_METADATA_NAME = "export_metadata.json"
ENGINES_RECORD_NAME = "foldquant_engines.json"
