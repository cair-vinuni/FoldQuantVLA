# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Locate the upstream openpi tree this integration sits in."""

from __future__ import annotations

from pathlib import Path

#: ``models/pi05`` — the trimmed upstream openpi checkout.
UPSTREAM_ROOT = Path(__file__).resolve().parents[1]

#: The upstream training config the released LIBERO checkpoint was fine-tuned
#: and served with (``examples/libero/README.md``: ``pi05_libero``). It names
#: the model variant, the data transforms and the norm-stats asset id, so every
#: tool takes ``--config`` and defaults to this one.
LIBERO_TRAIN_CONFIG = "pi05_libero"

#: The two modules FoldQuant replaces, the graph each is exported to and the
#: engine it is served from. ``llm`` is the PaliGemma language model run once
#: per observation over the image + prompt prefix (its KV cache is the only
#: thing that crosses the engine boundary); ``expert`` is the Gemma-300M action
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
