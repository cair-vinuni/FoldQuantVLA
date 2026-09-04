# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Locate the upstream Evo-1 tree this integration sits in."""

from __future__ import annotations

from pathlib import Path

#: ``models/evo_1`` — the trimmed upstream Evo-1 checkout.
UPSTREAM_ROOT = Path(__file__).resolve().parents[1]

#: Upstream's LIBERO client, run unmodified by :mod:`.eval_libero`.
LIBERO_CLIENT = UPSTREAM_ROOT / "LIBERO_evaluation" / "libero_client_4tasks.py"

#: The released LIBERO checkpoint the paper arms are built from.
LIBERO_CHECKPOINT = "MINT-SJTU/Evo1_LIBERO"

#: Normalizer keys, left empty so they are read off the checkpoint's own
#: ``norm_stats.json`` (see :func:`.calibration.resolve_norm_keys`). Upstream's
#: server has them written into its ``__main__`` for the checkpoint its authors
#: served, and a different release keys its stats differently — the released
#: LIBERO checkpoint has one arm, ``libero_robot``, and no dataset level at all.
LIBERO_ARM_KEY = ""
LIBERO_DATASET_KEY = ""

#: The four LIBERO suites, in the order upstream's client walks them.
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

#: The two modules FoldQuant replaces, the graph each is exported to and the
#: engine it is served from. ``llm`` is the InternVL3 (Qwen2) language tower run
#: once per observation over the image + prompt tokens, whose hidden states are
#: the context the head cross-attends; ``action_head`` is one denoise step of
#: the flow-matching head, run once per Euler step (50 by default upstream).
COMPONENTS = (
    ("llm", "llm_bf16.onnx", "llm_bf16.engine"),
    ("action_head", "action_head_bf16.onnx", "action_head_bf16.engine"),
)

#: Names of the FoldQuant export manifest, the shape-hint file the export
#: writes beside it, and the record :mod:`.build_engines` leaves in the engine
#: directory.
MANIFEST_NAME = "foldquant_export.json"
EXPORT_METADATA_NAME = "export_metadata.json"
ENGINES_RECORD_NAME = "foldquant_engines.json"
