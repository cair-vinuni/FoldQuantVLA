# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Compile a SmolVLA engine directory from FoldQuant graphs.

The graphs are compiled with :func:`foldquant.runtime.builder.build_engine`
(STRONGLY_TYPED — the plugin nodes declare their own dtypes), after the plugin
libraries the export manifest names are loaded (built for this device when no
cached binary matches). Both graphs carry one symbolic axis — the LLM's ``seq_len`` and the expert's
``prefix_len``, which are the same length — and both are profiled over the
range the export derived, not pinned at the length it happened to capture: a
SmolVLA prompt is tokenized to its own length, so the prefix changes with the
task string.

A module the export left float (``--expert-scheme none`` / ``--llm-scheme
none``) stays in PyTorch.

Example::

    python -m foldquant_integration.build_engines \\
        --onnx-dir exports/smolvla_w4a4/onnx --engine-dir exports/smolvla_w4a4/engines
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from foldquant.provenance import public_path
from foldquant.runtime.builder import build_engine, profiles_from_onnx
from foldquant.runtime.plugins import prepare_plugins

from ._upstream import COMPONENTS, ENGINES_RECORD_NAME, EXPORT_METADATA_NAME, MANIFEST_NAME

logger = logging.getLogger("foldquant.smolvla.build")


@dataclass
class BuildConfig:
    onnx_dir: str
    """The FoldQuant export's ``onnx/`` directory (holds foldquant_export.json and export_metadata.json)."""

    engine_dir: str
    """Destination engine directory."""

    workspace_mb: int = 8192


def load_manifest(onnx_dir: Path) -> dict[str, Any]:
    return json.loads((onnx_dir / MANIFEST_NAME).read_text())


def dim_ranges(metadata: dict[str, Any]) -> dict[str, Any]:
    """``{dim_name: (min, opt, max)}`` for the symbolic dimensions the graphs carry.

    The prefix length is the one range that matters, and it is a range rather
    than a constant: a SmolVLA prompt is tokenized to its own length (capped at
    ``tokenizer_max_length``), so every task string gives the LLM a different
    sequence and the expert a different KV stack. The export records the bounds
    it derived from a captured call; an export that predates them falls back to
    pinning, which only serves the one length it saw.
    """
    prefix_len = int(metadata["prefix_len"])
    low = int(metadata.get("prefix_len_min", prefix_len))
    high = int(metadata.get("prefix_len_max", prefix_len))
    if not low <= prefix_len <= high:
        raise ValueError(f"captured prefix {prefix_len} outside the recorded bounds [{low}, {high}]")
    span = (low, prefix_len, high)
    return {
        "batch": 1,
        "batch_size": 1,
        "prefix_len": span,
        "seq_len": span,
        "seq_len2": span,
    }


def build(args: BuildConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    onnx_dir = Path(args.onnx_dir)
    engine_dir = Path(args.engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = onnx_dir / EXPORT_METADATA_NAME
    if not metadata_path.is_file():
        raise SystemExit(f"captured shapes not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    ranges = dim_ranges(metadata)
    logger.info("dimension ranges: %s", ranges)

    manifest = load_manifest(onnx_dir)
    plugin_libs: list[str] = list(manifest["plugin_libs"])
    if plugin_libs:
        prepare_plugins(plugin_libs)

    record: dict[str, Any] = {
        "onnx_dir": public_path(str(onnx_dir)),
        "metadata": metadata,
        "dim_ranges": {k: list(v) if isinstance(v, tuple) else v for k, v in ranges.items()},
        "plugin_libs": plugin_libs,
        "schemes": dict(manifest["schemes"]),
        "components": {},
    }
    for name, _onnx_name, engine_name in COMPONENTS:
        if name not in manifest["files"]:
            logger.info("%s: not exported, stays in PyTorch", name)
            continue
        src = onnx_dir / manifest["files"][name]
        if not src.is_file():
            raise FileNotFoundError(src)
        profiles = profiles_from_onnx(src, ranges)
        t0 = time.time()
        logger.info("%s: building %s from %s", name, engine_name, src.name)
        is_float = manifest.get("schemes", {}).get(name) == "float"  # traced float graph: weakly typed, no plugins
        build_engine(
            src,
            engine_dir / engine_name,
            profiles=profiles,
            plugin_libs=() if is_float else plugin_libs,
            strongly_typed=not is_float,
            int8=not is_float,
            workspace_mb=args.workspace_mb,
        )
        record["components"][name] = {
            "onnx": str(src),
            "engine": engine_name,
            "source": "foldquant",
            "profiles": {k: {"min": v.min, "opt": v.opt, "max": v.max} for k, v in profiles.items()},
            "seconds": round(time.time() - t0, 1),
        }
        logger.info("%s: built in %.0fs", name, time.time() - t0)
    if not record["components"]:
        raise SystemExit("nothing to build")

    (engine_dir / ENGINES_RECORD_NAME).write_text(json.dumps(record, indent=2))
    shutil.copy2(onnx_dir / MANIFEST_NAME, engine_dir / MANIFEST_NAME)
    logger.info("engine directory complete: %s", engine_dir)
    return engine_dir


if __name__ == "__main__":
    build(tyro.cli(BuildConfig))
