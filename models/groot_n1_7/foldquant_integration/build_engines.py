# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Compile a complete N1.7 engine directory with FoldQuant graphs in it.

Upstream's ``build_tensorrt_engine.build_engine`` already does everything a
FoldQuant graph needs (STRONGLY_TYPED network, shape profiles derived from
the ONNX dim names) except knowing the plugins. This tool loads the plugin
libraries the export manifest names (building them for this device when no
cached binary matches), then builds each of the seven upstream pipeline
components from its graph in ``--onnx-dir``: the FoldQuant graphs and the
float components ``export`` writes beside them (``quantize`` writes both for
an arm with a float or ModelOpt tower).

A ModelOpt Q/DQ baseline graph (``modelopt_w8a8_smoothquant``) goes through
the same strongly-typed build, which is also how the recipe it reproduces
builds it.

The result is a directory ``trt_model_forward.setup_tensorrt_engines`` loads
as ``n17_full_pipeline``, after the same plugins are loaded in-process, which
:mod:`.verify`, :mod:`.eval_libero`, :mod:`.rollout` and :mod:`.benchmark` do.

Example::

    python -m foldquant_integration.build_engines \\
        --onnx-dir exports/n17_w4a4/onnx --engine-dir exports/n17_w4a4/engines
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Dict, List, Optional

from foldquant.provenance import public_path
from foldquant.runtime.plugins import prepare_plugins
import tyro

from ._upstream import (
    EXPORT_METADATA_NAME,
    MANIFEST_NAME,
    PIPELINE_COMPONENTS,
    ensure_deployment_on_path,
)


logger = logging.getLogger("foldquant.groot_n1_7.build")


@dataclass
class BuildConfig:
    onnx_dir: str
    """The ``onnx/`` directory ``export`` wrote: the FoldQuant graphs, the five float
    components and foldquant_export.json."""

    engine_dir: str
    """Destination engine directory."""

    workspace_mb: int = 8192
    """TensorRT workspace, MB."""

    max_batch: int = 8
    """Upper bound of the symbolic batch axis in every optimization profile (upstream's default).
    The engine reserves activation memory for this bound (the W4A4 DiT graph takes ~3 GB at
    8 against ~6 MB at 1), so pass ``--max-batch 1`` for a single-robot, batch-1 deployment."""

    verbose: bool = False
    """Full TensorRT builder log (upstream default); off keeps warnings and errors."""


def _pin_batch_axis(onnx_path: Path, mins: Dict, opts: Dict, maxs: Dict, max_batch: int):
    """Give every leading dynamic axis named ``batch`` the profile (1, 1, max_batch)."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if not dims or dims[0].dim_value > 0 or dims[0].dim_param != "batch" or inp.name not in opts:
            continue
        mins[inp.name] = (1,) + tuple(mins[inp.name][1:])
        opts[inp.name] = (1,) + tuple(opts[inp.name][1:])
        maxs[inp.name] = (max_batch,) + tuple(maxs[inp.name][1:])
    return mins, opts, maxs


def load_manifest(onnx_dir: Path) -> dict:
    path = onnx_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path}: is {onnx_dir} a FoldQuant export directory?")
    return json.loads(path.read_text())


def shape_hints(onnx_dir: Path) -> Dict[str, int]:
    """The ``opt_seq_lens`` upstream's ``build_full_pipeline`` derives from ``export_metadata.json``."""
    meta = json.loads((onnx_dir / EXPORT_METADATA_NAME).read_text())
    return {
        "sa_seq_len": meta["sa_seq_len"],
        "vl_seq_len": meta["vl_seq_len"],
        "sequence_length": meta["llm_seq_len"],
        "seq_len": meta["llm_seq_len"],
        "num_patches": meta.get("num_patches", 256),
        "num_merged_patches": meta.get("num_merged_patches", 64),
        "num_vis_tokens": meta.get("num_vis_tokens", 64),
    }


def _first_existing(directory: Path, names) -> Optional[Path]:
    for n in names:
        if (directory / n).is_file():
            return directory / n
    return None


def build(args: BuildConfig) -> Dict[str, str]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    onnx_dir = Path(args.onnx_dir)
    engine_dir = Path(args.engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(onnx_dir)
    libs: List[str] = list(manifest["plugin_libs"])
    loaded = prepare_plugins(libs)
    logger.info("plugins loaded: %s", [str(p) for p in loaded])

    ensure_deployment_on_path()
    from build_tensorrt_engine import build_engine, derive_shapes_with_hint
    import tensorrt as trt

    hints = shape_hints(onnx_dir)
    severity = None if args.verbose else trt.Logger.WARNING
    status: Dict[str, str] = {}
    for name, onnx_names, engine_name in PIPELINE_COMPONENTS:
        dst = engine_dir / engine_name
        src = _first_existing(onnx_dir, onnx_names)
        if src is None:
            logger.warning("%s: no ONNX found, skipped", name)
            status[name] = "missing"
            continue
        origin = "foldquant" if name in manifest.get("schemes", {}) else "float"
        t0 = time.time()
        mins, opts, maxs = derive_shapes_with_hint(str(src), opt_seq_lens=hints, max_batch=args.max_batch)
        # The FoldQuant DiT graphs declare their batch axis as ``batch``; upstream's profile
        # derivation only recognises ``batch_size`` and otherwise treats the axis as an unknown
        # sequence dimension (opt 256, max 512), which makes the engine reserve ~3 GB of
        # activation memory and optimise for a batch it never sees. Pin it like a batch axis.
        mins, opts, maxs = _pin_batch_axis(src, mins, opts, maxs, args.max_batch)
        for k in opts:
            logger.info("  %s: min=%s opt=%s max=%s", k, mins[k], opts[k], maxs[k])
        build_engine(
            onnx_path=str(src),
            engine_path=str(dst),
            precision="bf16",
            workspace_mb=args.workspace_mb,
            min_shapes=mins,
            opt_shapes=opts,
            max_shapes=maxs,
            trt_severity=severity,
        )
        logger.info("%s: built from %s graph %s in %.0fs", name, origin, src.name, time.time() - t0)
        status[name] = f"built:{origin}:{src}"

    record = {
        "onnx_dir": public_path(str(onnx_dir)),
        "plugin_libs": libs,
        "schemes": manifest["schemes"],
        "shape_hints": hints,
        "components": status,
    }
    (engine_dir / "foldquant_engines.json").write_text(json.dumps(record, indent=2))
    shutil.copy2(onnx_dir / MANIFEST_NAME, engine_dir / MANIFEST_NAME)

    # A component with no ONNX leaves a hole in the directory.
    # Reporting "complete" here is what lets the run continue: the record is
    # written, the caller exits 0, and trt_model_forward.setup_tensorrt_engines
    # then keeps that module in PyTorch with only a print, while verify.py and
    # serve.py still name the scheme from the export manifest. N1.5 and N1.6
    # raise FileNotFoundError in the equivalent position.
    missing = sorted(n for n, st in status.items() if st == "missing")
    if missing:
        raise FileNotFoundError(
            f"{engine_dir} is missing {', '.join(missing)}: no ONNX in {onnx_dir}. `export` writes every "
            "component there (`quantize` does, for an arm with a float or ModelOpt tower)."
        )
    logger.info("engine directory complete: %s", engine_dir)
    return status


if __name__ == "__main__":
    build(tyro.cli(BuildConfig))
