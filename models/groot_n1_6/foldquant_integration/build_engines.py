# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Compile an N1.6 engine directory from FoldQuant graphs.

The FoldQuant graphs are compiled with :func:`foldquant.runtime.builder.build_engine`
(STRONGLY_TYPED — the plugin nodes declare their own dtypes), after the plugin
libraries the export manifest names are loaded (built for this device when no
cached binary matches). Shape profiles come from the ONNX value infos plus the
captured shapes the export recorded in ``export_metadata.json``; the state +
action stream is static, the vision-language stream and the LLM sequence get a
range around the captured length because the task text differs per task.

A DiT the FoldQuant export left float (``--dit-scheme none``) can be built from
the ``dit_model.onnx`` upstream's ``export_onnx_n1d6.py`` writes
(``--float-onnx-dir``; weakly typed, BF16). Given alone, that directory yields
the float-engine arm — the floor of the drift metric.

Example::

    python -m foldquant_integration.build_engines \\
        --onnx-dir exports/n16_w4a4/onnx --engine-dir exports/n16_w4a4/engines
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional

from foldquant.provenance import public_path
from foldquant.runtime.builder import build_engine, profiles_from_onnx
from foldquant.runtime.plugins import prepare_plugins
import tyro

from ._upstream import (
    COMPONENTS,
    ENGINES_RECORD_NAME,
    EXPORT_METADATA_NAME,
    MANIFEST_NAME,
    UPSTREAM_DIT_ONNX,
)


logger = logging.getLogger("foldquant.groot_n1_6.build")

#: RoPE table length the FoldQuant LLM graph bakes (``foldquant.export.LLM_BAKE_MAX_SEQ_LEN``).
LLM_BAKED_MAX_SEQ_LEN = 4096


@dataclass
class BuildConfig:
    engine_dir: str
    """Destination engine directory."""

    onnx_dir: Optional[str] = None
    """The FoldQuant export's ``onnx/`` directory (holds foldquant_export.json and export_metadata.json)."""

    float_onnx_dir: Optional[str] = None
    """Directory holding upstream's float ``dit_model.onnx``; used for the DiT when the FoldQuant export has none."""

    metadata: Optional[str] = None
    """``export_metadata.json`` to take the captured shapes from (default: the one in ``--onnx-dir``)."""

    max_batch: int = 1
    """Upper bound of the batch profile. 1 pins it, which is what a robot client and
    the drift protocol need. Upstream's simulation clients vectorise the environment
    (``rollout_policy.py --n_envs``) and send that many observations at once, so an
    engine serving them has to be built with ``--max-batch`` at least as large: a
    batch outside the compiled profile is refused at inference, not adapted to."""

    llm_max_seq_len: Optional[int] = None
    """Upper bound of the LLM sequence profile (default: ``max(2 * captured, captured + 64)``, capped at 4096)."""

    vl_max_seq_len: Optional[int] = None
    """Upper bound of the DiT's vision-language sequence profile (same default rule)."""

    workspace_mb: int = 8192


def _default_max(opt: int) -> int:
    return max(2 * opt, opt + 64)


def load_manifest(onnx_dir: Path) -> Dict[str, Any]:
    return json.loads((onnx_dir / MANIFEST_NAME).read_text())


def dim_ranges(metadata: Dict[str, Any], args: BuildConfig) -> Dict[str, Any]:
    """``{dim_name: int | (min, opt, max)}`` for every symbolic dimension the two graphs carry."""
    llm_opt = int(metadata["llm_seq_len"])
    vl_opt = int(metadata["vl_seq_len"])
    llm_max = args.llm_max_seq_len or min(_default_max(llm_opt), LLM_BAKED_MAX_SEQ_LEN)
    vl_max = args.vl_max_seq_len or _default_max(vl_opt)
    if not llm_opt <= llm_max <= LLM_BAKED_MAX_SEQ_LEN:
        raise ValueError(
            f"--llm-max-seq-len {llm_max} must be >= the captured {llm_opt} and <= {LLM_BAKED_MAX_SEQ_LEN}"
        )
    if vl_max < vl_opt:
        raise ValueError(f"--vl-max-seq-len {vl_max} is below the captured {vl_opt}")
    batch = 1 if args.max_batch <= 1 else (1, 1, int(args.max_batch))
    return {
        "batch": batch,
        "batch_size": batch,
        "seq_len": (1, llm_opt, llm_max),
        "vl_seq_len": (1, vl_opt, vl_max),
        "sa_seq_len": int(metadata["sa_seq_len"]),
    }


def build(args: BuildConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else None
    float_dir = Path(args.float_onnx_dir) if args.float_onnx_dir else None
    if onnx_dir is None and float_dir is None:
        raise SystemExit(
            "give --onnx-dir (a FoldQuant export), --float-onnx-dir (upstream's DiT export), or both"
        )
    engine_dir = Path(args.engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = (
        Path(args.metadata)
        if args.metadata
        else (onnx_dir / EXPORT_METADATA_NAME if onnx_dir else None)
    )
    if metadata_path is None or not metadata_path.is_file():
        raise SystemExit(
            f"captured shapes not found ({metadata_path}); pass --metadata <export_metadata.json>"
        )
    metadata = json.loads(metadata_path.read_text())
    ranges = dim_ranges(metadata, args)
    logger.info("dimension ranges: %s", ranges)

    manifest = (
        load_manifest(onnx_dir) if onnx_dir else {"schemes": {}, "files": {}, "plugin_libs": []}
    )
    plugin_libs: List[str] = list(manifest["plugin_libs"])
    if plugin_libs:
        prepare_plugins(plugin_libs)

    record: Dict[str, Any] = {
        "onnx_dir": public_path(str(onnx_dir) if onnx_dir else None),
        "float_onnx_dir": public_path(str(float_dir) if float_dir else None),
        "metadata": metadata,
        "dim_ranges": {k: v for k, v in ranges.items()},
        "plugin_libs": plugin_libs,
        "schemes": dict(manifest["schemes"]),
        "components": {},
    }
    for name, _onnx_name, engine_name in COMPONENTS:
        if name in manifest["files"]:
            assert onnx_dir is not None
            src = onnx_dir / manifest["files"][name]
            if manifest.get("schemes", {}).get(name) == "float":
                # STRONGLY_TYPED, like every other arm: a weakly-typed network picks a
                # precision per layer, and the layers TensorRT runs in fp32 are *more* exact
                # than the bf16 reference this arm is scored against — which made pi05's float
                # engine drift further from PyTorch than its INT8 engine did. Honouring the
                # ONNX's own dtypes keeps the float arm a control that differs from the
                # quantized arms in the precision of the projections and in nothing else.
                strongly_typed, int8 = True, False
                source = "float"
            else:
                strongly_typed, int8 = True, True
                source = "foldquant"
        elif name == "dit" and float_dir is not None:
            src = float_dir / UPSTREAM_DIT_ONNX
            strongly_typed, int8 = False, False
            source = "float"
        else:
            logger.info("%s: not exported, stays in PyTorch", name)
            continue
        if not src.is_file():
            raise FileNotFoundError(src)
        profiles = profiles_from_onnx(src, ranges)
        t0 = time.time()
        logger.info("%s: building %s from %s (%s)", name, engine_name, src.name, source)
        build_engine(
            src,
            engine_dir / engine_name,
            profiles=profiles,
            plugin_libs=plugin_libs if source == "foldquant" else (),
            strongly_typed=strongly_typed,
            int8=int8,
            workspace_mb=args.workspace_mb,
        )
        record["components"][name] = {
            "onnx": str(src),
            "engine": engine_name,
            "source": source,
            "profiles": {
                k: {"min": v.min, "opt": v.opt, "max": v.max} for k, v in profiles.items()
            },
            "seconds": round(time.time() - t0, 1),
        }
        logger.info("%s: built in %.0fs", name, time.time() - t0)
    if not record["components"]:
        raise SystemExit("nothing to build")

    (engine_dir / ENGINES_RECORD_NAME).write_text(json.dumps(record, indent=2))
    if onnx_dir is not None:
        shutil.copy2(onnx_dir / MANIFEST_NAME, engine_dir / MANIFEST_NAME)
    logger.info("engine directory complete: %s", engine_dir)
    return engine_dir


if __name__ == "__main__":
    build(tyro.cli(BuildConfig))
