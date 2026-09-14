# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Locate the upstream Isaac GR00T tree this integration sits in."""

from __future__ import annotations

import os
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


#: The LIBERO checkout the release pins as a submodule. Its ``libero/`` package
#: has no ``__init__.py``, so an editable install leaves nothing importable and
#: ``import libero`` fails even after ``pip install -e`` reports success; the
#: directory has to be on ``sys.path`` instead.
LIBERO_DIR = UPSTREAM_ROOT / "external_dependencies" / "LIBERO"


def _seed_libero_config(benchmark_root: Path) -> None:
    """Write LIBERO's default config if none exists yet (idempotent).

    ``libero.libero`` asks on stdin, at import time, whether to use a custom
    dataset folder whenever ``$LIBERO_CONFIG_PATH/config.yaml`` (default
    ``~/.libero``) is missing. Under a redirected or closed stdin -- a script,
    a log file, a CI job -- that is an ``EOFError`` on a fresh machine, or a
    prompt nobody sees while the run appears to hang. Answer it the way
    upstream's ``setup_libero.sh`` does: the default paths of this checkout.
    An existing config is left untouched.
    """
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return
    import yaml

    root = str(benchmark_root)
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(
            {
                "benchmark_root": root,
                "bddl_files": os.path.join(root, "./bddl_files"),
                "init_states": os.path.join(root, "./init_files"),
                "datasets": os.path.join(root, "../datasets"),
                "assets": os.path.join(root, "./assets"),
            },
            f,
        )


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
    _seed_libero_config(LIBERO_DIR / "libero" / "libero")
