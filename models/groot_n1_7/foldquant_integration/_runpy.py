# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Run an upstream script as ``__main__`` with the FoldQuant plugin library preloaded."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
from typing import List, Optional

from foldquant.runtime.plugins import load_plugins

from ._upstream import MANIFEST_NAME


def _engine_dir_from_argv(argv: List[str], flags: tuple) -> Optional[str]:
    for i, tok in enumerate(argv):
        for flag in flags:
            if tok == flag and i + 1 < len(argv):
                return argv[i + 1]
            if tok.startswith(flag + "="):
                return tok[len(flag) + 1 :]
    return None


def run_upstream(
    module_or_path: str, argv: List[str], engine_flags: tuple, *, as_path: bool = False
) -> None:
    """Preload plugins named in ``<engine_dir>/foldquant_export.json`` and hand off.

    ``argv`` is forwarded verbatim; the engine directory is only read to find
    the manifest. An engine directory without a manifest (a float upstream
    build) loads nothing.
    """
    engine_dir = _engine_dir_from_argv(argv, engine_flags)
    if engine_dir:
        manifest_path = Path(engine_dir) / MANIFEST_NAME
        if manifest_path.is_file():
            load_plugins(json.loads(manifest_path.read_text())["plugin_libs"])
    sys.argv = [module_or_path, *argv]
    if as_path:
        runpy.run_path(module_or_path, run_name="__main__")
    else:
        runpy.run_module(module_or_path, run_name="__main__", alter_sys=True)
