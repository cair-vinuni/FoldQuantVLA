# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Locate the upstream openpi tree this integration sits in."""

from __future__ import annotations

from pathlib import Path

#: ``models/pi05``, the trimmed upstream openpi checkout.
UPSTREAM_ROOT = Path(__file__).resolve().parents[1]

#: A Python file to import before any training config is resolved, named by this
#: environment variable. A checkpoint fine-tuned outside the release carries a
#: ``TrainConfig`` the release has never heard of, and upstream resolves configs
#: from a module-level dict of its own entries. Rather than edit that file (the
#: upstream tree here is used unchanged), the plugin registers the entry itself::
#:
#:     # so101_plugin.py
#:     from openpi.training import config as _config
#:     _config._CONFIGS_DICT["pi05_so101"] = TrainConfig(name="pi05_so101", ...)
#:
#: and is then named by ``FOLDQUANT_PI05_PLUGIN=so101_plugin.py``. Its directory
#: goes on ``sys.path`` first, so modules it imports resolve beside it.
PLUGIN_ENV = "FOLDQUANT_PI05_PLUGIN"


def load_plugin() -> str | None:
    """Import the file named by ``FOLDQUANT_PI05_PLUGIN`` (idempotent).

    Returns the path imported, or ``None`` when the variable is unset. Import
    errors are not swallowed: a plugin that fails to load would otherwise show
    up as "config not found", pointing at the wrong thing.
    """
    import importlib.util
    import os
    import sys

    named = os.environ.get(PLUGIN_ENV)
    if not named:
        return None
    path = Path(named).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{PLUGIN_ENV}={path} is not a file")
    if str(path) in _PLUGINS_LOADED:
        return str(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"foldquant_pi05_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _PLUGINS_LOADED.add(str(path))
    return str(path)


_PLUGINS_LOADED: set[str] = set()

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
