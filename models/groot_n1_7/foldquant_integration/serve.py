# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Upstream's ZMQ policy server, serving the policy with FoldQuant engines installed.

This is what ``gr00t/eval/run_gr00t_server.py`` does, with one extra step:
after the ``Gr00tPolicy`` is assembled, the engine directory is installed into
it and the server then answers from the quantized pipeline. The wire protocol,
endpoints and observation / action dictionaries are upstream's ``PolicyServer``,
so an unmodified client talks to a FoldQuant arm exactly as it talks to the
bf16 policy::

    # terminal 1:  the arm under test
    python -m foldquant_integration.serve --model-path <ckpt> \\
        --embodiment-tag libero_panda --engine-dir exports/w4a4/engines

    # terminal 2:  upstream's own client, unchanged
    from gr00t.policy.server_client import PolicyClient
    client = PolicyClient(host="127.0.0.1", port=5555)
    action = client.get_action(observation)

The release ships no standalone client script; ``PolicyClient`` is what its
real-robot evaluators construct (e.g. ``gr00t/eval/real_robot/SO100``), so a
robot loop points at this server by changing a host and a port and nothing
else.

Omit ``--engine-dir`` to serve the bf16 PyTorch policy (the reference arm).
``--baseline-pack`` serves one of the emulated W4A4 comparison arms built by
:mod:`.baseline_w4a4` (HoloQ-style or DuQuant-style) instead of an engine.

Unlike the other families, the engines are installed by **upstream's own**
``trt_model_forward.setup_tensorrt_engines``, not by a ``runtime.install_engines``
of ours: this integration has no runtime module because the N1.7 release ships
the whole seven-component pipeline swap itself, and :mod:`.verify` and
:mod:`.benchmark` already go through it. Reproducing that here keeps one
install path for the family, since a second one could drift from it silently. The
FoldQuant plugin libraries the manifest names are loaded first, exactly as
``verify`` loads them, or the engines refuse to deserialise.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from foldquant.runtime.plugins import load_plugins
import tyro

from . import calibration
from ._upstream import MANIFEST_NAME, PIPELINE_COMPONENTS, ensure_deployment_on_path


logger = logging.getLogger("foldquant.groot_n1_7.serve")


@dataclass
class ServeConfig:
    model_path: str
    """Checkpoint directory or Hugging Face id."""

    embodiment_tag: Optional[str] = None
    """Embodiment tag (resolved from the checkpoint when omitted)."""

    engine_dir: Optional[str] = None
    """FoldQuant engine directory; omit to serve the bf16 PyTorch policy."""

    baseline_pack: Optional[str] = None
    """Emulated W4A4 baseline pack (:mod:`.baseline_w4a4`); mutually exclusive with ``--engine-dir``."""

    mode: str = "n17_full_pipeline"
    """``trt_model_forward.setup_tensorrt_engines`` mode, as :mod:`.verify` takes it."""

    host: str = "0.0.0.0"
    port: int = 5555
    device: str = "cuda"


def _load_manifest(engine_dir: Path) -> Optional[Dict[str, Any]]:
    """The FoldQuant export manifest beside the engines, when the arm has one.

    An all-float engine directory built from upstream's own export has none,
    and needs no plugin library either.
    """
    path = engine_dir / MANIFEST_NAME
    return json.loads(path.read_text()) if path.is_file() else None



#: Engine files each ``--mode`` of ``trt_model_forward.setup_tensorrt_engines``
#: needs on disk. A mode absent from this table is not checked.
_MODE_ENGINES = {
    "n17_full_pipeline": tuple(engine for _, _, engine in PIPELINE_COMPONENTS),
    "vit_llm_only": ("vit_bf16.engine", "llm_bf16.engine"),
    "dit_only": ("dit_bf16.engine",),
}

def main(args: ServeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    policy = calibration.load_policy(args.model_path, args.embodiment_tag, args.device)

    if args.engine_dir and args.baseline_pack:
        raise ValueError("--engine-dir and --baseline-pack are mutually exclusive")
    if args.baseline_pack:
        from .baselines import apply_pack, install_dit_step_context

        device = next(policy.model.parameters()).device
        summary = apply_pack(policy.model, args.baseline_pack, backend="fake")
        policy.model.to(device=device)
        install_dit_step_context(policy.model)
        manifest = policy.model.baseline_manifest
        logger.info(
            "serving the EMULATED %s W4A4 baseline (%d LLM + %d DiT linears; rotation %s, "
            "LLM activations %s, DiT activations %s) from %s; no INT4 kernel, not a latency arm",
            manifest.get("method"),
            summary.llm_linears,
            summary.dit_linears,
            manifest.get("rotation_mode"),
            manifest.get("llm_activation_granularity"),
            manifest.get("dit_activation_granularity"),
            args.baseline_pack,
        )
    elif args.engine_dir:
        engine_dir = Path(args.engine_dir)
        manifest = _load_manifest(engine_dir)
        if manifest is not None:
            load_plugins(manifest["plugin_libs"])
        ensure_deployment_on_path()
        from trt_model_forward import setup_tensorrt_engines

        # setup_tensorrt_engines keeps a module in PyTorch when its .engine is
        # absent, announcing it with a print and nothing else. Serving would then
        # log the scheme from the manifest while a robot talks to a policy that is
        # partly, or entirely, bf16 PyTorch. Check the files the mode needs before
        # the server binds, and say which engines are actually in use. This is what
        # runtime.install_engines does for N1.5 and N1.6.
        expected = _MODE_ENGINES.get(args.mode)
        if expected is not None:
            absent = [e for e in expected if not (engine_dir / e).is_file()]
            if absent:
                raise FileNotFoundError(
                    f"{engine_dir} has no {', '.join(absent)}; mode {args.mode!r} needs "
                    f"{', '.join(expected)}. Serving would silently fall back to PyTorch "
                    f"for the missing module(s) while still reporting the export's scheme."
                )
        setup_tensorrt_engines(policy, str(engine_dir), mode=args.mode)
        present = sorted(p.name for p in engine_dir.glob("*.engine"))
        logger.info(
            "serving %s in mode %s with engines: %s", engine_dir, args.mode, ", ".join(present)
        )
    else:
        logger.info("serving the bf16 PyTorch policy")
    logger.info(
        "embodiment %s, listening on %s:%d", policy.embodiment_tag.value, args.host, args.port
    )

    from gr00t.policy.server_client import PolicyServer

    server = PolicyServer(policy=policy, host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("stopped")


if __name__ == "__main__":
    main(tyro.cli(ServeConfig))
