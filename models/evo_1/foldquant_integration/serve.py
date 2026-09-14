# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Upstream's Evo-1 websocket server, with FoldQuant engines installed.

``Evo1_server.py`` builds its server inside ``if __name__ == "__main__"`` with
the checkpoint, port and normalizer keys written in the file, so this module
reproduces those five lines around an already-loaded model rather than editing
upstream: the request handler, the JSON contract, the normalizer and the
inference path are all upstream's own, imported and called unchanged. Upstream's
LIBERO client therefore talks to a quantized policy without knowing it.

Example::

    python -m foldquant_integration.serve --checkpoint-dir <ckpt> \\
        --engine-dir exports/evo1_w4a4/engines --port 9000
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import tyro

from . import calibration
from ._upstream import LIBERO_ARM_KEY, LIBERO_CHECKPOINT, LIBERO_DATASET_KEY
from .runtime import install_engines

logger = logging.getLogger("foldquant.evo_1.serve")


@dataclass
class ServeConfig:
    checkpoint_dir: str = LIBERO_CHECKPOINT
    engine_dir: str | None = None
    """FoldQuant engine directory; omitted serves the bf16 PyTorch model."""

    port: int = 9000
    arm_key: str = LIBERO_ARM_KEY
    dataset_key: str = LIBERO_DATASET_KEY
    device: str = "cuda"


def main(args: ServeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    import websockets

    calibration._upstream_on_path()  # noqa: SLF001 - the same path entry upstream's own modules add
    from scripts.Evo1_server import handle_request

    deployed = calibration.load_policy(
        args.checkpoint_dir, arm_key=args.arm_key, dataset_key=args.dataset_key, device=args.device
    )
    installed = None
    if args.engine_dir:
        installed = install_engines(deployed, args.engine_dir)
        logger.info("engines installed: %s", ", ".join(sorted(installed.engines)))
    else:
        logger.info("no engine directory: serving the bf16 PyTorch model")

    async def _serve() -> None:
        logger.info("EVO_1 server running at ws://0.0.0.0:%d", args.port)
        async with websockets.serve(
            # The resolved keys, not the CLI's: --arm-key defaults to "" and load_policy
            # reads the real one off norm_stats.json. Passing the empty string made every
            # request fail with "Arm key '' not found in normalization stats".
            lambda ws: handle_request(ws, deployed.model, deployed.normalizer, deployed.arm_key, deployed.dataset_key),
            "0.0.0.0",
            args.port,
            max_size=100_000_000,
            ping_interval=None,
            ping_timeout=None,
        ):
            await asyncio.Future()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        logger.info("stopped")
    finally:
        if installed is not None:
            installed.remove()


if __name__ == "__main__":
    main(tyro.cli(ServeConfig))
