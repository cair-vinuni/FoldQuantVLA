# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Upstream's websocket policy server, serving the policy with FoldQuant engines installed.

This is ``scripts/serve_policy.py policy:checkpoint`` with one extra step:
after the PyTorch policy is assembled, :func:`foldquant_integration.runtime.install_engines`
swaps the PaliGemma prefix and action-expert engines in. The wire protocol
and the observation / action dictionaries are upstream's, so the unmodified
LIBERO client (``examples/libero/main.py``, in its own environment) evaluates
a FoldQuant arm exactly as it evaluates the bf16 policy::

    # terminal 1 — the arm under test
    python -m foldquant_integration.serve --checkpoint-dir <ckpt> --engine-dir exports/pi05_w4a4/engines

    # terminal 2 — upstream's evaluation client, unchanged
    python examples/libero/main.py --args.task-suite-name libero_spatial

Omit ``--engine-dir`` to serve the bf16 PyTorch policy (the reference arm).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import tyro

from . import calibration
from ._upstream import LIBERO_TRAIN_CONFIG
from .runtime import install_engines

logger = logging.getLogger("foldquant.pi05.serve")


@dataclass
class ServeConfig:
    checkpoint_dir: str
    """PyTorch checkpoint directory, as ``--policy.dir`` of the upstream server."""

    engine_dir: str | None = None
    """FoldQuant engine directory; omit to serve the bf16 PyTorch policy."""

    config: str = LIBERO_TRAIN_CONFIG
    """Upstream training config name, as ``--policy.config`` of the upstream server."""

    port: int = 8000
    device: str = "cuda"


def main(args: ServeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    # The engines rebind two methods on the instance, which torch.compile would trace past (runtime.py);
    # the reference arm keeps upstream's compiled serving configuration.
    policy = calibration.load_policy(
        args.checkpoint_dir, config_name=args.config, device=args.device, compile=not args.engine_dir
    )
    installed = None
    if args.engine_dir:
        installed = install_engines(policy, args.engine_dir)
        logger.info("serving with FoldQuant engines: %s", ", ".join(sorted(installed.engines)))
    else:
        logger.info("serving the bf16 PyTorch policy")

    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    )
    logger.info("config %s, port %d — ready", args.config, args.port)
    try:
        server.serve_forever()
    finally:
        if installed is not None:
            installed.remove()


if __name__ == "__main__":
    main(tyro.cli(ServeConfig))
