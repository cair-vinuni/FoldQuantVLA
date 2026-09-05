# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Upstream's async policy server, serving SmolVLA with FoldQuant engines installed.

This is ``lerobot.async_inference.policy_server.serve`` with one extra step, and
the step has to land in a different place than it does for the other five
families. Upstream builds the policy **lazily**: ``PolicyServer.__init__`` leaves
``self.policy`` as ``None``, and the checkpoint is only loaded when a client
sends its ``RemotePolicyConfig`` in ``SendPolicyInstructions`` — the client, not
the server, names the checkpoint. So there is no assembled policy to install
engines into before the server starts, and the pattern the other integrations
use (load, install, hand to the server) does not transfer.

:class:`FoldQuantPolicyServer` therefore hooks the handshake instead: it calls
upstream's ``SendPolicyInstructions`` unchanged and installs the engines into
the policy that call just built, before the first observation can arrive. The
gRPC service, the wire protocol and the action contract stay upstream's, so
``lerobot.async_inference.robot_client`` talks to a quantized policy exactly as
it talks to the bf16 one::

    # inference host
    python -m foldquant_integration.serve --engine-dir exports/w8a8/engines --port 8080

    # robot host — upstream's own client, unchanged
    python -m lerobot.async_inference.robot_client \\
        --server_address=<host>:8080 --policy_type=smolvla \\
        --pretrained_name_or_path=<ckpt> ...

Omit ``--engine-dir`` to serve the bf16 PyTorch policy (the reference arm).

Two consequences of the lazy construction are worth stating. The engine
directory must match the checkpoint the client asks for — nothing here can
check that, because the engines carry shapes and the checkpoint carries
weights, and a mismatch surfaces as wrong actions rather than as an error. And
a client that reconnects with a *different* checkpoint rebuilds the policy, so
the engines are installed again against the new one; the install is idempotent
per policy instance, not per server.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import tyro

from .runtime import install_engines

logger = logging.getLogger("foldquant.smolvla.serve")


@dataclass
class ServeConfig:
    engine_dir: str | None = None
    """FoldQuant engine directory; omit to serve the bf16 PyTorch policy."""

    host: str = "0.0.0.0"
    port: int = 8080
    fps: int = 30
    inference_latency: float = 0.033
    obs_queue_timeout: float = 1.0


def main(args: ServeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from concurrent import futures

    import grpc

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.transport import services_pb2_grpc

    class FoldQuantPolicyServer(PolicyServer):
        """Upstream's server, with the engines swapped in after it builds the policy."""

        def __init__(self, cfg, engine_dir: str | None):
            super().__init__(cfg)
            self._engine_dir = engine_dir
            self._installed = None

        def SendPolicyInstructions(self, request, context):  # noqa: N802 - upstream's name
            reply = super().SendPolicyInstructions(request, context)
            if self._engine_dir and self.policy is not None:
                if self._installed is not None:
                    self._installed.remove()  # a reconnect rebuilt the policy
                self._installed = install_engines(self.policy, self._engine_dir)
                logger.info(
                    "engines installed into the client's policy: %s",
                    ", ".join(sorted(self._installed.engines)),
                )
            return reply

    cfg = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    server_impl = FoldQuantPolicyServer(cfg, args.engine_dir)
    if args.engine_dir:
        logger.info("engines will be installed when a client sends its policy: %s", args.engine_dir)
    else:
        logger.info("no engine directory: serving the bf16 PyTorch policy")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_impl, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    logger.info("listening on %s:%d", cfg.host, cfg.port)
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("stopped")
    finally:
        if server_impl._installed is not None:
            server_impl._installed.remove()


if __name__ == "__main__":
    main(tyro.cli(ServeConfig))
