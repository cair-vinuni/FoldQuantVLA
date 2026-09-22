# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Upstream's ZMQ policy server, serving the policy with FoldQuant engines installed.

This is what ``gr00t/eval/run_gr00t_server.py`` does, with one extra step:
after the ``Gr00tPolicy`` is assembled,
:func:`foldquant_integration.runtime.install_engines` swaps the LLM and DiT
engines in. The wire protocol, endpoints and observation / action dictionaries
are upstream's ``PolicyServer``, so an unmodified client talks to a FoldQuant
arm exactly as it talks to the bf16 policy::

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
Pass ``--use-sim-policy-wrapper`` for upstream's simulation clients
(``gr00t/eval/rollout_policy.py`` against SimplerEnv or RoboCasa), which send
flat ``video.*`` / ``state.*`` observations.

The N1.5 release exposes this as ``scripts/inference_service.py`` around a
``RobotInferenceServer``; this release renamed both, so the entry point here is
``PolicyServer`` and the few lines around it are reproduced rather than
imported: upstream's ``main`` builds the policy and starts the server in one
call, with nowhere to install engines in between.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import tyro

from . import calibration
from .runtime import install_engines


logger = logging.getLogger("foldquant.groot_n1_6.serve")


@dataclass
class ServeConfig:
    model_path: str
    """Checkpoint directory or Hugging Face id."""

    embodiment_tag: Optional[str] = None
    """Embodiment tag (the released LIBERO checkpoint uses ``libero_panda``)."""

    engine_dir: Optional[str] = None
    """FoldQuant engine directory; omit to serve the bf16 PyTorch policy."""

    host: str = "0.0.0.0"
    port: int = 5555
    device: str = "cuda"

    use_sim_policy_wrapper: bool = False
    """Wrap the policy in upstream's ``Gr00tSimPolicyWrapper`` (flat ``video.*`` /
    ``state.*`` observations), as ``run_gr00t_server.py --use-sim-policy-wrapper``
    does for the SimplerEnv and RoboCasa clients."""


def main(args: ServeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    policy = calibration.load_policy(args.model_path, args.embodiment_tag, args.device)

    installed = None
    if args.engine_dir:
        installed = install_engines(policy, args.engine_dir)
        logger.info("serving with FoldQuant engines: %s", ", ".join(sorted(installed.engines)))
    else:
        logger.info("serving the bf16 PyTorch policy")
    logger.info(
        "embodiment %s, listening on %s:%d", policy.embodiment_tag.value, args.host, args.port
    )

    from gr00t.policy.server_client import PolicyServer

    served = policy
    if args.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        served = Gr00tSimPolicyWrapper(policy)
        logger.info("sim policy wrapper on")

    server = PolicyServer(policy=served, host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("stopped")
    finally:
        if installed is not None:
            installed.remove()


if __name__ == "__main__":
    main(tyro.cli(ServeConfig))
