# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Upstream's ZMQ inference server, serving the policy with FoldQuant engines installed.

This is ``scripts/inference_service.py --server`` with one extra step: after
the ``Gr00tPolicy`` is assembled, :func:`foldquant_integration.runtime.install_engines`
swaps the LLM and DiT engines in. The wire protocol, endpoints and
observation / action dictionaries are upstream's, so the unmodified LIBERO
client evaluates a FoldQuant arm exactly as it evaluates the bf16 policy::

    # terminal 1 — the arm under test
    python -m foldquant_integration.serve --model-path <ckpt> --embodiment-tag new_embodiment \\
        --engine-dir exports/n15_w4a4/engines --denoising-steps 8

    # terminal 2 — upstream's evaluation client, unchanged
    python examples/Libero/eval/run_libero_eval.py --task_suite_name libero_spatial --headless

Omit ``--engine-dir`` to serve the bf16 PyTorch policy (the reference arm).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import tyro

from . import calibration
from ._upstream import LIBERO_DATA_CONFIG
from .runtime import install_engines

logger = logging.getLogger("foldquant.groot_n1_5.serve")


@dataclass
class ServeConfig:
    model_path: str
    """Checkpoint directory or Hugging Face id."""

    embodiment_tag: Optional[str] = None
    """Embodiment tag (the released LIBERO checkpoints use ``new_embodiment``)."""

    engine_dir: Optional[str] = None
    """FoldQuant engine directory; omit to serve the bf16 PyTorch policy."""

    data_config: str = LIBERO_DATA_CONFIG
    """``module:Class`` data config, as upstream's inference service takes it."""

    denoising_steps: Optional[int] = None
    """Flow-matching steps (upstream serves the LIBERO checkpoints with 8)."""

    port: int = 5555
    api_token: Optional[str] = None


def main(args: ServeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    policy = calibration.load_policy(
        args.model_path,
        args.embodiment_tag,
        "cuda",
        data_config=args.data_config,
        denoising_steps=args.denoising_steps,
    )
    installed = None
    if args.engine_dir:
        installed = install_engines(policy, args.engine_dir)
        logger.info("serving with FoldQuant engines: %s", ", ".join(sorted(installed.engines)))
    else:
        logger.info("serving the bf16 PyTorch policy")
    logger.info(
        "embodiment %s, %d denoising steps, port %d",
        policy.embodiment_tag.value,
        policy.denoising_steps,
        args.port,
    )

    from gr00t.eval.robot import RobotInferenceServer

    try:
        RobotInferenceServer(policy, port=args.port, api_token=args.api_token).run()
    finally:
        if installed is not None:
            installed.remove()


if __name__ == "__main__":
    main(tyro.cli(ServeConfig))
