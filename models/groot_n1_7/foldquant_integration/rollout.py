# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Upstream ``gr00t.eval.rollout_policy`` with the FoldQuant plugins preloaded.

Every argument is upstream's; see ``python -m gr00t.eval.rollout_policy --help``.
When ``--trt-engine-path`` names a directory built by :mod:`.build_engines`,
its plugin library is loaded before the engines are deserialised::

    python -m foldquant_integration.rollout --model-path ... \\
        --trt-engine-path exports/n17_w4a4/engines --n-envs 1 \\
        --env-name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
"""

from __future__ import annotations

import sys

from ._runpy import run_upstream


if __name__ == "__main__":
    run_upstream(
        "gr00t.eval.rollout_policy", sys.argv[1:], ("--trt-engine-path", "--trt_engine_path")
    )
