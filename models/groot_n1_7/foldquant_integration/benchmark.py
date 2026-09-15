# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Upstream ``scripts/deployment/benchmark_inference.py`` with the FoldQuant plugins preloaded.

Every argument is upstream's. The latency numbers in the paper come from this
script — the same timing loop, warm-up and iteration count for the float and
the FoldQuant arms::

    python -m foldquant_integration.benchmark --model-path ... \\
        --trt-engine-path exports/n17_w4a4/engines --trt-mode n17_full_pipeline
"""

from __future__ import annotations

import sys

from ._runpy import run_upstream
from ._upstream import DEPLOYMENT_DIR, ensure_deployment_on_path


if __name__ == "__main__":
    # `python scripts/deployment/benchmark_inference.py` puts its own directory on
    # sys.path (it imports trt_model_forward as a top-level module); runpy does not.
    ensure_deployment_on_path()
    run_upstream(
        str(DEPLOYMENT_DIR / "benchmark_inference.py"),
        sys.argv[1:],
        ("--trt-engine-path", "--trt_engine_path"),
        as_path=True,
    )
