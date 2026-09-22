# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""FoldQuant on GR00T N1.7: the integration layer over the upstream release.

Everything model-specific lives here; the algorithm, the emitters and the
TensorRT plugins are the top-level ``foldquant`` package. The upstream
``gr00t`` package and its ``scripts/deployment`` tools are used unchanged:

* :mod:`.calibration`:  load the upstream policy / dataset and sample
  calibration observations through the upstream data path.
* :mod:`.export_foldquant`:  emit the FoldQuant ``llm_bf16.onnx`` /
  ``dit_bf16.onnx`` plugin graphs in the upstream drop-in I/O contract.
* :mod:`.build_engines`:  compile them with the upstream engine builder
  (plugins loaded first) and complete the engine directory with the float
  components the upstream export produced.
* :mod:`.verify`:  held-out cosine / max-abs of the engine directory against
  the bf16 PyTorch policy, in the upstream ``verify_n1d7_trt.py`` protocol.
* :mod:`.eval_libero`:  LIBERO success rate with the upstream rollout loop.
* :mod:`.rollout` / :mod:`.benchmark`:  the upstream rollout and latency
  tools with the plugins loaded in-process.
"""
