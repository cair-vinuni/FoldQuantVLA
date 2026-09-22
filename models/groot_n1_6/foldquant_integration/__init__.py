# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""FoldQuant on GR00T N1.6: the integration layer over the upstream release.

Everything model-specific lives here; the algorithm, the emitters, the
TensorRT plugins and the engine builder / runtime wrapper are the top-level
``foldquant`` package. The upstream ``gr00t`` package is used unchanged:

* :mod:`.calibration`:  load the upstream policy / dataset and sample
  calibration observations through the upstream data path.
* :mod:`.export_foldquant`:  emit the FoldQuant ``llm_bf16.onnx`` /
  ``dit_bf16.onnx`` plugin graphs.
* :mod:`.build_engines`:  compile them (plugins loaded first) into an engine
  directory, optionally beside a float DiT engine built from upstream's
  ``export_onnx_n1d6.py`` graph.
* :mod:`.runtime`:  install the engines into a live ``Gr00tPolicy`` in place
  of the PyTorch LLM and DiT.
* :mod:`.verify`:  held-out cosine / max-abs of the engine directory against
  the bf16 PyTorch policy.
* :mod:`.eval_libero`:  LIBERO success rate with the upstream rollout loop.
* :mod:`.benchmark`:  upstream's component timing, PyTorch beside FoldQuant.
"""
