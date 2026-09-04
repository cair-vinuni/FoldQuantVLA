# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""FoldQuant on Evo-1 — the integration layer over the upstream Evo-1 release.

Everything model-specific lives here; the algorithm, the emitters, the
TensorRT plugins and the engine builder / runtime wrapper are the top-level
``foldquant`` package. The upstream ``Evo_1`` package is used unchanged:

* :mod:`.calibration` — load the upstream model / normalizer / dataset and
  sample calibration observations in the shape upstream's LIBERO client sends.
* :mod:`.export_foldquant` — emit the FoldQuant ``llm_bf16.onnx`` /
  ``action_head_bf16.onnx`` plugin graphs.
* :mod:`.build_engines` — compile them (plugins loaded first) into an engine
  directory.
* :mod:`.runtime` — install the engines into a live ``EVO1`` in place of the
  InternVL3 language tower and the flow-matching head's denoise step.
* :mod:`.verify` — held-out cosine / max-abs of the engine directory against
  the bf16 PyTorch model.
* :mod:`.serve` — upstream's websocket server over the engines, for upstream's
  unmodified LIBERO client.
* :mod:`.eval_libero` — the same LIBERO rollout driven from one script, with a
  per-suite summary.
* :mod:`.benchmark` — component latency, PyTorch beside FoldQuant.
"""
