# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""FoldQuant on SmolVLA — the integration layer over the upstream LeRobot release.

Everything model-specific lives here; the algorithm, the emitters, the
TensorRT plugins and the engine builder / runtime wrapper are the top-level
``foldquant`` package. The upstream ``lerobot`` package is used unchanged:

* :mod:`.calibration` — load the upstream policy / dataset and sample
  calibration observations through upstream's own processor pipeline.
* :mod:`.export_foldquant` — emit the FoldQuant ``llm_bf16.onnx`` /
  ``expert_bf16.onnx`` plugin graphs.
* :mod:`.build_engines` — compile them (plugins loaded first) into an engine
  directory.
* :mod:`.runtime` — install the engines into a live ``VLAFlowMatching`` in
  place of the SmolVLM prefix pass and the action-expert denoise step.
* :mod:`.verify` — held-out cosine / max-abs of the engine directory against
  the bf16 PyTorch policy.
* :mod:`.eval_libero` — upstream's LIBERO rollout loop over the engines, with
  a per-suite summary.
* :mod:`.benchmark` — component latency, PyTorch beside FoldQuant.
"""
