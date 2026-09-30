# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Export a GR00T N1.7 checkpoint to the ONNX graphs the engine builder compiles.

Reads ``--model-path`` and writes, into ``--output-dir``::

    llm_bf16.onnx / dit_bf16.onnx   the two towers
    vit_fp32.onnx, vl_self_attention.onnx, state_encoder.onnx,
    action_encoder.onnx, action_decoder.onnx   the five float components
    export_metadata.json            shape hints for the engine builder
    foldquant_export.json           what was exported, needing which plugins

A **quantized model** (what ``quantize`` wrote) needs no dataset: the recorded
weight codes are replayed into the FoldQuant plugin graphs, byte-identical to
the ones the quantization built, and the float components are traced from the
shapes it recorded. An **unquantized checkpoint** is the all-float baseline:
every graph is traced by upstream's own ``export_onnx_n1d7`` exporters on one
observation from ``--dataset-path`` (the same graphs
``scripts/deployment/build_trt_pipeline.py --steps export`` writes).
``build_engines --onnx-dir <output-dir>`` then builds the engines either way.

Example::

    python -m foldquant_integration.export --model-path exports/w4a4/quantized --output-dir exports/w4a4/onnx
    python -m foldquant_integration.export --model-path <checkpoint> --dataset-path <dataset> \\
        --embodiment-tag <tag> --output-dir exports/float/onnx
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

from foldquant.provenance import public_path
from foldquant.quantized import ToOnnx, to_onnx
from foldquant.quantized_checkpoint import is_quantized_checkpoint
import tyro

from ._upstream import EXPORT_METADATA_NAME, MANIFEST_NAME, ensure_deployment_on_path
from .quantized import FAMILY

logger = logging.getLogger("foldquant.groot_n1_7.export")


@dataclass
class ExportConfig:
    model_path: str
    """A quantized model (``quantize``'s ``--output-dir``) or an unquantized checkpoint (the all-float baseline)."""

    output_dir: str
    """Destination directory of the graphs and manifests."""

    dataset_path: Optional[str] = None
    """LeRobot dataset; one observation traces the graphs of an unquantized checkpoint (unused for a quantized model)."""

    embodiment_tag: Optional[str] = None
    """Defaults to the tag the model was quantized with, or the checkpoint's single embodiment."""

    seed: int = 0
    """Seed of the traced observation of an unquantized checkpoint."""

    video_backend: str = "torchcodec"

    device: str = "cuda"


def export_float(args: ExportConfig) -> Path:
    """The all-float baseline: upstream's seven graphs, traced on one observation."""
    import torch

    from . import calibration
    from .quantize import capture_shape_metadata, export_metadata
    from .quantized import export_float_components

    if not args.dataset_path:
        raise SystemExit(f"{args.model_path} is not a quantized model; tracing its graphs needs --dataset-path")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    policy = calibration.load_policy(args.model_path, args.embodiment_tag, args.device)
    dataset = calibration.load_dataset(policy, args.dataset_path, args.video_backend)
    samples, observations = calibration.sample_observations(policy, dataset, 1, seed=args.seed)
    logger.info("policy + dataset in %.0fs; tracing on %s", time.time() - t0, samples[0])
    shapes = capture_shape_metadata(policy, observations[0])

    ensure_deployment_on_path()
    import export_onnx_n1d7 as upstream

    llm_capture, dit_capture = upstream.LLMInputCapture(), upstream.DiTInputCapture()
    handles = [
        policy.model.backbone.model.model.language_model.register_forward_pre_hook(llm_capture.hook_fn, with_kwargs=True),
        policy.model.action_head.model.register_forward_pre_hook(dit_capture.hook_fn, with_kwargs=True),
    ]
    try:
        with torch.inference_mode():
            policy.get_action(observations[0])
    finally:
        for h in handles:
            h.remove()
    if not (llm_capture.captured and dit_capture.captured):
        raise RuntimeError("the forward never reached the LLM / DiT; is this an N1.7 policy?")
    bs = int(shapes["batch_size"])
    t1 = time.time()
    upstream.export_llm_to_onnx(policy, llm_capture, str(out), use_bf16=True, batch_size=bs)
    upstream.export_dit_to_onnx(policy, dit_capture, str(out / "dit_bf16.onnx"), use_bf16=True, batch_size=bs)
    export_float_components(policy, out, shapes)
    logger.info("seven float graphs traced in %.0fs", time.time() - t1)

    (out / EXPORT_METADATA_NAME).write_text(json.dumps(export_metadata(shapes, policy), indent=2))
    manifest: Dict[str, Any] = {
        "family": FAMILY,
        "model_path": public_path(args.model_path),
        "embodiment_tag": str(policy.embodiment_tag),
        "dataset_path": public_path(args.dataset_path),
        "schemes": {},
        "params": {},
        "cascade": False,
        "plugin_libs": [],
        "files": {"llm": "llm_bf16.onnx", "dit": "dit_bf16.onnx"},
        "float": True,
        "calibration": {"seed": args.seed, "num_samples": 1, "samples": [asdict(s) for s in samples]},
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s (all-float baseline, total %.0fs)", out, time.time() - t0)
    return out


def main(args: ExportConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if is_quantized_checkpoint(args.model_path):
        return to_onnx(ToOnnx(quantized_model=args.model_path, output_dir=args.output_dir,
                              embodiment_tag=args.embodiment_tag, device=args.device))
    return export_float(args)


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
