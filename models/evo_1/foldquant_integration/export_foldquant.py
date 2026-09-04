# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Emit the FoldQuant plugin graphs for an Evo-1 checkpoint.

Writes, under ``--output-dir``::

    onnx/llm_bf16.onnx          FoldQuant InternVL3 (Qwen2) tower graph (unless --llm-scheme none)
    onnx/action_head_bf16.onnx  FoldQuant denoise-step graph (unless --head-scheme none)
    onnx/export_metadata.json   captured shapes, for the engine builder
    onnx/foldquant_export.json  what was exported, from which samples, needing which plugins

The tower graph is the whole fused-token pass: ``inputs_embeds`` (vision tiles
and prompt embeddings, produced in PyTorch) and the padding ``attention_mask``
in, ``hidden_states`` out. Upstream pads the prompt to a fixed length and sends
a fixed three-image set, so that sequence is a constant of the checkpoint and
the engine is static; the flash-attention treatment of the padded keys is what
the emitter's ``padded_query_mask`` reproduces.

The head graph is one denoise step (``action_seq``, ``context_tokens``,
``time_emb`` -> ``velocity``); upstream's 50-step Euler loop stays in PyTorch.

Example::

    python -m foldquant_integration.export_foldquant \\
        --checkpoint-dir <Evo1_LIBERO checkpoint> \\
        --dataset-path <LeRobot LIBERO dataset> \\
        --output-dir exports/evo1_w8a8_w4a4 \\
        --llm-scheme w8a8_sr --head-scheme w4a4_shg
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tyro

from foldquant import schemes
from foldquant.export import export_action_head, export_llm, install_llm_emulation
from foldquant.provenance import public_path

from . import calibration
from ._upstream import (
    EXPORT_METADATA_NAME,
    LIBERO_ARM_KEY,
    LIBERO_CHECKPOINT,
    LIBERO_DATASET_KEY,
    MANIFEST_NAME,
)
from .runtime import ContextCapture, head_module, llm_module, model_of

logger = logging.getLogger("foldquant.evo_1.export")

_NONE = ("", "none", "float")


@dataclass
class ExportConfig:
    output_dir: str
    """Destination; the graphs land in ``<output_dir>/onnx``."""

    dataset_path: str
    """LeRobot dataset the calibration observations are drawn from (local path or hub id)."""

    checkpoint_dir: str = LIBERO_CHECKPOINT
    """Upstream checkpoint directory (``config.json``, ``norm_stats.json``, ``mp_rank_00_model_states.pt``)."""

    llm_scheme: str = schemes.W8A8_SR
    """FoldQuant scheme for the InternVL3 language tower, or ``none`` to keep it float."""

    head_scheme: str = schemes.W4A4_SHG
    """FoldQuant scheme for the flow-matching action head, or ``none`` to keep it float."""

    num_calib: int = 128
    """Calibration observations (episode, step) pairs spread over the dataset."""

    seed: int = 0
    """Seed of the calibration sample and of the flow-matching start replayed during calibration."""

    episodes: str = ""
    """Optional episodes to restrict the dataset load to, as indices or ranges (``0-149``, ``0,3,7``)."""

    arm_key: str = LIBERO_ARM_KEY
    """Normalizer arm key; empty reads it off the checkpoint's norm_stats.json."""

    dataset_key: str = LIBERO_DATASET_KEY
    """Normalizer dataset key; empty reads it off the checkpoint's norm_stats.json."""

    cascade: bool = False
    """Calibrate the head on the activations of the *quantized* tower (fake-quant emulation)."""

    llm_params: str = "{}"
    """JSON overrides for the tower fold (sq_alpha, act_clip_ratio, site_bits, rot_block_size)."""

    head_params: str = "{}"
    """JSON overrides for the head fold (sq_alpha, sq_fold_order)."""

    device: str = "cuda"


def _scheme_or_none(value: str) -> str | None:
    return None if value.strip().lower() in _NONE else value.strip()


def capture_shape_metadata(deployed, request: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """One inference with the tower seam recording: the shapes the engine builder profiles."""
    model = model_of(deployed)
    head = head_module(deployed)
    seen: dict[str, Any] = {}

    def _step_hook(_m, _args):
        seen["head_calls"] = seen.get("head_calls", 0) + 1

    handle = head.norm_out.register_forward_pre_hook(_step_hook)
    try:
        with ContextCapture(deployed) as capture:
            calibration.infer(deployed, request, seed=seed)
    finally:
        handle.remove()
    if len(capture.hidden) != 1:
        raise RuntimeError(f"expected one tower pass per inference, saw {len(capture.hidden)}")
    batch, seq_len, hidden = capture.hidden[0].shape
    if batch != 1:
        raise RuntimeError(f"unexpected fused-token shape {tuple(capture.hidden[0].shape)}")
    config = model.config
    seen.update(
        {
            "batch_size": int(batch),
            "seq_len": int(seq_len),
            "llm_hidden_size": int(hidden),
            "llm_layers": len(llm_module(deployed).model.layers),
            # The head cross-attends the fused tokens plus the state token.
            "context_len": int(seq_len) + (1 if head.state_encoder is not None else 0),
            "horizon": int(model.horizon),
            "per_action_dim": int(model.per_action_dim),
            "action_dim": int(config.action_dim),
            "num_steps": int(seen.get("head_calls", getattr(config, "num_inference_timesteps", 50))),
        }
    )
    return seen


def main(args: ExportConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    llm_scheme = _scheme_or_none(args.llm_scheme)
    head_scheme = _scheme_or_none(args.head_scheme)
    if llm_scheme is None and head_scheme is None:
        raise SystemExit("nothing to export: both --llm-scheme and --head-scheme are none")
    if llm_scheme is not None:
        schemes.validate("llm", llm_scheme)
    if head_scheme is not None:
        schemes.validate("action_head", head_scheme)
    if args.cascade and (llm_scheme is None or head_scheme is None):
        raise SystemExit("--cascade needs both an LLM scheme and a head scheme")
    if args.cascade and llm_scheme not in schemes.LLM_FOLDED_SCHEMES:
        raise SystemExit(f"--cascade emulates a folded tower; {llm_scheme!r} folds nothing")
    llm_params = json.loads(args.llm_params)
    head_params = json.loads(args.head_params)

    out = Path(args.output_dir) / "onnx"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    deployed = calibration.load_policy(
        args.checkpoint_dir, arm_key=args.arm_key, dataset_key=args.dataset_key, device=args.device
    )
    dataset = calibration.load_dataset(args.dataset_path, episodes=calibration.parse_episodes(args.episodes))
    logger.info("model + dataset in %.0fs", time.time() - t0)

    samples, observations = calibration.sample_observations(dataset, args.num_calib, seed=args.seed)
    loop = calibration.make_forward_loop(deployed, observations, seed=args.seed)
    tower = llm_module(deployed)

    shapes = capture_shape_metadata(deployed, observations[0], seed=args.seed)
    logger.info("captured shapes: %s", shapes)

    results = []
    plugin_libs: list = []
    llm_result = None
    if llm_scheme is not None:
        t1 = time.time()
        llm_result = export_llm(
            tower, out / "llm_bf16.onnx", scheme=llm_scheme, forward_loop=loop, params=llm_params or None
        )
        results.append(llm_result)
        logger.info("tower %s exported in %.0fs", llm_scheme, time.time() - t1)

    if head_scheme is not None:
        t1 = time.time()
        emulation = None
        if args.cascade:
            assert llm_result is not None
            emulation = install_llm_emulation(tower, llm_result)
            logger.info("cascade: head calibration runs under the quantized-tower emulation")
        try:
            head_result = export_action_head(
                head_module(deployed),
                out / "action_head_bf16.onnx",
                scheme=head_scheme,
                forward_loop=loop,
                params=head_params or None,
            )
        finally:
            if emulation is not None:
                emulation.remove()
        results.append(head_result)
        logger.info("action head %s exported in %.0fs", head_scheme, time.time() - t1)

    for r in results:
        for lib in r.plugin_libs:
            if lib not in plugin_libs:
                plugin_libs.append(lib)

    metadata = {
        "model": "evo_1",
        "checkpoint_dir": public_path(args.checkpoint_dir),
        "seq_len": shapes["seq_len"],
        "context_len": shapes["context_len"],
        "llm_hidden_size": shapes["llm_hidden_size"],
        "llm_layers": shapes["llm_layers"],
        "horizon": shapes["horizon"],
        "per_action_dim": shapes["per_action_dim"],
        "action_dim": shapes["action_dim"],
        "num_steps": shapes["num_steps"],
        "export_mode": "foldquant",
        "precision": "bf16",
        "batch_size": shapes["batch_size"],
    }
    (out / EXPORT_METADATA_NAME).write_text(json.dumps(metadata, indent=2))
    manifest = {
        "checkpoint_dir": public_path(args.checkpoint_dir),
        "dataset_path": public_path(args.dataset_path),
        "episodes": args.episodes,
        "arm_key": args.arm_key,
        "dataset_key": args.dataset_key,
        "schemes": {r.module: r.scheme for r in results},
        "params": {"llm": llm_params, "action_head": head_params},
        "cascade": bool(args.cascade),
        "plugin_libs": plugin_libs,
        "files": {r.module: r.onnx_path.name for r in results},
        "calibration": {
            "seed": args.seed,
            "num_samples": len(samples),
            "samples": [asdict(s) for s in samples],
        },
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s and %s (total %.0fs)", EXPORT_METADATA_NAME, MANIFEST_NAME, time.time() - t0)
    return out


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
