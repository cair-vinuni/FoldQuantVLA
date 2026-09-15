# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Emit the FoldQuant plugin graphs for a GR00T N1.6 checkpoint.

Writes, under ``--output-dir``::

    onnx/llm_bf16.onnx        FoldQuant LLM graph  (unless --llm-scheme none)
    onnx/dit_bf16.onnx        FoldQuant DiT graph  (unless --dit-scheme none)
    onnx/export_metadata.json captured shapes, for the engine builder
    onnx/foldquant_export.json what was exported, from which samples, needing which plugins

The DiT graph keeps the I/O contract of the ``dit_model.onnx`` upstream's
``export_onnx_n1d6.py`` writes (same input / output names and dtypes, batch
pinned to 1). The LLM graph is FoldQuant's: ``inputs_embeds`` and
``attention_mask`` in, the decoder's ``hidden_states`` out — what the Eagle
wrapper hands the Qwen3 tower and what the backbone reads back as
``hidden_states[-1]``. Whether that last entry is the post-norm stream is
decided by the transformers release upstream pins, not by this code: the
export checks it on a real forward and emits the final RMSNorm accordingly.

Example::

    python -m foldquant_integration.export_foldquant \\
        --model-path nvidia/GR00T-N1.6-LIBERO --embodiment-tag libero_panda \\
        --dataset-path <calibration dataset> \\
        --output-dir exports/n16_w4a4 \\
        --llm-scheme w4a4_srg --dit-scheme w4a4_shg --cascade
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

from foldquant import schemes
from foldquant.export import export_dit, export_llm, install_llm_emulation
from foldquant.float_export import FLOAT, Binding, causal_additive_mask_4d, export_module_float
from foldquant.provenance import public_path
import torch
import tyro

from . import calibration
from ._upstream import EXPORT_METADATA_NAME, MANIFEST_NAME


logger = logging.getLogger("foldquant.groot_n1_6.export")

_NONE = ("", "none")


@dataclass
class ExportConfig:
    model_path: str
    """Checkpoint directory or Hugging Face id (as for the upstream tools)."""

    dataset_path: str
    """LeRobot-format dataset the calibration observations are drawn from."""

    output_dir: str
    """Destination; the graphs land in ``<output_dir>/onnx``."""

    embodiment_tag: Optional[str] = None
    """Embodiment tag; read off the checkpoint's processor_config.json when omitted."""

    llm_scheme: str = schemes.W8A8_SR
    """FoldQuant scheme for the Qwen3 text tower; ``float`` exports the unquantized engine; ``none`` keeps PyTorch."""

    dit_scheme: str = schemes.W4A4_SHG
    """FoldQuant scheme for the action-head DiT; ``float`` exports the unquantized engine; ``none`` keeps PyTorch."""

    num_calib: int = 128
    """Calibration observations (episode, step) pairs spread over the dataset."""

    seed: int = 0
    """Seed of the calibration sample and of the denoising noise replayed during calibration."""

    cascade: bool = False
    """Calibrate the DiT on the activations of the *quantized* LLM (fake-quant emulation)."""

    llm_params: str = "{}"
    """JSON overrides for the LLM fold (sq_alpha, act_clip_ratio, site_bits, rot_block_size)."""

    dit_params: str = "{}"
    """JSON overrides for the DiT fold (sq_alpha, sq_fold_order)."""

    video_backend: str = "torchcodec"
    """Video decoder for the dataset loader."""

    device: str = "cuda"


def _scheme_or_none(value: str) -> Optional[str]:
    return None if value.strip().lower() in _NONE else value.strip()


def _module_paths(policy) -> Dict[str, torch.nn.Module]:
    """The two quantizable modules, at their upstream attribute paths."""
    return {
        "llm": policy.model.backbone.model.language_model,
        "dit": policy.model.action_head.model,
    }


def capture_shape_metadata(policy, observation: Dict[str, Any]) -> Dict[str, Any]:
    """One forward with hooks: the tensor shapes the engine builder profiles, and where the backbone reads.

    ``final_norm`` records whether ``hidden_states[-1]`` of the Qwen3 decoder —
    the entry the Eagle backbone consumes — is the post-final-norm stream
    (``last_hidden_state``) under the installed transformers, so the emitted
    graph ends where the PyTorch tower's output does.
    """
    modules = _module_paths(policy)
    seen: Dict[str, Any] = {}

    def _llm_hook(_m, args, kwargs):
        embeds = args[0] if args else kwargs.get("inputs_embeds")
        seen["llm_seq_len"] = int(embeds.shape[1])
        seen["llm_hidden_size"] = int(embeds.shape[2])
        seen["batch_size"] = int(embeds.shape[0])

    def _decoder_hook(_m, _args, output):
        last = output.hidden_states[-1]
        seen["final_norm"] = bool(torch.equal(last, output.last_hidden_state))

    def _dit_hook(_m, args, kwargs):
        seen["vl_seq_len"] = int(kwargs["encoder_hidden_states"].shape[1])

    handles = [
        modules["llm"].register_forward_pre_hook(_llm_hook, with_kwargs=True),
        modules["llm"].model.register_forward_hook(_decoder_hook),
        modules["dit"].register_forward_pre_hook(_dit_hook, with_kwargs=True),
    ]
    try:
        with torch.inference_mode():
            policy.get_action(observation)
    finally:
        for h in handles:
            h.remove()
    missing = [k for k in ("llm_seq_len", "vl_seq_len", "final_norm") if k not in seen]
    if missing:
        raise RuntimeError(f"shape capture never reached {missing}; is this an N1.6 policy?")
    cfg = policy.model.action_head.config
    seen["sa_seq_len"] = 1 + int(cfg.action_horizon)
    seen["action_horizon"] = int(cfg.action_horizon)
    return seen


def main(args: ExportConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    llm_scheme = _scheme_or_none(args.llm_scheme)
    dit_scheme = _scheme_or_none(args.dit_scheme)
    if llm_scheme is None and dit_scheme is None:
        raise SystemExit("nothing to export: both --llm-scheme and --dit-scheme are none")
    if llm_scheme is not None:
        schemes.validate("llm", llm_scheme)
    if dit_scheme is not None:
        schemes.validate("dit", dit_scheme)
    if args.cascade and (llm_scheme is None or dit_scheme is None):
        raise SystemExit("--cascade needs both an LLM scheme and a DiT scheme")
    if args.cascade and llm_scheme not in schemes.LLM_FOLDED_SCHEMES:
        raise SystemExit(f"--cascade emulates a folded LLM; {llm_scheme!r} folds nothing")
    llm_params = json.loads(args.llm_params)
    dit_params = json.loads(args.dit_params)

    out = Path(args.output_dir) / "onnx"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    policy = calibration.load_policy(args.model_path, args.embodiment_tag, args.device)
    dataset = calibration.load_dataset(policy, args.dataset_path, args.video_backend)
    logger.info("policy + dataset (%d episodes) in %.0fs", len(dataset), time.time() - t0)

    samples, observations = calibration.sample_observations(
        policy, dataset, args.num_calib, seed=args.seed
    )
    loop = calibration.make_forward_loop(policy, observations, seed=args.seed)
    modules = _module_paths(policy)

    shapes = capture_shape_metadata(policy, observations[0])
    logger.info("captured shapes: %s", shapes)

    results = []
    plugin_libs: list = []
    llm_result = None
    if llm_scheme is not None:
        t1 = time.time()
        if llm_scheme == FLOAT:
            llm_result = export_module_float(
                modules["llm"],
                out / "llm_bf16.onnx",
                module_name="llm",
                bindings=[
                    Binding("inputs_embeds", "inputs_embeds", torch.bfloat16, {1: "seq_len"}),
                    Binding("attention_mask", "attention_mask", torch.int64, {1: "seq_len"}, transform=causal_additive_mask_4d),
                ],
                output_name="hidden_states",
                output_dynamic={1: "seq_len"},
                forward_loop=loop,
                extract=lambda o: o.hidden_states[-1],
            )
        else:
            llm_result = export_llm(
                modules["llm"],
                out / "llm_bf16.onnx",
                scheme=llm_scheme,
                forward_loop=loop,
                params=llm_params or None,
                final_norm=shapes["final_norm"],
            )
        results.append(llm_result)
        logger.info("LLM %s exported in %.0fs", llm_scheme, time.time() - t1)

    if dit_scheme is not None:
        t1 = time.time()
        emulation = None
        if args.cascade:
            assert llm_result is not None
            emulation = install_llm_emulation(modules["llm"], llm_result)
            logger.info("cascade: DiT calibration runs under the quantized-LLM emulation")
        try:
            if dit_scheme == FLOAT:
                _ones = lambda kw: torch.ones(kw["encoder_hidden_states"].shape[:2], dtype=torch.bool,
                                              device=kw["encoder_hidden_states"].device)
                dit_result = export_module_float(
                    modules["dit"],
                    out / "dit_bf16.onnx",
                    module_name="dit",
                    bindings=[
                        Binding("sa_embs", "hidden_states", torch.bfloat16),
                        Binding("vl_embs", "encoder_hidden_states", torch.bfloat16, {1: "vl_seq_len"}),
                        Binding("timestep", "timestep", torch.int64),
                        Binding("image_mask", "image_mask", torch.bool, {1: "vl_seq_len"}, default=_ones),
                        Binding("backbone_attention_mask", "backbone_attention_mask", torch.bool, {1: "vl_seq_len"}, default=_ones),
                    ],
                    output_name="output",
                    forward_loop=loop,
                    extract=lambda o: o[0] if isinstance(o, (tuple, list)) else o,
                )
            else:
                dit_result = export_dit(
                    modules["dit"],
                    out / "dit_bf16.onnx",
                    scheme=dit_scheme,
                    forward_loop=loop,
                    params=dit_params or None,
                )
        finally:
            if emulation is not None:
                emulation.remove()
        results.append(dit_result)
        logger.info("DiT %s exported in %.0fs", dit_scheme, time.time() - t1)

    for r in results:
        for lib in r.plugin_libs:
            if lib not in plugin_libs:
                plugin_libs.append(lib)

    metadata = {
        "model_version": "n1d6",
        "sa_seq_len": shapes["sa_seq_len"],
        "vl_seq_len": shapes["vl_seq_len"],
        "llm_seq_len": shapes["llm_seq_len"],
        "llm_hidden_size": shapes["llm_hidden_size"],
        "llm_final_norm": shapes["final_norm"],
        "action_horizon": shapes["action_horizon"],
        "embodiment_tag": policy.embodiment_tag.value,
        "export_mode": "foldquant",
        "precision": "bf16",
        "batch_size": shapes["batch_size"],
    }
    (out / EXPORT_METADATA_NAME).write_text(json.dumps(metadata, indent=2))

    manifest = {
        "model_path": public_path(args.model_path),
        "embodiment_tag": policy.embodiment_tag.value,
        "dataset_path": public_path(args.dataset_path),
        "schemes": {r.module: r.scheme for r in results},
        "params": {"llm": llm_params, "dit": dit_params},
        "cascade": bool(args.cascade),
        "llm_final_norm": shapes["final_norm"],
        "plugin_libs": plugin_libs,
        "files": {r.module: r.onnx_path.name for r in results},
        "calibration": {
            "seed": args.seed,
            "num_samples": len(samples),
            "samples": [asdict(s) for s in samples],
        },
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    logger.info(
        "wrote %s and %s (total %.0fs)", EXPORT_METADATA_NAME, MANIFEST_NAME, time.time() - t0
    )
    return out


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
