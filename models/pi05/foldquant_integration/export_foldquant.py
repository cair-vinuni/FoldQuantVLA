# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Emit the FoldQuant plugin graphs for a Pi0 / Pi0.5 PyTorch checkpoint.

Writes, under ``--output-dir``::

    onnx/llm_bf16.onnx         FoldQuant PaliGemma prefix graph (unless --llm-scheme none)
    onnx/expert_bf16.onnx      FoldQuant action-expert denoise-step graph (unless --expert-scheme none)
    onnx/export_metadata.json  captured shapes, for the engine builder
    onnx/foldquant_export.json what was exported, from which samples, needing which plugins

The LLM graph is the prefix pass only: ``prefix_embs`` (SigLIP features +
prompt embeddings, produced in PyTorch), the 4-D additive attention mask and
``position_ids`` in, the stacked post-RoPE KV cache out. Its sequence length
is pinned from a calibration call — the Pi processor pads the prompt to a
fixed token count and the camera set is fixed, so the prefix is a constant
of the (config, checkpoint) pair. The expert graph is one denoise step
(``x_t``, ``timestep``, ``prefix_pad_masks``, ``kv_stack`` -> ``velocity``);
the Euler loop stays in PyTorch.

Example::

    python -m foldquant_integration.export_foldquant \\
        --checkpoint-dir <pi05_libero PyTorch checkpoint> \\
        --dataset-path <LeRobot LIBERO dataset> \\
        --output-dir exports/pi05_w4a4 \\
        --llm-scheme w8a8_sr --expert-scheme w4a4_shg
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

from foldquant import schemes
from foldquant.export import export_expert
from foldquant.export import export_llm
from foldquant.export import install_llm_emulation
from foldquant.provenance import public_path
import tyro

from . import calibration
from ._upstream import EXPORT_METADATA_NAME
from ._upstream import LIBERO_TRAIN_CONFIG
from ._upstream import MANIFEST_NAME
from .runtime import PrefixCapture
from .runtime import expert_view
from .runtime import llm_module
from .runtime import model_of

logger = logging.getLogger("foldquant.pi05.export")

_NONE = ("", "none", "float")


@dataclass
class ExportConfig:
    checkpoint_dir: str
    """PyTorch checkpoint directory (``model.safetensors`` + ``assets/``), as for ``scripts/serve_policy.py``."""

    dataset_path: str
    """LeRobot-format dataset the calibration observations are drawn from."""

    output_dir: str
    """Destination; the graphs land in ``<output_dir>/onnx``."""

    config: str = LIBERO_TRAIN_CONFIG
    """Upstream training config name (model variant, transforms, norm-stats asset)."""

    llm_scheme: str = schemes.W8A8_SR
    """FoldQuant scheme for the PaliGemma language model, or ``none`` to keep it float."""

    expert_scheme: str = schemes.W4A4_SHG
    """FoldQuant scheme for the Gemma-300M action expert, or ``none`` to keep it float."""

    num_calib: int = 128
    """Calibration observations (episode, step) pairs spread over the dataset."""

    seed: int = 0
    """Seed of the calibration sample and of the flow-matching noise replayed during calibration."""

    cascade: bool = False
    """Calibrate the expert on the activations of the *quantized* LLM (fake-quant emulation)."""

    llm_params: str = "{}"
    """JSON overrides for the LLM fold (sq_alpha, act_clip_ratio, site_bits, rot_block_size)."""

    expert_params: str = "{}"
    """JSON overrides for the expert fold (sq_alpha, sq_fold_order)."""

    device: str = "cuda"


def _scheme_or_none(value: str) -> str | None:
    return None if value.strip().lower() in _NONE else value.strip()


def capture_shape_metadata(policy, observation: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """One ``infer`` with the prefix seam recording: the shapes the engine builder profiles."""
    model = model_of(policy)
    seen: dict[str, Any] = {}

    def _expert_hook(_m, args):
        seen["expert_calls"] = seen.get("expert_calls", 0) + 1

    handle = model.action_in_proj.register_forward_pre_hook(_expert_hook)
    try:
        with PrefixCapture(policy) as capture:
            calibration.infer(policy, observation, seed=seed)
    finally:
        handle.remove()
    if len(capture.stacks) != 1:
        raise RuntimeError(f"expected one prefix pass per infer, saw {len(capture.stacks)}; is this a PI0Pytorch?")
    layers, two, batch, kv_heads, prefix_len, head_dim = capture.stacks[0].shape
    if two != 2 or batch != 1:
        raise RuntimeError(f"unexpected KV stack shape {tuple(capture.stacks[0].shape)}")
    cfg = model.config
    seen.update(
        {
            "batch_size": int(batch),
            "prefix_len": int(prefix_len),
            "llm_layers": int(layers),
            "llm_kv_heads": int(kv_heads),
            "llm_head_dim": int(head_dim),
            "llm_hidden_size": int(llm_module(policy).config.hidden_size),
            "action_horizon": int(cfg.action_horizon),
            "action_dim": int(cfg.action_dim),
            "use_adarms": bool(model.pi05),
            "num_steps": int(seen["expert_calls"]),
        }
    )
    return seen


def main(args: ExportConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    llm_scheme = _scheme_or_none(args.llm_scheme)
    expert_scheme = _scheme_or_none(args.expert_scheme)
    if llm_scheme is None and expert_scheme is None:
        raise SystemExit("nothing to export: both --llm-scheme and --expert-scheme are none")
    if llm_scheme is not None:
        schemes.validate("llm", llm_scheme)
    if expert_scheme is not None:
        schemes.validate("expert", expert_scheme)
    if args.cascade and (llm_scheme is None or expert_scheme is None):
        raise SystemExit("--cascade needs both an LLM scheme and an expert scheme")
    if args.cascade and llm_scheme not in schemes.LLM_FOLDED_SCHEMES:
        raise SystemExit(f"--cascade emulates a folded LLM; {llm_scheme!r} folds nothing")
    llm_params = json.loads(args.llm_params)
    expert_params = json.loads(args.expert_params)

    out = Path(args.output_dir) / "onnx"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    policy = calibration.load_policy(args.checkpoint_dir, config_name=args.config, device=args.device, compile=False)
    dataset = calibration.load_dataset(args.dataset_path)
    logger.info("policy + dataset (%d episodes) in %.0fs", len(dataset.meta.episodes), time.time() - t0)

    samples, observations = calibration.sample_observations(dataset, args.num_calib, seed=args.seed)
    loop = calibration.make_forward_loop(policy, observations, seed=args.seed)
    llm = llm_module(policy)

    shapes = capture_shape_metadata(policy, observations[0], seed=args.seed)
    logger.info("captured shapes: %s", shapes)

    results = []
    plugin_libs: list = []
    llm_result = None
    if llm_scheme is not None:
        t1 = time.time()
        # Gemma pins the prefix length from a captured call, so the loop is
        # passed for every scheme, the per-row one included.
        llm_result = export_llm(
            llm, out / "llm_bf16.onnx", scheme=llm_scheme, forward_loop=loop, params=llm_params or None
        )
        results.append(llm_result)
        logger.info("LLM %s exported in %.0fs", llm_scheme, time.time() - t1)

    if expert_scheme is not None:
        t1 = time.time()
        emulation = None
        if args.cascade:
            assert llm_result is not None
            emulation = install_llm_emulation(llm, llm_result)
            logger.info("cascade: expert calibration runs under the quantized-LLM emulation")
        try:
            expert_result = export_expert(
                expert_view(policy),
                out / "expert_bf16.onnx",
                scheme=expert_scheme,
                forward_loop=loop,
                params=expert_params or None,
            )
        finally:
            if emulation is not None:
                emulation.remove()
        results.append(expert_result)
        logger.info("expert %s exported in %.0fs", expert_scheme, time.time() - t1)

    for r in results:
        for lib in r.plugin_libs:
            if lib not in plugin_libs:
                plugin_libs.append(lib)

    metadata = {
        "model": "pi05" if shapes["use_adarms"] else "pi0",
        "config": args.config,
        "prefix_len": shapes["prefix_len"],
        "llm_hidden_size": shapes["llm_hidden_size"],
        "llm_layers": shapes["llm_layers"],
        "llm_kv_heads": shapes["llm_kv_heads"],
        "llm_head_dim": shapes["llm_head_dim"],
        "action_horizon": shapes["action_horizon"],
        "action_dim": shapes["action_dim"],
        "use_adarms": shapes["use_adarms"],
        "num_steps": shapes["num_steps"],
        "export_mode": "foldquant",
        "precision": "bf16",
        "batch_size": shapes["batch_size"],
    }
    (out / EXPORT_METADATA_NAME).write_text(json.dumps(metadata, indent=2))
    manifest = {
        "checkpoint_dir": public_path(args.checkpoint_dir),
        "config": args.config,
        "dataset_path": public_path(args.dataset_path),
        "schemes": {r.module: r.scheme for r in results},
        "params": {"llm": llm_params, "expert": expert_params},
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
