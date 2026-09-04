# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Emit the FoldQuant plugin graphs for a SmolVLA checkpoint.

Writes, under ``--output-dir``::

    onnx/llm_bf16.onnx         FoldQuant SmolVLM2 prefix graph (unless --llm-scheme none)
    onnx/expert_bf16.onnx      FoldQuant action-expert denoise-step graph (unless --expert-scheme none)
    onnx/export_metadata.json  captured shapes, for the engine builder
    onnx/foldquant_export.json what was exported, from which samples, needing which plugins

The LLM graph is the prefix pass only: ``prefix_embs`` (SigLIP features, the
prompt embeddings and the state token, produced in PyTorch), the BOOL block
attention mask and ``position_ids`` in, the stacked post-RoPE KV cache out. Its
sequence length stays symbolic, and one capture plus the config's
``tokenizer_max_length`` give the engine profile its bounds: the images and the
state contribute a fixed number of tokens, the prompt its own length. The
expert graph is one denoise step
(``x_t``, ``timestep``, ``prefix_pad_masks``, ``kv_stack`` -> ``velocity``);
the Euler loop stays in PyTorch.

Example::

    python -m foldquant_integration.export_foldquant \\
        --checkpoint HuggingFaceVLA/smolvla_libero \\
        --dataset-path HuggingFaceVLA/libero \\
        --output-dir exports/smolvla_w8a8_w4a4 \\
        --llm-scheme w8a8_sr --expert-scheme w4a4_shg
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
from foldquant.export import export_expert, export_llm, install_llm_emulation

from . import calibration
from ._upstream import EXPORT_METADATA_NAME, LIBERO_CHECKPOINT, MANIFEST_NAME
from .runtime import PrefixCapture, expert_view, llm_module, model_of

logger = logging.getLogger("foldquant.smolvla.export")

_NONE = ("", "none", "float")


@dataclass
class ExportConfig:
    output_dir: str
    """Destination; the graphs land in ``<output_dir>/onnx``."""

    dataset_path: str
    """LeRobot dataset the calibration observations are drawn from (local path or hub id)."""

    checkpoint: str = LIBERO_CHECKPOINT
    """SmolVLA checkpoint directory or hub id (config, norm stats and processors travel inside it)."""

    llm_scheme: str = schemes.W8A8_SR
    """FoldQuant scheme for the SmolVLM2 text decoder, or ``none`` to keep it float."""

    expert_scheme: str = schemes.W4A4_SHG
    """FoldQuant scheme for the action expert, or ``none`` to keep it float."""

    num_calib: int = 128
    """Calibration observations (episode, step) pairs spread over the dataset."""

    seed: int = 0
    """Seed of the calibration sample and of the flow-matching noise replayed during calibration."""

    episodes: str = ""
    """Optional episodes to restrict the dataset load to, as indices or ranges (``0-149``, ``0,3,7``);
    a hub dataset then fetches only the files those episodes live in."""

    cascade: bool = False
    """Calibrate the expert on the activations of the *quantized* LLM (fake-quant emulation)."""

    llm_params: str = "{}"
    """JSON overrides for the LLM fold (sq_alpha, act_clip_ratio, site_bits, rot_block_size)."""

    expert_params: str = "{}"
    """JSON overrides for the expert fold (sq_alpha, sq_fold_order)."""

    device: str = "cuda"


def _scheme_or_none(value: str) -> str | None:
    return None if value.strip().lower() in _NONE else value.strip()


def capture_shape_metadata(deployed, observation: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """One inference with the prefix seam recording: the shapes the engine builder profiles.

    The prefix is **not** a constant of a SmolVLA checkpoint. Its images and
    state contribute a fixed number of tokens, but the prompt contributes its
    own length: the processor tokenizes with ``pad_language_to = "longest"``,
    which at batch 1 is no padding at all, truncated to
    ``tokenizer_max_length``. So one capture pins the fixed part and the config
    pins the bound, and the engines are profiled over the whole range rather
    than at the length that happened to be captured.
    """
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

    model = model_of(deployed)
    seen: dict[str, Any] = {}

    def _expert_hook(_m, _args):
        seen["expert_calls"] = seen.get("expert_calls", 0) + 1

    handle = model.action_in_proj.register_forward_pre_hook(_expert_hook)
    try:
        with PrefixCapture(deployed) as capture:
            calibration.infer(deployed, observation, seed=seed)
    finally:
        handle.remove()
    if len(capture.stacks) != 1:
        raise RuntimeError(f"expected one prefix pass per inference, saw {len(capture.stacks)}")
    layers, two, batch, prefix_len, kv_heads, head_dim = capture.stacks[0].shape
    if two != 2 or batch != 1:
        raise RuntimeError(f"unexpected KV stack shape {tuple(capture.stacks[0].shape)}")
    config = model.config
    lang_tokens = int(deployed.preprocessor(dict(observation))[OBS_LANGUAGE_TOKENS].shape[1])
    tokenizer_max_length = int(config.tokenizer_max_length)
    fixed = int(prefix_len) - lang_tokens  # image tokens + the state token
    if fixed < 1:
        raise RuntimeError(f"prefix {prefix_len} is shorter than its {lang_tokens} language tokens")
    seen.update(
        {
            "batch_size": int(batch),
            "prefix_len": int(prefix_len),
            "lang_tokens": lang_tokens,
            "tokenizer_max_length": tokenizer_max_length,
            "prefix_len_min": fixed + 1,
            "prefix_len_max": fixed + tokenizer_max_length,
            "llm_layers": int(layers),
            "llm_kv_heads": int(kv_heads),
            "llm_head_dim": int(head_dim),
            "llm_hidden_size": int(llm_module(deployed).config.hidden_size),
            "chunk_size": int(config.chunk_size),
            "max_action_dim": int(config.max_action_dim),
            "expert_hidden_size": int(model.vlm_with_expert.expert_hidden_size),
            "self_attn_every_n_layers": int(model.vlm_with_expert.self_attn_every_n_layers),
            "num_steps": int(seen.get("expert_calls", config.num_steps)),
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
    deployed = calibration.load_policy(args.checkpoint, device=args.device, compile=False)
    dataset = calibration.load_dataset(args.dataset_path, episodes=calibration.parse_episodes(args.episodes))
    logger.info("policy + dataset (%d episodes) in %.0fs", dataset.meta.total_episodes, time.time() - t0)

    samples, observations = calibration.sample_observations(deployed, dataset, args.num_calib, seed=args.seed)
    loop = calibration.make_forward_loop(deployed, observations, seed=args.seed)
    llm = llm_module(deployed)

    shapes = capture_shape_metadata(deployed, observations[0], seed=args.seed)
    logger.info("captured shapes: %s", shapes)

    results = []
    plugin_libs: list = []
    llm_result = None
    if llm_scheme is not None:
        t1 = time.time()
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
                expert_view(deployed),
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
        "model": "smolvla",
        "checkpoint": args.checkpoint,
        "prefix_len": shapes["prefix_len"],
        "prefix_len_min": shapes["prefix_len_min"],
        "prefix_len_max": shapes["prefix_len_max"],
        "lang_tokens": shapes["lang_tokens"],
        "tokenizer_max_length": shapes["tokenizer_max_length"],
        "llm_hidden_size": shapes["llm_hidden_size"],
        "llm_layers": shapes["llm_layers"],
        "llm_kv_heads": shapes["llm_kv_heads"],
        "llm_head_dim": shapes["llm_head_dim"],
        "chunk_size": shapes["chunk_size"],
        "max_action_dim": shapes["max_action_dim"],
        "expert_hidden_size": shapes["expert_hidden_size"],
        "self_attn_every_n_layers": shapes["self_attn_every_n_layers"],
        "num_steps": shapes["num_steps"],
        "export_mode": "foldquant",
        "precision": "bf16",
        "batch_size": shapes["batch_size"],
    }
    (out / EXPORT_METADATA_NAME).write_text(json.dumps(metadata, indent=2))
    manifest = {
        "checkpoint": args.checkpoint,
        "dataset_path": args.dataset_path,
        "episodes": args.episodes,
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
