# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Quantize a GR00T N1.7 checkpoint with FoldQuant and save the quantized model.

Calibrates the LLM and DiT folds on ``--num-calib`` dataset observations
and writes ``--output-dir``, the quantized model::

    the base checkpoint with each quantized projection's weight
                       replaced by its integer codes (qweight) and per-row scale
                       (weight_scale), plus the SmoothQuant scales and fold settings

The quantized model runs in PyTorch with fake-quant layers in place of the
quantized projections (``serve``, ``verify`` and ``eval_libero`` load it like a
checkpoint), can be pushed to the Hugging Face Hub, and is the input of the
rest of the pipeline: ``export`` (quantized model -> plugin ONNX) and
``build_engines`` (ONNX -> TensorRT engines). A tower quantized by ModelOpt is
saved the same way: the checkpoint holds its smoothed weights, the quantizers'
scales and the call its graph is traced from, and ``export`` traces it.

``--llm-scheme`` / ``--dit-scheme modelopt_w8a8_smoothquant`` (INT8 SmoothQuant
Q/DQ) and ``modelopt_w4a16_awq`` (INT4 weight-only AWQ, group 128, rewritten to
``Int4GroupwiseGemmPlugin`` nodes) export a comparison baseline instead of a
FoldQuant graph: NVIDIA ModelOpt graphs with the same file names and I/O
contract (see :mod:`.modelopt_export`). ``build_engines``, ``verify`` and
``serve`` take them unchanged; the INT4 arm's plugin library is declared in the
manifest like any other.

The two graphs are drop-in replacements for the files of the same name that
upstream ``export_onnx_n1d7.py --export-mode full_pipeline`` writes: same
input / output names, dtypes and dynamic-dim names, so upstream's engine
builder and ``trt_model_forward.py`` consume them unchanged. The other five
pipeline components (ViT, VL self-attention, state / action encoders, action
decoder) stay float; ``export`` writes them beside the two graphs with
upstream's own exporters, from the shapes recorded here.

Example::

    python -m foldquant_integration.quantize \\
        --model-path nvidia/GR00T-N1.7-LIBERO/libero_spatial \\
        --dataset-path demo_data/libero_demo \\
        --output-dir exports/n17_w4a4/quantized \\
        --llm-scheme w4a4_srg --dit-scheme w4a4_shg --cascade
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import shutil
from pathlib import Path
import time
from typing import Any, Dict, Optional

from foldquant import modelopt_int8, schemes
from foldquant.export import ExportResult, export_dit, export_llm, install_llm_emulation
from foldquant.trace_export import call_kwargs, record_trace
from foldquant.quant_state import ModuleQuantState
from foldquant.provenance import public_path
from foldquant.quantized_checkpoint import is_quantized_checkpoint
import torch
import tyro

from . import calibration
from .quantized import tower_trace


logger = logging.getLogger("foldquant.groot_n1_7.quantize")

#: The upstream export writes ``export_metadata.json`` with these keys; the
#: engine builder reads the first three as shape hints. ``batch_size`` is what
#: the FoldQuant LLM graph pins its batch to (the captured batch, 1).
_NONE = ("", "none")


@dataclass
class QuantizeConfig:
    model_path: str
    """Checkpoint directory or Hugging Face id (as for the upstream tools)."""

    dataset_path: str
    """LeRobot-format dataset the calibration observations are drawn from."""

    output_dir: str
    """Destination directory of the quantized model (written as the checkpoint itself)."""

    embodiment_tag: Optional[str] = None
    """Embodiment tag; read off the checkpoint's processor_config.json when omitted."""

    llm_scheme: str = schemes.W8A8_SR
    """FoldQuant scheme for the Qwen3-VL text tower; ``none`` keeps it in PyTorch."""

    dit_scheme: str = schemes.W4A4_SHG
    """FoldQuant scheme for the action-head DiT; ``none`` keeps it in PyTorch."""

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

    modelopt_opset: int = modelopt_int8.DEFAULT_OPSET
    """ONNX opset of a ModelOpt graph (the authors' framework presets export at 20)."""

    video_backend: str = "torchcodec"
    """Video decoder for the dataset loader."""

    device: str = "cuda"

    base_model_id: Optional[str] = None
    """Where others get the base checkpoint (``org/name[@revision]`` on the Hub), recorded in the
    state and its model card; defaults to the checkpoint directory's name."""



def _scheme_or_none(value: str) -> Optional[str]:
    return None if value.strip().lower() in _NONE else value.strip()


def _module_paths(policy) -> Dict[str, torch.nn.Module]:
    """The two quantizable modules, at their upstream attribute paths."""
    return {
        "llm": policy.model.backbone.model.model.language_model,
        "dit": policy.model.action_head.model,
    }


def capture_shape_metadata(policy, observation: Dict[str, Any]) -> Dict[str, int]:
    """One forward with pre-hooks, for the shape hints upstream's builder reads."""
    modules = _module_paths(policy)
    seen: Dict[str, Any] = {}

    def _llm_hook(_m, args, kwargs):
        embeds = args[0] if args else kwargs.get("inputs_embeds")
        seen["llm_seq_len"] = int(embeds.shape[1])
        seen["llm_hidden_size"] = int(embeds.shape[2])
        seen["batch_size"] = int(embeds.shape[0])
        ds = list(kwargs.get("deepstack_visual_embeds") or [])
        seen["num_deepstack"] = len(ds)
        seen["num_vis_tokens"] = int(ds[0].shape[0]) if ds else 0

    def _dit_hook(_m, args, kwargs):
        seen["vl_seq_len"] = int(kwargs["encoder_hidden_states"].shape[1])

    def _vit_hook(_m, args, kwargs, output):
        # What upstream's ViT exporter traces with: the patch tensor's shape and the image
        # grid, fixed by the checkpoint's cameras and resolution (export_onnx_n1d7.ViTInputCapture).
        pixel_values = args[0] if args else kwargs["hidden_states"]
        grid = args[1] if len(args) > 1 else kwargs.get("grid_thw")
        seen["vit_pixel_values_shape"] = [int(d) for d in pixel_values.shape]
        seen["vit_grid_thw"] = [[int(v) for v in row] for row in grid.detach().cpu().tolist()]
        seen["num_patches"] = int(pixel_values.shape[0])
        merged = output[0] if isinstance(output, tuple) else output
        seen["num_merged_patches"] = int(merged.shape[0])

    handles = [
        modules["llm"].register_forward_pre_hook(_llm_hook, with_kwargs=True),
        modules["dit"].register_forward_pre_hook(_dit_hook, with_kwargs=True),
        policy.model.backbone.model.model.visual.register_forward_hook(_vit_hook, with_kwargs=True),
    ]
    try:
        with torch.inference_mode():
            policy.get_action(observation)
    finally:
        for h in handles:
            h.remove()
    missing = [k for k in ("llm_seq_len", "vl_seq_len", "vit_grid_thw") if k not in seen]
    if missing:
        raise RuntimeError(f"shape capture never reached {missing}; is this an N1.7 policy?")
    cfg = policy.model.action_head.config
    seen["sa_seq_len"] = 1 + int(cfg.action_horizon)
    seen["action_horizon"] = int(cfg.action_horizon)
    return seen


def export_metadata(shapes: Dict[str, Any], policy: Any) -> Dict[str, Any]:
    """``export_metadata.json``: the shape hints the engine builder reads, for any export of this checkpoint."""
    return {
        "model_version": "n1d7",
        "sa_seq_len": shapes["sa_seq_len"],
        "vl_seq_len": shapes["vl_seq_len"],
        "llm_seq_len": shapes["llm_seq_len"],
        "llm_hidden_size": shapes["llm_hidden_size"],
        "num_deepstack": shapes["num_deepstack"],
        "num_vis_tokens": shapes["num_vis_tokens"],
        "num_patches": shapes["num_patches"],
        "num_merged_patches": shapes["num_merged_patches"],
        "vit_pixel_values_shape": shapes["vit_pixel_values_shape"],
        "vit_grid_thw": shapes["vit_grid_thw"],
        "action_horizon": shapes["action_horizon"],
        "embodiment_tag": str(policy.embodiment_tag),
        "export_mode": "foldquant",
        "precision": "bf16",
        "batch_size": shapes["batch_size"],
    }


#: The file each module's graph is exported to (the manifest records it; `export` writes it).
_GRAPH_FILES = {"llm": "llm_bf16.onnx", "dit": "dit_bf16.onnx"}


def main(args: QuantizeConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    llm_scheme = _scheme_or_none(args.llm_scheme)
    dit_scheme = _scheme_or_none(args.dit_scheme)
    if llm_scheme is None and dit_scheme is None:
        raise SystemExit("nothing to export: both --llm-scheme and --dit-scheme are none")
    # ModelOpt Q/DQ baselines are routed to .modelopt_export, never to the FoldQuant emitters.
    modelopt_towers = {
        t: sch for t, sch in (("llm", llm_scheme), ("dit", dit_scheme)) if modelopt_int8.is_modelopt_scheme(sch)
    }
    if llm_scheme is not None and "llm" not in modelopt_towers:
        schemes.validate("llm", llm_scheme)
    if dit_scheme is not None and "dit" not in modelopt_towers:
        schemes.validate("dit", dit_scheme)
    if modelopt_towers and args.cascade:
        raise SystemExit("--cascade emulates a FoldQuant LLM fold; the ModelOpt baseline calibrates without it")
    for tower, tower_params in (("llm", args.llm_params), ("dit", args.dit_params)):
        if tower in modelopt_towers and json.loads(tower_params):
            raise SystemExit(f"--{tower}-params tunes a FoldQuant fold; {modelopt_towers[tower]} takes none")
    if args.cascade and (llm_scheme is None or dit_scheme is None):
        raise SystemExit("--cascade needs both an LLM scheme and a DiT scheme")
    if args.cascade and llm_scheme not in schemes.LLM_FOLDED_SCHEMES:
        raise SystemExit(f"--cascade emulates a folded LLM; {llm_scheme!r} folds nothing")
    llm_params = json.loads(args.llm_params)
    dit_params = json.loads(args.dit_params)
    if not Path(args.model_path).is_dir():
        raise SystemExit("saving the quantized model needs --model-path to be a local checkpoint directory (its files are hashed)")
    if modelopt_towers:
        modelopt_int8.ensure_cuda_ext()
    # Every arm is saved as a quantized model. The FoldQuant graphs are built in memory,
    # only to fold, round and record the weight codes; a ModelOpt tower records the call its
    # graph is traced from. `export` writes the ONNX from the quantized model.

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
    modelopt_calls: Dict[str, list] = {}
    if modelopt_towers:
        from . import modelopt_export

        # Captured before anything is quantized: every ModelOpt tower calibrates on float inputs.
        t1 = time.time()
        modelopt_calls = modelopt_export.capture({t: modules[t] for t in modelopt_towers}, loop)
        logger.info("ModelOpt calibration capture in %.0fs", time.time() - t1)

    plugin_libs: list = []
    llm_result = None
    if llm_scheme is not None and "llm" not in modelopt_towers:
        t1 = time.time()
        # No final norm: the upstream backbone reads hidden_states[-1], the last
        # decoder layer's PRE-norm output (see upstream export_llm_to_onnx).
        llm_result = export_llm(
            modules["llm"],
            None,
            scheme=llm_scheme,
            forward_loop=loop,
            params=llm_params or None,
            final_norm=False,
            record=True,
        )
        results.append(llm_result)
        logger.info("LLM %s exported in %.0fs", llm_scheme, time.time() - t1)

    if dit_scheme is not None and "dit" not in modelopt_towers:
        t1 = time.time()
        emulation = None
        if args.cascade:
            assert llm_result is not None
            emulation = install_llm_emulation(modules["llm"], llm_result)
            logger.info("cascade: DiT calibration runs under the quantized-LLM emulation")
        try:
            dit_result = export_dit(
                modules["dit"],
                None,
                scheme=dit_scheme,
                forward_loop=loop,
                params=dit_params or None,
                record=True,
            )
        finally:
            if emulation is not None:
                emulation.remove()
        results.append(dit_result)
        logger.info("DiT %s exported in %.0fs", dit_scheme, time.time() - t1)

    # After the FoldQuant towers: ModelOpt quantizes in place, so a FoldQuant replay
    # that ran later would calibrate through a fake-quantized tower. The checkpoint keeps
    # the smoothed weights, the quantizers' scales and the first captured call as the trace.
    for tower, algo in modelopt_towers.items():
        t1 = time.time()
        module = modules[tower]
        record = modelopt_export.quantize_tower(module, modelopt_calls[tower], algorithm=algo)
        config = {
            "bits": 4 if modelopt_int8.is_weight_only(algo) else 8,
            "act_bits": 16 if modelopt_int8.is_weight_only(algo) else 8,
            "modelopt": record,
            "modelopt_state": modelopt_int8.modelopt_state_of(module),
            "opset": args.modelopt_opset,
        }
        if tower == "llm":
            config["num_layers"] = int(policy.model.backbone.select_layer)
        ms = ModuleQuantState(tower, algo, config=config)
        ms.put_group("modelopt", modelopt_int8.quantizer_buffers(module))
        record_trace(ms, tower_trace(tower, call_kwargs(module, *modelopt_calls[tower][0])))
        libs = []
        if modelopt_int8.is_weight_only(algo):
            from foldquant.kernels.locator import INT4_GROUPWISE_LIB

            libs = [INT4_GROUPWISE_LIB]
        results.append(ExportResult(tower, algo, None, libs, state=ms))
        logger.info("%s %s quantized in %.0fs", tower, algo, time.time() - t1)
    modelopt_calls = {}

    for r in results:
        for lib in r.plugin_libs:
            if lib not in plugin_libs:
                plugin_libs.append(lib)

    metadata = export_metadata(shapes, policy)
    results_by = {r.module: r for r in results}
    manifest = {
        "model_path": public_path(args.model_path),
        "embodiment_tag": str(policy.embodiment_tag),
        "dataset_path": public_path(args.dataset_path),
        "schemes": {r.module: r.scheme for r in results},
        "params": {"llm": llm_params, "dit": dit_params},
        "cascade": bool(args.cascade),
        "plugin_libs": plugin_libs,
        "files": {r.module: _GRAPH_FILES[r.module] for r in results},
        "modelopt": {t: results_by[t].state.config["modelopt"] for t in modelopt_towers},
        "calibration": {
            "seed": args.seed,
            "num_samples": len(samples),
            "samples": [asdict(s) for s in samples],
        },
    }
    from .quantized import save_arm_state

    model_dir = Path(args.output_dir)
    if model_dir.exists() and any(model_dir.iterdir()):
        if not is_quantized_checkpoint(model_dir):
            raise SystemExit(f"{model_dir} exists and is not a FoldQuant quantized model; refusing to overwrite it")
        logger.info("replacing %s", model_dir)
        shutil.rmtree(model_dir)
    save_arm_state(
        model_dir,
        model_path=args.model_path,
        results=results,
        export_metadata=metadata,
        export_manifest=manifest,
        base_model_id=args.base_model_id,
        policy=policy,
    )
    logger.info("quantized model: %s", model_dir)
    return model_dir


if __name__ == "__main__":
    main(tyro.cli(QuantizeConfig))
