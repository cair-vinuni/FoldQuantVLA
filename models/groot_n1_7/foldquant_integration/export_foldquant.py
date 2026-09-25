# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Emit the FoldQuant plugin graphs for a GR00T N1.7 checkpoint.

Writes, under ``--output-dir``::

    onnx/llm_bf16.onnx        FoldQuant LLM graph  (unless --llm-scheme none)
    onnx/dit_bf16.onnx        FoldQuant DiT graph  (unless --dit-scheme none)
    onnx/export_metadata.json upstream shape hints for the engine builder
    onnx/foldquant_export.json what was exported, from which samples, needing which plugins

``--save-fakequant <dir>`` also writes the arm's quant state there (see
:mod:`.fakequant`): the fake-quant checkpoint that runs the arm in PyTorch,
converts back to these graphs without calibration data, and can be pushed to
the Hugging Face Hub.

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
decoder) stay float and come from the upstream export; :mod:`.build_engines`
merges the two.

Example::

    python -m foldquant_integration.export_foldquant \\
        --model-path nvidia/GR00T-N1.7-LIBERO/libero_spatial \\
        --dataset-path demo_data/libero_demo \\
        --output-dir exports/n17_w4a4 \\
        --llm-scheme w4a4_srg --dit-scheme w4a4_shg --cascade
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

from foldquant import modelopt_int8, schemes
from foldquant.export import export_dit, export_llm, install_llm_emulation
from foldquant.provenance import public_path
import torch
import tyro

from . import calibration
from ._upstream import EXPORT_METADATA_NAME, MANIFEST_NAME


logger = logging.getLogger("foldquant.groot_n1_7.export")

#: The upstream export writes ``export_metadata.json`` with these keys; the
#: engine builder reads the first three as shape hints. ``batch_size`` is what
#: the FoldQuant LLM graph pins its batch to (the captured batch, 1).
_NONE = ("", "none")
FLOAT = "float"


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
    """FoldQuant scheme for the Qwen3-VL text tower; ``float`` takes upstream's full-pipeline export of it
    (build with ``--float-onnx-dir``); ``none`` keeps PyTorch."""

    dit_scheme: str = schemes.W4A4_SHG
    """FoldQuant scheme for the action-head DiT; ``float`` takes upstream's full-pipeline export of it
    (build with ``--float-onnx-dir``); ``none`` keeps PyTorch."""

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

    save_fakequant: Optional[str] = None
    """Also write the quant state (the fake-quant checkpoint) to this directory; needs a local
    ``--model-path``, whose content digest it records."""

    base_model_id: Optional[str] = None
    """Where others get the base checkpoint (``org/name[@revision]`` on the Hub), recorded in the
    state and its model card; defaults to the checkpoint directory's name."""

    fakequant_state_only: bool = False
    """With ``--save-fakequant``: write the quant state alone, not a self-contained model with the
    base checkpoint's files linked in."""

    fakequant_copy_base: bool = False
    """With ``--save-fakequant``: copy the base checkpoint's files into the model instead of linking them."""


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

    handles = [
        modules["llm"].register_forward_pre_hook(_llm_hook, with_kwargs=True),
        modules["dit"].register_forward_pre_hook(_dit_hook, with_kwargs=True),
    ]
    try:
        with torch.inference_mode():
            policy.get_action(observation)
    finally:
        for h in handles:
            h.remove()
    missing = [k for k in ("llm_seq_len", "vl_seq_len") if k not in seen]
    if missing:
        raise RuntimeError(f"shape capture never reached {missing}; is this an N1.7 policy?")
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
    if args.cascade and llm_scheme == FLOAT:
        raise SystemExit("--cascade emulates a folded LLM; a float LLM folds nothing")
    # N1.7's FoldQuant graphs share upstream's full-pipeline I/O contract, so the float arm
    # of a tower IS upstream's ONNX for it: nothing is emitted here, the manifest records the
    # scheme, and build_engines sources the file from --float-onnx-dir like the other five
    # components.
    float_towers = [t for t, sch in (("llm", llm_scheme), ("dit", dit_scheme)) if sch == FLOAT]
    if float_towers:
        logger.info("float tower(s) %s: taken from upstream's export at build time (--float-onnx-dir)", float_towers)
    llm_scheme = None if llm_scheme == FLOAT else llm_scheme
    dit_scheme = None if dit_scheme == FLOAT else dit_scheme
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
    record = args.save_fakequant is not None
    if record and modelopt_towers:
        raise SystemExit("--save-fakequant records FoldQuant folds; the ModelOpt baselines have no quant state")
    if record and not Path(args.model_path).is_dir():
        raise SystemExit("--save-fakequant needs --model-path to be a local checkpoint directory (its digest is recorded)")

    if modelopt_towers:
        modelopt_int8.ensure_cuda_ext()

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

    modelopt_calls: Dict[str, list] = {}
    if modelopt_towers:
        from . import modelopt_export

        # Captured before anything is quantized: every ModelOpt tower calibrates on float inputs.
        t1 = time.time()
        modelopt_calls = modelopt_export.capture({t: modules[t] for t in modelopt_towers}, loop)
        logger.info("ModelOpt calibration capture in %.0fs", time.time() - t1)

    results = []
    plugin_libs: list = []
    llm_result = None
    if llm_scheme is not None and "llm" not in modelopt_towers:
        t1 = time.time()
        # No final norm: the upstream backbone reads hidden_states[-1], the last
        # decoder layer's PRE-norm output (see upstream export_llm_to_onnx).
        llm_result = export_llm(
            modules["llm"],
            out / "llm_bf16.onnx",
            scheme=llm_scheme,
            forward_loop=loop,
            params=llm_params or None,
            final_norm=False,
            record=record,
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
                out / "dit_bf16.onnx",
                scheme=dit_scheme,
                forward_loop=loop,
                params=dit_params or None,
                record=record,
            )
        finally:
            if emulation is not None:
                emulation.remove()
        results.append(dit_result)
        logger.info("DiT %s exported in %.0fs", dit_scheme, time.time() - t1)

    # After the FoldQuant towers: ModelOpt quantizes in place, so a FoldQuant replay
    # that ran later would calibrate through a fake-quantized tower.
    modelopt_records: Dict[str, Any] = {}
    modelopt_files: Dict[str, str] = {}
    if "llm" in modelopt_towers:
        t1 = time.time()
        modelopt_records["llm"] = modelopt_export.export_llm(
            modules["llm"],
            modelopt_calls["llm"],
            out / "llm_bf16.onnx",
            num_layers=int(policy.model.backbone.select_layer),
            algorithm=modelopt_towers["llm"],
            opset=args.modelopt_opset,
        )
        modelopt_files["llm"] = "llm_bf16.onnx"
        logger.info("LLM %s exported in %.0fs", modelopt_towers["llm"], time.time() - t1)
    if "dit" in modelopt_towers:
        t1 = time.time()
        modelopt_records["dit"] = modelopt_export.export_dit(
            modules["dit"],
            modelopt_calls["dit"],
            out / "dit_bf16.onnx",
            algorithm=modelopt_towers["dit"],
            opset=args.modelopt_opset,
        )
        modelopt_files["dit"] = "dit_bf16.onnx"
        logger.info("DiT %s exported in %.0fs", modelopt_towers["dit"], time.time() - t1)

    for r in results:
        for lib in r.plugin_libs:
            if lib not in plugin_libs:
                plugin_libs.append(lib)
    # A weight-only ModelOpt arm carries Int4GroupwiseGemmPlugin nodes after the
    # surgery in modelopt_export; the manifest is what build_engines and serve read
    # to load a plugin library, so declare it here rather than re-derive it.
    if any(modelopt_int8.is_weight_only(sch) for sch in modelopt_towers.values()):
        from foldquant.kernels.locator import INT4_GROUPWISE_LIB

        if INT4_GROUPWISE_LIB not in plugin_libs:
            plugin_libs.append(INT4_GROUPWISE_LIB)

    metadata = {
        "model_version": "n1d7",
        "sa_seq_len": shapes["sa_seq_len"],
        "vl_seq_len": shapes["vl_seq_len"],
        "llm_seq_len": shapes["llm_seq_len"],
        "llm_hidden_size": shapes["llm_hidden_size"],
        "num_deepstack": shapes["num_deepstack"],
        "num_vis_tokens": shapes["num_vis_tokens"],
        "action_horizon": shapes["action_horizon"],
        "embodiment_tag": str(policy.embodiment_tag),
        "export_mode": "foldquant",
        "precision": "bf16",
        "batch_size": shapes["batch_size"],
    }
    (out / EXPORT_METADATA_NAME).write_text(json.dumps(metadata, indent=2))

    manifest = {
        "model_path": public_path(args.model_path),
        "embodiment_tag": str(policy.embodiment_tag),
        "dataset_path": public_path(args.dataset_path),
        "schemes": {
            **{r.module: r.scheme for r in results},
            **modelopt_towers,
            **{t: FLOAT for t in float_towers},
        },
        **{t: FLOAT for t in float_towers},
        "params": {"llm": llm_params, "dit": dit_params},
        "cascade": bool(args.cascade),
        "plugin_libs": plugin_libs,
        "files": {**{r.module: r.onnx_path.name for r in results}, **modelopt_files},
        "modelopt": modelopt_records,
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
    if record:
        from .fakequant import save_arm_state

        save_arm_state(
            Path(args.save_fakequant),
            model_path=args.model_path,
            results=results,
            export_metadata=metadata,
            export_manifest=manifest,
            base_model_id=args.base_model_id,
            bundle=not args.fakequant_state_only,
            copy_base=args.fakequant_copy_base,
        )
    return out


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
