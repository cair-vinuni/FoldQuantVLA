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

``--llm-scheme`` / ``--expert-scheme modelopt_w8a8_smoothquant`` export a
comparison baseline instead of a FoldQuant graph: NVIDIA ModelOpt INT8
SmoothQuant Q/DQ graphs with the same file names and I/O contract (see
:mod:`.modelopt_export`). ``build_engines``, ``verify`` and ``serve`` take them
unchanged.

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

from foldquant import modelopt_int8
from foldquant import schemes
from foldquant.export import export_expert
from foldquant.export import export_llm
from foldquant.export import install_llm_emulation
from foldquant.float_export import FLOAT, Binding, export_with_example
from foldquant.provenance import public_path
import torch
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

_NONE = ("", "none")


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

    modelopt_opset: int = modelopt_int8.DEFAULT_OPSET
    """ONNX opset of a ``modelopt_w8a8_smoothquant`` graph (the framework preset exports at 20)."""

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



# ---------------------------------------------------------------------------
# Float engines under the KV-stack contract. Pi0.5's
# prefix seam already receives upstream's 4-D additive mask, so it passes through.
# ---------------------------------------------------------------------------
class _Stop(Exception):
    pass


def _capture_prefix(model, forward_loop):
    pwe = model.paligemma_with_expert
    seen, orig = {}, pwe.forward

    def spy(*a, **k):
        if k.get("inputs_embeds") is not None and k["inputs_embeds"][1] is None:
            seen.update({kk: (vv.detach() if torch.is_tensor(vv) else vv) for kk, vv in k.items()})
            raise _Stop()
        return orig(*a, **k)

    pwe.forward = spy
    try:
        with torch.inference_mode():
            try:
                forward_loop(pwe)
            except _Stop:
                pass
    finally:
        pwe.__dict__.pop("forward", None)
    if "attention_mask" not in seen:
        raise RuntimeError("forward_loop never reached the prefix pass of paligemma_with_expert")
    return seen


def export_llm_float_pi05(policy, onnx_path, *, forward_loop=None, seen=None, opset=17):
    """Trace the live prefix pass. *seen* (the prefix call's kwargs) skips the capture replay."""
    from .runtime import model_of, stack_cache
    model = model_of(policy)
    if seen is None:
        seen = _capture_prefix(model, forward_loop)
    prefix = seen["inputs_embeds"][0]
    am, pid = seen["attention_mask"], seen["position_ids"]
    am_dtype = torch.bfloat16 if am.is_floating_point() else torch.bool   # the runtime feeds the engine bf16

    pwe = model.paligemma_with_expert

    def call(prefix_embs, attention_mask, position_ids, **_):
        # upstream's own prefix forward: HF's decoder applies RoPE differently.
        _hidden, cache = pwe.forward(attention_mask=attention_mask, position_ids=position_ids,
                                     past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
        return stack_cache(cache)

    return export_with_example(
        pwe, onnx_path, module_name="llm",
        bindings=[Binding("prefix_embs", "prefix_embs", torch.bfloat16),
                  Binding("attention_mask", "attention_mask", am_dtype),
                  Binding("position_ids", "position_ids", torch.int64)],
        output_name="kv_stack",
        example_kwargs={"prefix_embs": prefix.to(torch.bfloat16), "attention_mask": am.to(am_dtype), "position_ids": pid.to(torch.int64)},
        extract=lambda o: o, call=call, opset=opset,
    )


def export_expert_float_pi05(policy, onnx_path, *, forward_loop=None, seen=None, opset=17):
    """Trace the live denoise step. *seen* (one step's inputs, KV stacked) skips the capture replay."""
    from .runtime import cache_from_stack, model_of, stack_cache
    model = model_of(policy)
    orig = model.denoise_step
    if seen is not None:
        seen = dict(seen)
        forward_loop = None
    else:
        seen = {}

    def spy(state, prefix_pad_masks, past_key_values, x_t, timestep):
        seen.update(state=state.detach(), prefix_pad_masks=prefix_pad_masks.detach(),
                    kv_stack=(past_key_values if torch.is_tensor(past_key_values) else stack_cache(past_key_values)).detach(),
                    x_t=x_t.detach(), timestep=timestep.detach())
        raise _Stop()

    if forward_loop is not None:
        model.denoise_step = spy
        try:
            with torch.inference_mode():
                try:
                    forward_loop(model)
                except _Stop:
                    pass
        finally:
            model.__dict__.pop("denoise_step", None)
    if "x_t" not in seen:
        raise RuntimeError("forward_loop never reached denoise_step")

    def call(x_t, timestep, prefix_pad_masks, kv_stack, state, **_):
        return orig(state, prefix_pad_masks.to(torch.bool), cache_from_stack(kv_stack), x_t, timestep.reshape(-1))

    return export_with_example(
        model, onnx_path, module_name="expert",
        bindings=[Binding("x_t", "x_t", torch.float32), Binding("timestep", "timestep", torch.float32),
                  Binding("prefix_pad_masks", "prefix_pad_masks", torch.bool), Binding("kv_stack", "kv_stack", torch.bfloat16),
                  Binding("state", "state", torch.float32)],
        output_name="velocity",
        example_kwargs={"x_t": seen["x_t"].to(torch.float32), "timestep": seen["timestep"].reshape(1).to(torch.float32),
                        "prefix_pad_masks": seen["prefix_pad_masks"].to(torch.bool), "kv_stack": seen["kv_stack"].to(torch.bfloat16),
                        "state": seen["state"].to(torch.float32)},
        extract=lambda o: o, call=call, opset=opset,
    )

def main(args: ExportConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    llm_scheme = _scheme_or_none(args.llm_scheme)
    expert_scheme = _scheme_or_none(args.expert_scheme)
    if llm_scheme is None and expert_scheme is None:
        raise SystemExit("nothing to export: both --llm-scheme and --expert-scheme are none")
    # ModelOpt Q/DQ baselines are routed to .modelopt_export, never to the FoldQuant emitters.
    modelopt_towers = {
        t: sch for t, sch in (("llm", llm_scheme), ("expert", expert_scheme)) if modelopt_int8.is_modelopt_scheme(sch)
    }
    if llm_scheme is not None and "llm" not in modelopt_towers:
        schemes.validate("llm", llm_scheme)
    if expert_scheme is not None and "expert" not in modelopt_towers:
        schemes.validate("expert", expert_scheme)
    if modelopt_towers and args.cascade:
        raise SystemExit("--cascade emulates a FoldQuant LLM fold; the ModelOpt baseline calibrates without it")
    for tower, tower_params in (("llm", args.llm_params), ("expert", args.expert_params)):
        if tower in modelopt_towers and json.loads(tower_params):
            raise SystemExit(f"--{tower}-params tunes a FoldQuant fold; {modelopt_towers[tower]} takes none")
    if args.cascade and (llm_scheme is None or expert_scheme is None):
        raise SystemExit("--cascade needs both an LLM scheme and an expert scheme")
    if args.cascade and llm_scheme not in schemes.LLM_FOLDED_SCHEMES:
        raise SystemExit(f"--cascade emulates a folded LLM; {llm_scheme!r} folds nothing")
    llm_params = json.loads(args.llm_params)
    expert_params = json.loads(args.expert_params)

    if modelopt_towers:
        modelopt_int8.ensure_cuda_ext()

    out = Path(args.output_dir) / "onnx"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    policy = calibration.load_policy(args.checkpoint_dir, config_name=args.config, device=args.device, compile=False)
    dataset = calibration.load_dataset(args.dataset_path)
    logger.info("policy + dataset (%d episodes) in %.0fs", len(dataset.meta.episodes), time.time() - t0)

    samples, observations = calibration.sample_observations(
        dataset, args.num_calib, seed=args.seed, keys=calibration.resolve_keys(dataset, args.config)
    )
    loop = calibration.make_forward_loop(policy, observations, seed=args.seed)
    llm = llm_module(policy)

    shapes = capture_shape_metadata(policy, observations[0], seed=args.seed)
    logger.info("captured shapes: %s", shapes)

    modelopt_captures = None
    if modelopt_towers:
        from . import modelopt_export

        # Captured before anything is quantized: every ModelOpt tower calibrates on float inputs.
        t1 = time.time()
        modelopt_captures = modelopt_export.capture(policy, loop, modelopt_towers)
        logger.info("ModelOpt calibration capture in %.0fs", time.time() - t1)

    results = []
    plugin_libs: list = []
    llm_result = None
    if llm_scheme is not None and "llm" not in modelopt_towers:
        t1 = time.time()
        # Gemma pins the prefix length from a captured call, so the loop is
        # passed for every scheme, the per-row one included.
        if llm_scheme == FLOAT:
            llm_result = export_llm_float_pi05(policy, out / "llm_bf16.onnx", forward_loop=loop)
        else:
            llm_result = export_llm(
                llm, out / "llm_bf16.onnx", scheme=llm_scheme, forward_loop=loop, params=llm_params or None
            )
        results.append(llm_result)
        logger.info("LLM %s exported in %.0fs", llm_scheme, time.time() - t1)

    if expert_scheme is not None and "expert" not in modelopt_towers:
        t1 = time.time()
        emulation = None
        if args.cascade:
            assert llm_result is not None
            emulation = install_llm_emulation(llm, llm_result)
            logger.info("cascade: expert calibration runs under the quantized-LLM emulation")
        try:
            if expert_scheme == FLOAT:
                expert_result = export_expert_float_pi05(policy, out / "expert_bf16.onnx", forward_loop=loop)
            else:
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

    # After the FoldQuant towers: ModelOpt quantizes in place, so a FoldQuant replay
    # that ran later would calibrate through a fake-quantized tower.
    modelopt_records: dict[str, Any] = {}
    modelopt_files: dict[str, str] = {}
    for tower, exporter, file_name in (
        ("llm", "export_llm", "llm_bf16.onnx"),
        ("expert", "export_expert", "expert_bf16.onnx"),
    ):
        if tower not in modelopt_towers:
            continue
        t1 = time.time()
        modelopt_records[tower] = getattr(modelopt_export, exporter)(
            policy, modelopt_captures, out / file_name, algorithm=modelopt_towers[tower], opset=args.modelopt_opset
        )
        modelopt_files[tower] = file_name
        logger.info("%s %s exported in %.0fs", tower, modelopt_towers[tower], time.time() - t1)
    modelopt_captures = None

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
        "schemes": {**{r.module: r.scheme for r in results}, **modelopt_towers},
        "params": {"llm": llm_params, "expert": expert_params},
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
    logger.info("wrote %s and %s (total %.0fs)", EXPORT_METADATA_NAME, MANIFEST_NAME, time.time() - t0)
    return out


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
