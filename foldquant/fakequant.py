# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The FoldQuant fake-quant pipeline: fake-quant state -> real-quant ONNX -> TensorRT engines.

A fake-quant model is one directory: the base checkpoint's files (weights,
index, configs; linked or copied, see :func:`bundle_base`) next to the
:mod:`foldquant.quant_state` files calibrated on them. Loading it means loading
the policy from that directory as usual and calling :func:`install_fake_quant`
on its quantized modules, which replaces their projections with the arithmetic
the TensorRT engine runs (the same codes, scales, rotations and per-token
quantization). The family tools do that by themselves when ``--model-path``
names a fake-quant model. A state saved without the base files (``state_only``)
is applied to a separately loaded base checkpoint instead. The same directory converts to the plugin ONNX graphs with
:func:`foldquant.export.export_llm` / ``export_dit`` / ``export_expert`` and
``state=``, without calibration data.

Coverage: the LLM backbones (Qwen3 / Qwen3-VL, Gemma prefix) for the folded
schemes, the GR00T DiT for every DiT scheme, and the π₀.₅ action expert for
every expert scheme (:mod:`foldquant.expert_fake_quant`).

Command line, run inside a model family's environment (``models/<family>``, with
``foldquant`` and the family's ``foldquant_integration`` importable)::

    python -m foldquant.fakequant info    --fakequant-dir D
    python -m foldquant.fakequant bundle  --fakequant-dir D --model-path CKPT [--copy]
    python -m foldquant.fakequant push    --fakequant-dir D --repo-id org/name [--public] [--dry-run]
    python -m foldquant.fakequant to-onnx --fakequant-dir D --output-dir OUT
    python -m foldquant.fakequant build   --output-dir OUT [family engine options]
    python -m foldquant.fakequant convert --fakequant-dir D --output-dir OUT [...]

(``--model-path CKPT`` names the base checkpoint only for a state saved without it.)

``to-onnx`` writes ``OUT/onnx`` (the real-quant plugin graphs, byte-identical to
the calibrating export), ``build`` compiles ``OUT/onnx`` into ``OUT/engines``,
``convert`` does both. The family-specific parts (loading the policy, which
modules are quantized, how the engine directory is assembled) come from the
family's adapter, ``foldquant_integration.fakequant``, which defines ``FAMILY``,
``load_policy(model_path, embodiment_tag, device)``, ``module_paths(policy)``
and ``build_engines(onnx_dir, engine_dir, **options)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from . import schemes
from .quant_state import MANIFEST_NAME, TENSORS_NAME, ModuleQuantState, QuantState, load_state, save_state

logger = logging.getLogger(__name__)

__all__ = [
    "EXPORT_MANIFEST_NAME",
    "FakeQuantHandle",
    "bundle_base",
    "fakequant_arm",
    "is_fakequant_model",
    "resolve_model_path",
    "export_onnx",
    "install_fake_quant",
    "load_adapter",
    "push_to_hub",
    "save_arm_state",
    "verify_base_checkpoint",
    "write_model_card",
]

#: What a push uploads: the state and its card, nothing else that lands in the directory.
PUSHED_FILES = (MANIFEST_NAME, TENSORS_NAME, "README.md")

#: The export manifest every family writes beside its graphs (``build_engines`` and
#: the runtimes read the plugin libraries off it).
EXPORT_MANIFEST_NAME = "foldquant_export.json"


class FakeQuantHandle:
    """Undo token for :func:`install_fake_quant`."""

    def __init__(self, handles: List[Any]) -> None:
        self._handles = handles

    def remove(self) -> None:
        for h in reversed(self._handles):
            h.remove()
        self._handles = []


#: The row order each LLM site's weights are concatenated in by the emitter
#: (``llm._emit_layer``): the order its recorded codes are in.
_LLM_SITES = {
    "qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "o": ("self_attn.o_proj",),
    "gateup": ("mlp.gate_proj", "mlp.up_proj"),
    "down": ("mlp.down_proj",),
}


def install_llm_fake_quant(module: Any, ms: ModuleQuantState) -> Any:
    """Fake-quant a Qwen / Gemma LLM from its state, with the engine's codes, in fp32.

    Every projection becomes a :class:`foldquant.fake_quant_linear.FakeQuantLinear`
    over the site's weight codes and per-row scales: the recorded GPTQ codes for
    the INT4 sites, round-to-nearest of the folded weight (the emitter's rounding)
    for the INT8 ones. The SmoothQuant fold the kernel absorbs into the norm gain
    is applied as a division of the (unchanged) norm output, which is the same
    function; the down site's scale is already inside the up codes.
    """
    import torch

    from .fake_quant_linear import FakeQuantLinear, SwapHandle, swap_module
    from .llm import resolve_qwen3_decoder
    from .llm_rotation_sq import apply_rot_fold, apply_sq_fold
    from .rotation import hadamard_blocks
    from .weights import quant_weight_per_row

    if ms.scheme not in schemes.LLM_FOLDED_SCHEMES:
        raise NotImplementedError(
            f"LLM fake-quant covers the folded schemes {sorted(schemes.LLM_FOLDED_SCHEMES)}; {ms.scheme!r} is not one"
        )
    cfg = ms.config
    bits = int(cfg["bits"])
    rot_bs = int(cfg["rot_bs"])
    site_bits = dict(cfg.get("site_bits") or {})
    act_bits = cfg.get("act_bits")
    act_clip = cfg.get("act_clip", 1.0)
    sq = ms.tensor_group("sq")
    decoder = resolve_qwen3_decoder(module)
    gemma = "Gemma" in type(decoder).__name__
    qmax = {4: 7.0, 8: 127.0}
    handle = SwapHandle()
    try:
        for i, layer in enumerate(decoder.layers):
            state = {k: v.detach() for k, v in layer.state_dict().items()}
            if gemma:
                for g in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                    state[g] = state[g] + 1.0
            dev = state["input_layernorm.weight"].device
            s_qkv, s_gu, s_dn = (sq[f"L{i}_{k}"].to(dev) for k in ("qkv", "gateup", "down"))
            folded = apply_sq_fold(state, s_qkv=s_qkv, s_gu=s_gu, s_dn=s_dn)
            if rot_bs > 1:
                folded = apply_rot_fold(folded, rot_bs)
            s_pre = {"qkv": s_qkv, "gateup": s_gu}
            for site, names in _LLM_SITES.items():
                w_bits = int(site_bits.get(site, bits))
                a_bits = int(act_bits) if act_bits is not None else w_bits
                clip = float(act_clip.get(f"L{i}_{site}", 1.0)) if isinstance(act_clip, dict) else float(act_clip)
                clip = clip if a_bits == 4 else 1.0
                entries = ms.gptq.get(f"llm.L{i}_{site}") or ms.gptq.get(f"llm.rtn.L{i}_{site}")
                rows = [int(folded[n + ".weight"].shape[0]) for n in names]
                if entries:
                    codes, scale = entries[0]
                    if int(codes.shape[0]) != sum(rows):
                        raise ValueError(f"llm.L{i}_{site}: recorded codes have {codes.shape[0]} rows, the layer {sum(rows)}")
                else:
                    # A format-1 state: recompute the emitter's round-to-nearest.
                    if w_bits == 4:
                        raise ValueError(f"llm.L{i}_{site}: an INT4 site without recorded GPTQ codes")
                    merged = torch.cat([folded[n + ".weight"] for n in names], dim=0)
                    c8, sc8 = quant_weight_per_row(merged)
                    codes, scale = torch.from_numpy(c8), torch.from_numpy(sc8)
                k = int(folded[names[0] + ".weight"].shape[1])
                rot = {}
                if rot_bs > 1:
                    perm, R = hadamard_blocks(k, rot_bs)
                    rot = {"perm": perm, "rot": R, "rot_bs": rot_bs}
                start = 0
                for name, n in zip(names, rows):
                    parent_name, _, leaf = name.rpartition(".")
                    parent = layer.get_submodule(parent_name)
                    lin = getattr(parent, leaf)
                    if name == "mlp.up_proj" and lin.bias is not None:
                        raise NotImplementedError("an up_proj bias would need the down site's scale folded in")
                    fq = FakeQuantLinear(
                        codes[start : start + n], scale[start : start + n],
                        None if lin.bias is None else lin.bias.detach(),
                        a_qmax=qmax[a_bits], s_pre=s_pre.get(site), a_clip=clip, **rot,
                    )
                    swap_module(handle, parent, leaf, fq)
                    start += n
    except Exception:
        handle.remove()
        raise
    logger.info(
        "LLM fake-quant (%s) installed: %d layers, %d projections, %s", ms.scheme, len(decoder.layers), len(handle),
        "recorded GPTQ codes" if ms.gptq else "round-to-nearest",
    )
    return handle


def install_fake_quant(modules: Mapping[str, Any], state: QuantState) -> FakeQuantHandle:
    """Install the fake-quant of every module *state* quantized.

    Args:
        modules: ``{"llm": ..., "dit": ...}`` (or ``"expert"``), the live modules
            at the same paths the export read.
        state: a loaded quant state.

    Raises:
        KeyError: *state* quantized a module *modules* does not provide.
        NotImplementedError: a module kind or scheme without a PyTorch reader.
    """
    handles: List[Any] = []
    try:
        for name, ms in state.modules.items():
            if name not in modules:
                raise KeyError(f"quant state covers {name!r}, which was not passed in")
            if name == "llm":
                handles.append(install_llm_fake_quant(modules[name], ms))
            elif name == "dit":
                from .dit_fake_quant import install_dit_fake_quant

                handles.append(install_dit_fake_quant(modules[name], ms))
            elif name == "expert":
                from .expert_fake_quant import install_expert_fake_quant

                handles.append(install_expert_fake_quant(modules[name], ms))
            else:
                raise NotImplementedError(
                    f"no PyTorch fake-quant for the {name!r} module yet; its state converts to ONNX only"
                )
    except Exception:
        for h in reversed(handles):
            h.remove()
        raise
    return FakeQuantHandle(handles)


def verify_base_checkpoint(state: QuantState, checkpoint: Any) -> None:
    """Refuse a base checkpoint whose content is not the one *state* was calibrated on.

    Every file the state recorded (weights, index, configs) must be present with
    the same content; files it did not record (a README, a licence, Hub metadata)
    are ignored, so a download or a copy of the checkpoint passes.
    """
    from .eval_protocol import artifact_digest, file_hashes

    recorded = (state.manifest.get("base") or {}).get("file_hashes")
    if recorded:
        root = Path(checkpoint)
        if not root.is_dir():
            raise FileNotFoundError(f"base checkpoint {checkpoint} not found")
        have = file_hashes(root)
        missing = sorted(set(recorded) - set(have))
        changed = sorted(k for k in recorded if k in have and have[k] != recorded[k])
        if missing or changed:
            raise ValueError(
                f"{checkpoint} is not the checkpoint this quant state was calibrated on: "
                + (f"missing {missing[:3]} " if missing else "")
                + (f"different content in {changed[:3]}" if changed else "")
                + ". Its codes and scales would be applied to other weights."
            )
        return
    want = (state.manifest.get("base") or {}).get("digest")
    if not want:
        raise ValueError("the quant state records no base-checkpoint digest")
    got = artifact_digest(checkpoint)
    if got is None or got.get("missing"):
        raise FileNotFoundError(f"base checkpoint {checkpoint} not found")
    if got["digest"] != want:
        raise ValueError(
            f"{checkpoint} is not the checkpoint this quant state was calibrated on "
            f"(content digest {got['digest'][:12]}..., expected {want[:12]}...). "
            "Its codes and scales would be applied to other weights."
        )


def write_model_card(directory: Any, state: QuantState, *, repo_id: Optional[str] = None) -> Path:
    """A ``README.md`` for the Hub: what the arm is, what it needs, how to use it."""
    m = state.manifest
    base = m.get("base") or {}
    bundled = bool(base.get("bundled"))
    mods = {k: v.scheme for k, v in state.modules.items()}
    family = m.get("family", "?")
    lines = [
        "---",
        "library_name: foldquant",
        "tags: [foldquant, quantization, vision-language-action, fake-quant]",
        *( [f"base_model: {base['model_id']}"] if base.get("model_id") and "/" in str(base.get("model_id")) else [] ),
        "---",
        "",
        f"# FoldQuant fake-quant state: {family} {' / '.join(f'{k} {v}' for k, v in mods.items())}",
        "",
        *(
            [
                "A self-contained FoldQuant fake-quant model: the base checkpoint's files plus the quant state",
                "(SmoothQuant scales, fold settings and the integer weight codes) of the modules listed below.",
                "Load it like the base checkpoint to run the quantized policy in PyTorch",
            ]
            if bundled
            else [
                "This repository holds the calibration result of a FoldQuant arm, not model weights:",
                "SmoothQuant scales, fold settings and the integer weight codes, for the modules listed below.",
                "Apply it to the base checkpoint it was calibrated on to run the quantized policy in PyTorch",
            ]
        ),
        "(fake-quant: the engine's own weight codes, scales and activation transforms, computed in fp32),",
        "or convert it to the FoldQuant plugin ONNX graphs, byte-identical to the calibrating export, and",
        "build TensorRT engines, with no calibration data.",
        "",
        "| module | scheme | params |",
        "|---|---|---|",
        *[f"| `{k}` | `{v.scheme}` | `{json.dumps(v.config.get('params') or {})}` |" for k, v in state.modules.items()],
        "",
        f"Base checkpoint: `{base.get('model_id', '?')}` (content digest `{str(base.get('digest', '?'))[:16]}`).",
        f"Calibration: {m.get('calibration', {}).get('num_samples', '?')} observations, "
        f"seed {m.get('calibration', {}).get('seed', '?')}, from `{m.get('dataset_path', '?')}`.",
        "",
        "## Use",
        "",
        "With the FoldQuantVLA repository and the family's environment:",
        "",
        "```bash",
        f"huggingface-cli download {repo_id or '<this repo>'} --local-dir fq_model",
        *(
            [
                "# PyTorch fake-quant policy, e.g. LIBERO or a robot server",
                "python -m foldquant_integration.eval_libero --protocol p3 --model-path fq_model --output <out>",
                "# real-quant plugin ONNX graphs, then TensorRT engines on the target device",
                "python -m foldquant.fakequant convert --fakequant-dir fq_model --output-dir exports/<arm>",
            ]
            if bundled
            else [
                "# PyTorch fake-quant policy, e.g. LIBERO or a robot server",
                "python -m foldquant_integration.eval_libero --protocol p3 --model-path <base checkpoint> \\",
                "    --fakequant-dir fq_model --output <out>",
                "# real-quant plugin ONNX graphs, then TensorRT engines on the target device",
                "python -m foldquant.fakequant convert --fakequant-dir fq_model \\",
                "    --model-path <base checkpoint> --output-dir exports/<arm>",
            ]
        ),
        "```",
        "",
        "The base checkpoint is checked against the recorded digest before anything is applied.",
        "Its license governs any use of this state together with it.",
        "",
        f"Files: `{MANIFEST_NAME}` (settings), `{TENSORS_NAME}` (scales and codes).",
        "",
    ]
    path = Path(directory) / "README.md"
    path.write_text("\n".join(lines))
    return path


def is_fakequant_model(path: Any) -> bool:
    """True when *path* is a directory holding a FoldQuant quant state."""
    return path is not None and (Path(path) / MANIFEST_NAME).is_file()


def fakequant_arm(model_path: Any, fakequant_dir: Optional[str], *, no_fakequant: bool = False, other_arms: Any = ()) -> Optional[str]:
    """The fake-quant state a family tool should run: *fakequant_dir* if given, else *model_path*
    when it is a fake-quant model and no other arm (engines, a baseline pack) was asked for."""
    if fakequant_dir or no_fakequant or any(other_arms):
        return fakequant_dir
    if is_fakequant_model(model_path):
        logger.info("%s is a FoldQuant fake-quant model; running it fake-quantized (--no-fakequant for bf16)", model_path)
        return str(model_path)
    return None


def _base(state: QuantState) -> Dict[str, Any]:
    return state.manifest.get("base") or {}


def bundle_base(directory: Any, model_path: Any, *, copy: bool = False) -> Dict[str, Any]:
    """Put the base checkpoint's files into a state directory: the self-contained fake-quant model.

    Every file the state recorded for the base (weights, index, configs) is
    linked (or, with ``copy``, copied) under the same relative path, in real
    sub-directories, so the directory loads as the base checkpoint, uploads
    with its weights (the Hub client reads through file links, not directory
    links) and converts with no other path. The files are checked against the
    recorded hashes first.
    """
    import os
    import shutil

    directory = Path(directory)
    state = load_state(directory)
    verify_base_checkpoint(state, model_path)
    src_root = Path(model_path).expanduser().resolve()
    ours = {MANIFEST_NAME, TENSORS_NAME, "README.md"}
    names = sorted(_base(state).get("file_hashes") or {})
    clash = sorted(set(names) & ours)
    if clash:
        raise ValueError(f"the base checkpoint has files named like the quant state's: {clash}")
    total = 0
    for rel in names:
        src = (src_root / rel).resolve()
        dst = directory / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
        total += src.stat().st_size
    manifest_path = directory / MANIFEST_NAME
    data = json.loads(manifest_path.read_text())
    data.setdefault("base", {})["bundled"] = "copy" if copy else "link"
    manifest_path.write_text(json.dumps(data, indent=2))
    write_model_card(directory, load_state(directory))
    logger.info("bundled %d base files (%.1f GB, %s) into %s", len(names), total / 1e9, "copied" if copy else "linked", directory)
    return {"files": len(names), "bytes": total, "mode": "copy" if copy else "link"}


def resolve_model_path(state: QuantState, fakequant_dir: Any, model_path: Optional[str]) -> str:
    """The base checkpoint to load: *model_path*, else the fake-quant model directory itself."""
    if model_path:
        return str(model_path)
    if _base(state).get("bundled"):
        return str(fakequant_dir)
    raise SystemExit(
        f"{fakequant_dir} holds the quant state only; pass --model-path <base checkpoint>, or make it a "
        "self-contained model with `python -m foldquant.fakequant bundle`"
    )


def push_to_hub(
    directory: Any,
    repo_id: str,
    *,
    private: bool = True,
    dry_run: bool = False,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    state_only: bool = False,
) -> Dict[str, Any]:
    """Upload a fake-quant model to a Hugging Face model repo (private by default).

    A self-contained model uploads its base files too (read through their
    links); ``state_only`` uploads the quant state alone. Nothing else in the
    directory is uploaded. ``dry_run`` lists the files without contacting the Hub.
    """
    src = Path(directory)
    for required in (MANIFEST_NAME, TENSORS_NAME):
        if not (src / required).is_file():
            raise FileNotFoundError(f"{src} holds no {required}; not a quant-state directory")
    names = list(PUSHED_FILES)
    state = load_state(src)
    if _base(state).get("bundled") and not state_only:
        names += sorted(_base(state).get("file_hashes") or {})
    missing = [n for n in names if n not in PUSHED_FILES and not (src / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{src}: bundled base files missing {missing[:3]}; rerun `bundle`")
    files = sorted(p for p in (src / n for n in names) if p.is_file())
    listing = [{"path": p.relative_to(src).as_posix(), "bytes": p.stat().st_size} for p in files]
    report: Dict[str, Any] = {"repo_id": repo_id, "private": private, "files": listing, "dry_run": dry_run,
                              "self_contained": bool(_base(state).get("bundled")) and not state_only}
    if dry_run:
        return report
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.upload_folder(
        folder_path=str(src),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f["path"] for f in listing],
        commit_message=commit_message or "Upload FoldQuant fake-quant model",
    )
    report["commit"] = getattr(info, "oid", None) or str(info)
    return report


# The pipeline: state -> ONNX -> engines


def save_arm_state(
    directory: Any,
    *,
    family: str,
    model_path: str,
    results: List[Any],
    export_manifest: Dict[str, Any],
    extra_files: Optional[Dict[str, Any]] = None,
    base_model_id: Optional[str] = None,
    embodiment_tag: Optional[str] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
    bundle: bool = True,
    copy_base: bool = False,
) -> Path:
    """Write the fake-quant model of an export that ran with ``record=True``.

    With ``bundle`` (the default) the directory is self-contained: the base
    checkpoint's files are linked into it (copied with ``copy_base``), see
    :func:`bundle_base`. Without it only the quant state is written.

    ``base_model_id`` names the base checkpoint where others can get it (a Hub
    ``org/name``, optionally ``@revision``); it defaults to the path's basename.
    ``embodiment_tag`` / ``load_kwargs`` are what the family adapter's
    ``load_policy`` needs to rebuild the same policy (a data config, a train
    config name); both default to what the export manifest says.
    The digest of every graph the export wrote is recorded, and
    :func:`export_onnx` refuses to finish when its rebuild differs.

    *export_manifest* is the family's ``foldquant_export.json`` content;
    *extra_files* are other JSON files the family writes beside its graphs
    (``{"export_metadata.json": {...}}``), rewritten verbatim by :func:`export_onnx`.
    """
    from .eval_protocol import file_hashes
    from .provenance import public_path

    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"{model_path}: the base checkpoint must be a local directory")
    hashes = file_hashes(model_path)
    import hashlib

    base = {"digest": hashlib.sha256(json.dumps(sorted(hashes.items())).encode()).hexdigest(), "files": len(hashes)}
    state = QuantState(
        manifest={
            "family": family,
            "base": {
                "model_id": base_model_id or public_path(model_path),
                "digest": base["digest"],
                "files": base["files"],
                "file_hashes": hashes,
            },
            "graphs": {r.module: _graph_digest(r.onnx_path) for r in results if r.state is not None},
            "graph_digest": "content",
            "embodiment_tag": embodiment_tag if embodiment_tag is not None else export_manifest.get("embodiment_tag"),
            "load_kwargs": dict(load_kwargs or {}),
            "dataset_path": export_manifest.get("dataset_path"),
            "calibration": {k: (export_manifest.get("calibration") or {}).get(k) for k in ("seed", "num_samples")},
            "cascade": export_manifest.get("cascade", False),
            "export_manifest": export_manifest,
            "extra_files": dict(extra_files or {}),
        },
        modules={r.module: r.state for r in results if r.state is not None},
    )
    save_state(state, directory)
    write_model_card(directory, state)
    if bundle:
        bundle_base(directory, model_path, copy=copy_base)
    logger.info(
        "wrote fake-quant %s %s (%s)", "model" if bundle else "state", directory,
        ", ".join(f"{k} {v.scheme}" for k, v in state.modules.items()),
    )
    return Path(directory)


def _graph_digest(onnx_path: Any) -> str:
    """SHA-256 over a graph's content (nodes, attributes, initializers, I/O, opsets) and
    its external-data sidecars, not the file header: ``ir_version`` and the producer
    fields an ``onnx`` release stamps would otherwise refuse a correct conversion."""
    import hashlib

    import onnx

    path = Path(onnx_path)
    model = onnx.load(str(path), load_external_data=False)
    h = hashlib.sha256()
    h.update(model.graph.SerializeToString(deterministic=True))
    for op in sorted((o.domain, o.version) for o in model.opset_import):
        h.update(repr(op).encode())
    for f in sorted(p for p in path.parent.glob(path.name + ".*") if p.is_file()):
        h.update(f.name.replace(path.name, "", 1).encode() + b"\0")
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


class GraphMismatchError(RuntimeError):
    """A graph rebuilt from a quant state differs from the one its calibrating export wrote."""


def export_onnx(state: QuantState, modules: Mapping[str, Any], output_dir: Any, *, source: Any = None) -> Path:
    """Write the real-quant plugin graphs of *state* to ``<output_dir>/onnx``.

    Each module's graph is emitted from its state (no calibration data, no GPTQ
    solve), under the file name the calibrating export used, together with the
    export manifest and the family's other JSON files.
    """
    from .eval_protocol import artifact_digest
    from .export import export_module
    from .provenance import public_path

    out = Path(output_dir) / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(state.manifest.get("export_manifest") or {})
    files = manifest.get("files") or {}
    results = []
    for name, ms in state.modules.items():
        if name not in modules:
            raise KeyError(f"quant state covers {name!r}, which the family adapter does not provide")
        kwargs: Dict[str, Any] = {}
        if name == "llm" and ms.config.get("max_seq_len"):
            kwargs["max_seq_len"] = int(ms.config["max_seq_len"])
        path = out / files.get(name, f"{name}.onnx")
        results.append(export_module(name, modules[name], path, scheme=ms.scheme, state=ms, **kwargs))
        want = (state.manifest.get("graphs") or {}).get(name) if state.manifest.get("graph_digest") == "content" else None
        if want is not None and _graph_digest(path) != want:
            raise GraphMismatchError(
                f"{path.name} rebuilt from the quant state differs from the graph the calibrating export "
                "wrote. The weights, the FoldQuant version or the numerical libraries differ from the "
                "machine that recorded the state (a dense rotation's SVD, for one); this graph would not "
                "match the fake-quant model. Rebuild with the recording environment or recalibrate."
            )
    for fname, content in (state.manifest.get("extra_files") or {}).items():
        (out / fname).write_text(json.dumps(content, indent=2))
    libs: List[str] = list(manifest.get("plugin_libs") or [])
    for r in results:
        for lib in r.plugin_libs:
            if lib not in libs:
                libs.append(lib)
    manifest["plugin_libs"] = libs
    if state.manifest.get("family"):
        manifest["family"] = state.manifest["family"]
    if source is not None:
        manifest["from_quant_state"] = {
            "path": public_path(str(source)),
            "digest": (artifact_digest(source) or {}).get("digest"),
        }
    (out / EXPORT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s from the quant state (%s)", out, ", ".join(r.onnx_path.name for r in results))
    return out


def load_adapter(family: Optional[str] = None) -> Any:
    """The active family's adapter, ``foldquant_integration.fakequant``.

    Raises:
        ImportError: no family integration is importable (run from ``models/<family>``).
        ValueError: the importable integration is not *family*.
    """
    try:
        adapter = importlib.import_module("foldquant_integration.fakequant")
    except ImportError as exc:
        raise ImportError(
            "no family adapter importable: run inside a model family's environment with "
            "models/<family> on PYTHONPATH (it provides foldquant_integration.fakequant)"
        ) from exc
    have = getattr(adapter, "FAMILY", None)
    if family is not None and have != family:
        raise ValueError(f"the quant state is for {family!r}, but the importable integration is {have!r}")
    return adapter


def install_on_policy(policy: Any, fakequant_dir: Any, model_path: Optional[str] = None, *, check_checkpoint: bool = True) -> Any:
    """Load a state, check the checkpoint *policy* came from, install the fake-quant: ``(handle, state)``."""
    state = load_state(fakequant_dir)
    adapter = load_adapter(state.manifest.get("family"))
    if check_checkpoint:
        verify_base_checkpoint(state, resolve_model_path(state, fakequant_dir, model_path))
    handle = install_fake_quant(adapter.module_paths(policy), state)
    logger.info("fake-quant arm from %s: %s", fakequant_dir, ", ".join(f"{k} {v.scheme}" for k, v in state.modules.items()))
    return handle, state


# Command line


@dataclass
class Info:
    """Print what a quant-state directory holds."""

    fakequant_dir: str


@dataclass
class Bundle:
    """Make a state-only directory a self-contained fake-quant model (link or copy the base files in)."""

    fakequant_dir: str
    model_path: str
    """The base checkpoint the state was calibrated on (its files are checked first)."""
    copy: bool = False
    """Copy the base files instead of linking them (needed to move the directory elsewhere)."""


@dataclass
class Push:
    """Upload a fake-quant model to a Hugging Face model repo."""

    fakequant_dir: str
    repo_id: str
    public: bool = False
    """Create the repo public; private by default."""
    dry_run: bool = False
    """List what would be uploaded, without contacting the Hub."""
    state_only: bool = False
    """Upload the quant state without the base checkpoint's files."""
    token: Optional[str] = None


@dataclass
class ToOnnx:
    """Fake-quant state -> real-quant plugin ONNX in ``<output_dir>/onnx`` (no dataset, no GPTQ)."""

    fakequant_dir: str
    output_dir: str
    model_path: Optional[str] = None
    """The base checkpoint, for a state saved without it; a self-contained model is its own."""
    embodiment_tag: Optional[str] = None
    """Defaults to the tag the state was calibrated with."""
    device: str = "cuda"
    skip_checkpoint_check: bool = False


@dataclass
class Build:
    """``<output_dir>/onnx`` -> TensorRT engines in ``<output_dir>/engines``, by the family's builder."""

    output_dir: str
    float_onnx_dir: Optional[str] = None
    """Float graphs of the components FoldQuant does not replace (GR00T N1.7: upstream's export)."""
    float_engine_dir: Optional[str] = None
    """Or their already-built engines, copied."""
    max_batch: int = 1
    """Batch bound of every optimization profile; 1 for a single-robot deployment."""
    workspace_mb: int = 8192


@dataclass
class Convert(ToOnnx):
    """``to-onnx`` then ``build``: fake-quant state -> real-quant ONNX -> TensorRT engines."""

    float_onnx_dir: Optional[str] = None
    float_engine_dir: Optional[str] = None
    max_batch: int = 1
    workspace_mb: int = 8192


def info(args: Info) -> Dict[str, Any]:
    state = load_state(args.fakequant_dir)
    summary = {
        "family": state.manifest.get("family"),
        "base": state.manifest.get("base"),
        "embodiment_tag": state.manifest.get("embodiment_tag"),
        "calibration": state.manifest.get("calibration"),
        "cascade": state.manifest.get("cascade"),
        "modules": {
            k: {"scheme": v.scheme, "params": v.config.get("params"), "gptq_sites": len(v.gptq), "tensors": len(v.tensors)}
            for k, v in state.modules.items()
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


def bundle(args: Bundle) -> Dict[str, Any]:
    return bundle_base(args.fakequant_dir, args.model_path, copy=args.copy)


def push(args: Push) -> Dict[str, Any]:
    state = load_state(args.fakequant_dir)
    if not args.dry_run:
        write_model_card(args.fakequant_dir, state, repo_id=args.repo_id)  # names this repo in its commands
    report = push_to_hub(
        args.fakequant_dir, args.repo_id, private=not args.public, dry_run=args.dry_run, token=args.token,
        state_only=args.state_only,
    )
    size = sum(f["bytes"] for f in report["files"])
    logger.info(
        "%s %d files (%.1f MB) to %s (%s)", "would upload" if args.dry_run else "uploaded", len(report["files"]),
        size / 1e6, args.repo_id, "public" if args.public else "private",
    )
    return report


def to_onnx(args: ToOnnx) -> Path:
    state = load_state(args.fakequant_dir)
    adapter = load_adapter(state.manifest.get("family"))
    model_path = resolve_model_path(state, args.fakequant_dir, args.model_path)
    if not args.skip_checkpoint_check:
        verify_base_checkpoint(state, model_path)
    policy = adapter.load_policy(
        model_path,
        args.embodiment_tag or state.manifest.get("embodiment_tag"),
        args.device,
        **(state.manifest.get("load_kwargs") or {}),
    )
    return export_onnx(state, adapter.module_paths(policy), args.output_dir, source=args.fakequant_dir)


def build(args: Union[Build, Convert]) -> Path:
    onnx_dir = Path(args.output_dir) / "onnx"
    manifest = json.loads((onnx_dir / EXPORT_MANIFEST_NAME).read_text())
    adapter = load_adapter(manifest.get("family"))
    engine_dir = Path(args.output_dir) / "engines"
    adapter.build_engines(
        onnx_dir, engine_dir, float_onnx_dir=args.float_onnx_dir, float_engine_dir=args.float_engine_dir,
        max_batch=args.max_batch, workspace_mb=args.workspace_mb,
    )
    logger.info("engines in %s", engine_dir)
    return engine_dir


def convert(args: Convert) -> Path:
    to_onnx(args)
    return build(args)


def main(argv: Optional[List[str]] = None) -> Any:
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cmd = tyro.extras.subcommand_cli_from_dict(
        {"info": Info, "bundle": Bundle, "push": Push, "to-onnx": ToOnnx, "build": Build, "convert": Convert}, args=argv
    )
    handlers = {Info: info, Bundle: bundle, Push: push, ToOnnx: to_onnx, Build: build, Convert: convert}
    return handlers[type(cmd)](cmd)


if __name__ == "__main__":
    main()
