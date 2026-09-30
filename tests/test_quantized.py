# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Quant state, GPTQ record/replay and the fake-quant path.

The acceptance gate of the fake-quant path is that a graph exported from a
saved quant state is byte-identical to the graph the calibrating export wrote,
for every scheme, with no calibration data. The fake-quant PyTorch modules are
read off that same graph, so they are checked for wiring (a wrong permutation,
rotation or scale shows up as a collapsed output cosine), not re-derived.
"""

from __future__ import annotations

import filecmp
import sys
from pathlib import Path

import pytest
import torch

from foldquant import schemes
from foldquant.llm_gptq import gptq_prepare, gptq_quant_codes, tag_site
from foldquant.quant_state import ModuleQuantState, QuantState, ReplayError, ReplaySite, record_gptq, replay_sites

_N17 = Path(__file__).resolve().parents[1] / "models" / "groot_n1_7"


def _hessian(k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4 * k, k, generator=g)
    return (x.T @ x).double()


# Quant state and GPTQ record / replay


def test_replay_returns_the_recorded_codes_in_call_order() -> None:
    w1, w2 = torch.randn(8, 32), torch.randn(8, 32)
    prep = tag_site(gptq_prepare(_hessian(32)), "dit.encoder")
    with record_gptq() as sink:
        c1, s1 = gptq_quant_codes(w1, prep, qmax=7)
        c2, s2 = gptq_quant_codes(w2, prep, qmax=7)
        gptq_quant_codes(w1, gptq_prepare(_hessian(32)), qmax=7)  # untagged: not recorded
    assert list(sink) == ["dit.encoder"] and len(sink["dit.encoder"]) == 2
    ms = ModuleQuantState("dit", schemes.W4A4_SHG, gptq=sink)
    session = replay_sites(ms)
    site = gptq_prepare(session["encoder"])
    assert isinstance(site, ReplaySite)
    r1, rs1 = gptq_quant_codes(torch.zeros(8, 32), site, qmax=7)
    r2, _ = gptq_quant_codes(torch.zeros(8, 32), site, qmax=7)
    assert torch.equal(r1, c1) and torch.equal(r2, c2) and torch.equal(rs1, s1)
    session.assert_consumed()


def test_replay_fails_loudly_on_a_mismatch() -> None:
    codes = torch.zeros(4, 8, dtype=torch.int8)
    site = ReplaySite("dit.x", [(codes, torch.ones(4))])
    with pytest.raises(ReplayError, match="checkpoint"):
        site.take(torch.zeros(5, 8))
    site.take(torch.zeros(4, 8))
    with pytest.raises(ReplayError, match="holds only"):
        site.take(torch.zeros(4, 8))
    session = replay_sites(ModuleQuantState("dit", "w4a4_shg", gptq={"dit.a": [(codes, torch.ones(4))]}))
    with pytest.raises(ReplayError, match="unused"):
        session.assert_consumed()


# Exports: record, save, load, replay; byte-identical graphs


def _same_export(a: Path, b: Path) -> None:
    assert filecmp.cmp(a, b, shallow=False), f"{a.name}: graph differs from the recorded export"
    for sidecar in a.parent.glob(a.name + "*"):
        if sidecar != a:
            other = b.parent / sidecar.name.replace(a.name, b.name)
            assert filecmp.cmp(sidecar, other, shallow=False), f"{sidecar.name}: weights differ"


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


@pytest.fixture(scope="module")
def tiny_dit():
    pytest.importorskip("diffusers")
    sys.path.insert(0, str(_N17))
    try:
        from gr00t.model.modules.dit import AlternateVLDiT
    except Exception as exc:  # pragma: no cover - environment without the N1.7 stack
        pytest.skip(f"upstream N1.7 DiT not importable: {exc}")
    torch.manual_seed(0)
    dit = AlternateVLDiT(
        num_attention_heads=2, attention_head_dim=64, output_dim=16, num_layers=4, dropout=0.0,
        interleave_self_attention=True, cross_attention_dim=128, final_dropout=False,
        attend_text_every_n_blocks=1,
    ).eval()
    g = torch.Generator().manual_seed(1)
    inputs = []
    for i in range(3):
        mask = torch.zeros(1, 12, dtype=torch.bool)
        mask[:, :7] = True
        inputs.append(
            dict(
                hidden_states=torch.randn(1, 9, 128, generator=g),
                encoder_hidden_states=torch.randn(1, 12, 128, generator=g) * 3,
                timestep=torch.tensor([100 * (i + 1)]),
                image_mask=mask,
                backbone_attention_mask=torch.ones(1, 12, dtype=torch.bool),
            )
        )

    def loop(module):
        with torch.no_grad():
            for kw in inputs:
                module(**kw)

    return dit, loop, inputs


@pytest.mark.parametrize(
    "scheme,floor", [(schemes.W8A8, 0.995), (schemes.W8A8_SH, 0.995), (schemes.W4A4_SHG, 0.85), (schemes.W4A4_SR, 0.85)]
)
def test_the_dit_fake_quant_tracks_the_float_module(tiny_dit, tmp_path: Path, scheme: str, floor: float) -> None:
    from foldquant.dit_fake_quant import FakeQuantLinear, install_dit_fake_quant
    from foldquant.export import export_dit

    dit, loop, inputs = tiny_dit
    res = export_dit(dit, tmp_path / "dit_bf16.onnx", scheme=scheme, forward_loop=loop, record=True)
    with torch.no_grad():
        ref = [dit(**kw) for kw in inputs]
    handle = install_dit_fake_quant(dit, res.state)
    try:
        n_fq = sum(isinstance(m, FakeQuantLinear) for m in dit.modules())
        assert n_fq == 4 * 6 + (4 if scheme.startswith("w4a4") else 0)  # q,k,v,o,ffn0,ffn2 (+ AdaLN at W4)
        with torch.no_grad():
            out = [dit(**kw) for kw in inputs]
    finally:
        handle.remove()
    assert not any(isinstance(m, FakeQuantLinear) for m in dit.modules())
    cos = min(_cos(a, b) for a, b in zip(ref, out))
    assert cos > floor, f"{scheme}: fake-quant output cosine {cos:.4f} against float"


@pytest.fixture(scope="module")
def tiny_qwen3():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(
        vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64, max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3Model(cfg).eval()
    g = torch.Generator().manual_seed(2)
    embeds = [torch.randn(1, 10, 128, generator=g) for _ in range(3)]

    def loop(module):
        with torch.no_grad():
            for e in embeds:
                module(inputs_embeds=e)

    return model, loop, embeds


@pytest.mark.parametrize("scheme", [schemes.W4A4_SRG, schemes.W8A8_SR])
def test_the_llm_fake_quant_uses_the_engine_codes(tiny_qwen3, tmp_path: Path, scheme: str) -> None:
    """Every projection runs on the codes and scales the graph packs, and tracks the float model."""
    from foldquant.export import export_llm
    from foldquant.fake_quant_linear import FakeQuantLinear
    from foldquant.quantized import install_llm_fake_quant

    model, loop, embeds = tiny_qwen3
    res = export_llm(model, tmp_path / "llm_bf16.onnx", scheme=scheme, forward_loop=loop, record=True, max_seq_len=64)
    with torch.no_grad():
        ref = [model(inputs_embeds=e).last_hidden_state for e in embeds]
    handle = install_llm_fake_quant(model, res.state)
    try:
        assert sum(isinstance(m, FakeQuantLinear) for m in model.modules()) == 2 * 7
        k = model.layers[1].self_attn.k_proj
        key = "llm.L1_qkv" if scheme == schemes.W4A4_SRG else "llm.rtn.L1_qkv"
        codes, scale = res.state.gptq[key][0]
        q_rows = model.layers[1].self_attn.q_proj.out_features
        assert torch.equal(k.codes, codes[q_rows : q_rows + k.out_features].to(k.codes.dtype))
        assert torch.equal(k.w_scale, scale[q_rows : q_rows + k.out_features])
        with torch.no_grad():
            out = [model(inputs_embeds=e).last_hidden_state for e in embeds]
    finally:
        handle.remove()
    assert not any(isinstance(m, FakeQuantLinear) for m in model.modules())
    floor = 0.85 if scheme == schemes.W4A4_SRG else 0.995
    assert min(_cos(a, b) for a, b in zip(ref, out)) > floor


def test_the_llm_fake_quant_keeps_8_bit_sites_at_8_bit_activations(tiny_qwen3, tmp_path: Path) -> None:
    """W4A4 with o/down at 8 bit: those sites run the W8A8 plugin pair, so their fake-quant
    quantizes activations to 8 bit too, not to the arm's 4."""
    from foldquant.export import export_llm
    from foldquant.quantized import install_llm_fake_quant

    model, loop, _ = tiny_qwen3
    res = export_llm(model, tmp_path / "llm_bf16.onnx", scheme=schemes.W4A4_SRG, forward_loop=loop, record=True,
                     max_seq_len=64, params={"site_bits": {"o": 8, "down": 8}})
    handle = install_llm_fake_quant(model, res.state)
    try:
        layer = model.layers[0]
        assert layer.self_attn.o_proj.a_qmax == 127.0 and layer.mlp.down_proj.a_qmax == 127.0
        assert layer.self_attn.q_proj.a_qmax == 7.0 and layer.mlp.gate_proj.a_qmax == 7.0
    finally:
        handle.remove()


# Checkpoint guard and Hub upload (dry run)



def _base_checkpoint(directory: Path, module, *, shards: int = 1) -> Path:
    """A tiny base checkpoint: *module*'s tensors as safetensors (sharded if asked), a config,
    and tokenizer / processor files as a real checkpoint carries."""
    import json

    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().clone().contiguous() for k, v in module.state_dict().items()}
    keys = sorted(sd)
    if shards == 1:
        save_file(sd, str(directory / "model.safetensors"))
    else:
        weight_map = {}
        per = -(-len(keys) // shards)
        for i in range(shards):
            name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
            part = {k: sd[k] for k in keys[i * per : (i + 1) * per]}
            save_file(part, str(directory / name))
            weight_map.update({k: name for k in part})
        (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (directory / "config.json").write_text("{}")
    (directory / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 1000000000000000019884624838656, "padding_side": "left"}))
    (directory / "processor_config.json").write_text(json.dumps({"max_length": None}))
    return directory


def _save(out: Path, ckpt: Path, module, res, name: str, **kw):
    from foldquant import quantized

    return quantized.save_arm_state(
        out, family=kw.pop("family", "t"), model_path=str(ckpt), results=[res], modules={name: module},
        checkpoint_root=module, export_manifest=kw.pop("export_manifest", {"files": {name: f"{name}_bf16.onnx"}}), **kw,
    )


def _without_quantized_weights(module, out: Path):
    """A copy of *module* whose quantized projections hold zeros, as a loader of the checkpoint sees them."""
    import copy

    from foldquant.quantized_checkpoint import quantized_keys

    m = copy.deepcopy(module)
    with torch.no_grad():
        for key in quantized_keys(out):
            m.get_parameter(key).zero_()
    return m


def test_a_downloaded_copy_passes_and_changed_weights_do_not(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    """Extra files (a README, Hub metadata) are ignored; a changed or missing weight file is refused."""
    import os

    from foldquant import quantized
    from foldquant.export import export_dit

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, loop, _ = tiny_dit
    ckpt = _base_checkpoint(tmp_path / "ckpt", dit, shards=2)
    res = export_dit(dit, None, scheme=schemes.W8A8, forward_loop=loop, record=True)
    _save(tmp_path / "s", ckpt, dit, res, "dit")
    state = quantized.load_quantized_model(tmp_path / "s")
    assert set(state.manifest["base"]["file_hashes"]) == {
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json",
        "config.json", "tokenizer_config.json", "processor_config.json",
    }
    quantized.verify_base_checkpoint(state, tmp_path / "s")  # the checkpoint is its own base

    copy = tmp_path / "copy"
    copy.mkdir()
    for f in ckpt.iterdir():
        os.link(f, copy / f.name)
    (copy / "README.md").write_text("model card")
    (copy / "LICENSE").write_text("licence")
    (copy / ".cache").mkdir()
    (copy / ".cache" / "x").write_text("etag")
    quantized.verify_base_checkpoint(state, copy)

    (copy / "model-00002-of-00002.safetensors").unlink()
    (copy / "model-00002-of-00002.safetensors").write_bytes(b"C" * 64)
    with pytest.raises(ValueError, match="different content"):
        quantized.verify_base_checkpoint(state, copy)
    (copy / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="missing"):
        quantized.verify_base_checkpoint(state, copy)


def test_a_state_refuses_a_checkpoint_it_was_not_calibrated_on(tmp_path: Path, monkeypatch) -> None:
    from foldquant.eval_protocol import artifact_digest
    from foldquant.quantized import verify_base_checkpoint

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"A" * 64)
    state = QuantState(manifest={"base": {"digest": artifact_digest(ckpt)["digest"]}})
    verify_base_checkpoint(state, ckpt)
    (ckpt / "model.safetensors").write_bytes(b"B" * 64)
    with pytest.raises(ValueError, match="not the checkpoint"):
        verify_base_checkpoint(state, ckpt)


def test_push_dry_run_lists_the_checkpoint_without_contacting_the_hub(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    from foldquant import quantized
    from foldquant.export import export_dit
    from foldquant.quantized import push_to_hub, write_model_card

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, loop, _ = tiny_dit
    ckpt = _base_checkpoint(tmp_path / "ckpt", dit)
    res = export_dit(dit, None, scheme=schemes.W8A8_SH, forward_loop=loop, record=True)
    out = _save(tmp_path / "q", ckpt, dit, res, "dit", family="groot_n1_7", base_model_id="org/base")
    (out / "scratch.log").write_text("x")
    (out / ".hidden").write_text("x")
    card = write_model_card(out, quantized.load_quantized_model(out), repo_id="org/fq")
    assert "base_model: org/base" in card.read_text()
    report = push_to_hub(out, "org/fq", dry_run=True)
    names = sorted(f["path"] for f in report["files"])
    assert report["private"] and "model.safetensors" in names and "foldquant_quant.json" in names
    assert "tokenizer_config.json" in names and "scratch.log" in names and ".hidden" not in names
    with pytest.raises(FileNotFoundError):
        push_to_hub(tmp_path / "missing", "org/fq", dry_run=True)


def test_the_pipeline_saves_a_state_and_rebuilds_the_same_graph(tiny_dit, tmp_path: Path, monkeypatch, capsys) -> None:
    """save_arm_state -> info -> export_onnx from the checkpoint alone, through foldquant.quantized."""
    import json

    from foldquant import quantized
    from foldquant.export import export_dit

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, loop, _ = tiny_dit
    ckpt = _base_checkpoint(tmp_path / "ckpt", dit)
    direct = tmp_path / "direct" / "onnx"
    res = export_dit(dit, direct / "dit_bf16.onnx", scheme=schemes.W4A4_SHG, forward_loop=loop, record=True)
    manifest = {"files": {"dit": "dit_bf16.onnx"}, "calibration": {"seed": 0, "num_samples": 3}, "plugin_libs": []}
    _save(tmp_path / "state", ckpt, dit, res, "dit", family="test_family", export_manifest=manifest,
          extra_files={"export_metadata.json": {"sa_seq_len": 9}})
    quantized.main(["info", "--quantized-model", str(tmp_path / "state")])
    assert json.loads(capsys.readouterr().out)["modules"]["dit"]["scheme"] == schemes.W4A4_SHG

    state = quantized.load_quantized_model(tmp_path / "state")
    quantized.verify_base_checkpoint(state, ckpt)
    # The export runs on the module as a loader of the checkpoint sees it: no bf16 weights left
    # in the quantized projections. The recorded codes, scales and rotations carry the graph.
    out = quantized.export_onnx(state, {"dit": _without_quantized_weights(dit, tmp_path / "state")},
                                tmp_path / "replayed", source=tmp_path / "state")
    _same_export(direct / "dit_bf16.onnx", out / "dit_bf16.onnx")
    written = json.loads((out / quantized.EXPORT_MANIFEST_NAME).read_text())
    assert written["family"] == "test_family" and written["plugin_libs"] == res.plugin_libs
    assert json.loads((out / "export_metadata.json").read_text()) == {"sa_seq_len": 9}



def test_the_pipeline_rebuilds_an_llm_graph_too(tiny_qwen3, tmp_path: Path, monkeypatch) -> None:
    from foldquant import quantized
    from foldquant.export import export_llm

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    model, loop, _ = tiny_qwen3
    ckpt = _base_checkpoint(tmp_path / "ckpt", model)
    direct = tmp_path / "direct"
    direct.mkdir()
    res = export_llm(model, direct / "llm_bf16.onnx", scheme=schemes.W4A4_SRG, forward_loop=loop, record=True,
                     max_seq_len=64, params={"site_bits": {"o": 8, "down": 8}})
    _save(tmp_path / "state", ckpt, model, res, "llm")
    state = quantized.load_quantized_model(tmp_path / "state")
    out = quantized.export_onnx(state, {"llm": _without_quantized_weights(model, tmp_path / "state")}, tmp_path / "replayed")
    _same_export(direct / "llm_bf16.onnx", out / "llm_bf16.onnx")


def test_hub_metadata_in_a_checkpoint_copy_does_not_change_its_digest(tmp_path: Path, monkeypatch) -> None:
    from foldquant.eval_protocol import artifact_digest

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"W" * 32)
    before = artifact_digest(ckpt)["digest"]
    (ckpt / ".cache" / "huggingface").mkdir(parents=True)
    (ckpt / ".cache" / "huggingface" / "model.safetensors.metadata").write_text("etag")
    (ckpt / ".gitattributes").write_text("*.safetensors filter=lfs")
    assert artifact_digest(ckpt)["digest"] == before



def test_a_rebuilt_graph_that_differs_is_refused(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    from foldquant import quantized
    from foldquant.export import export_dit

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, loop, _ = tiny_dit
    ckpt = _base_checkpoint(tmp_path / "ckpt", dit)
    res = export_dit(dit, None, scheme=schemes.W8A8_SH, forward_loop=loop, record=True)
    _save(tmp_path / "s", ckpt, dit, res, "dit")
    state = quantized.load_quantized_model(tmp_path / "s")
    assert state.manifest["graphs"]["dit"] == res.graph_digest
    assert state.manifest["graph_digest"] == "inline"
    state.manifest["graphs"]["dit"] = "0" * 64
    with pytest.raises(quantized.GraphMismatchError):
        quantized.export_onnx(state, {"dit": dit}, tmp_path / "r")


def _noisy(fn):
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, dict):
            return {k: (v * (1 + 1e-4 * torch.randn_like(v.float())).to(v.dtype) if k.endswith("proj.weight") else v)
                    for k, v in out.items()}
        return out * (1 + 1e-4 * torch.randn_like(out.float())).to(out.dtype)
    return wrapped


def test_a_replay_does_not_depend_on_the_device_arithmetic_of_the_fold(tiny_dit, tiny_qwen3, tmp_path: Path, monkeypatch) -> None:
    """Round-to-nearest codes of a folded weight flip with the fold's float rounding (CPU vs GPU);
    recorded, the replayed graph stays byte-identical. A fold perturbed at 1e-4 stands in for another device."""
    import foldquant.llm as llm_mod
    import foldquant.rotation as rot_mod
    from foldquant.export import export_dit, export_llm

    dit, dloop, _ = tiny_dit
    model, lloop, _ = tiny_qwen3
    for d in ("a", "b", "c"):
        (tmp_path / d).mkdir()
    dit_first = export_dit(dit, tmp_path / "a" / "dit_bf16.onnx", scheme=schemes.W8A8_SH, forward_loop=dloop, record=True)
    llm_first = export_llm(model, tmp_path / "a" / "llm_bf16.onnx", scheme=schemes.W8A8_SR, forward_loop=lloop,
                           record=True, max_seq_len=64)
    assert any(k.startswith("dit.rtn") for k in dit_first.state.gptq)
    assert any(k.startswith("llm.rtn.L0_") for k in llm_first.state.gptq)

    monkeypatch.setattr(rot_mod, "fold_weight_sq", _noisy(rot_mod.fold_weight_sq))
    monkeypatch.setattr(llm_mod, "apply_rot_fold", _noisy(llm_mod.apply_rot_fold))
    export_dit(dit, tmp_path / "b" / "dit_bf16.onnx", scheme=schemes.W8A8_SH, state=dit_first.state)
    export_llm(model, tmp_path / "b" / "llm_bf16.onnx", scheme=schemes.W8A8_SR, state=llm_first.state, max_seq_len=64)
    _same_export(tmp_path / "a" / "dit_bf16.onnx", tmp_path / "b" / "dit_bf16.onnx")
    _same_export(tmp_path / "a" / "llm_bf16.onnx", tmp_path / "b" / "llm_bf16.onnx")

    # The perturbation is large enough to matter: without the state, the codes move.
    export_dit(dit, tmp_path / "c" / "dit_bf16.onnx", scheme=schemes.W8A8_SH, forward_loop=dloop)
    assert not filecmp.cmp(tmp_path / "a" / "dit_bf16.onnx", tmp_path / "c" / "dit_bf16.onnx", shallow=False)


def test_a_checkpoint_assembled_from_symlinks_is_read_through(tmp_path: Path, monkeypatch) -> None:
    from foldquant.eval_protocol import file_hashes

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    real = tmp_path / "real"
    (real / "experiment_cfg").mkdir(parents=True)
    (real / "experiment_cfg" / "conf.yaml").write_text("a: 1")
    (real / "model.safetensors").write_bytes(b"W")
    linked = tmp_path / "linked"
    linked.mkdir()
    for f in real.iterdir():
        (linked / f.name).symlink_to(f)
    assert file_hashes(linked) == file_hashes(real)


def _saved_model(tiny_dit, tmp_path: Path, **kw):
    from foldquant.export import export_dit

    dit, loop, _ = tiny_dit
    ckpt = _base_checkpoint(tmp_path / "ckpt", dit)
    (ckpt / "experiment_cfg").mkdir()
    (ckpt / "experiment_cfg" / "conf.yaml").write_text("a: 1")
    (ckpt / "README.md").write_text("upstream card")
    res = export_dit(dit, None, scheme=schemes.W8A8_SH, forward_loop=loop, record=True)
    out = tmp_path / "fq"
    _save(out, ckpt, dit, res, "dit", **kw)
    return ckpt, out



def test_a_saved_quantized_checkpoint_is_the_base_with_its_weights_replaced(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    import json

    from safetensors import safe_open

    from foldquant import quantized
    from foldquant.quantized_checkpoint import HF_QUANT_CONFIG_NAME, quantized_keys

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, _, _ = tiny_dit
    ckpt, out = _saved_model(tiny_dit, tmp_path)
    assert quantized.is_quantized_model(out)
    # Every non-weight file is the base's, byte for byte: in particular the tokenizer and
    # processor files, which must never be re-saved from the calibrated policy (a tokenizer
    # saved after calibration can carry that run's max_length and truncate at inference).
    for rel in ("config.json", "tokenizer_config.json", "processor_config.json", "experiment_cfg/conf.yaml"):
        assert (out / rel).is_file() and not (out / rel).is_symlink()
        assert (out / rel).read_bytes() == (ckpt / rel).read_bytes(), rel
    assert "quantized checkpoint" in (out / "README.md").read_text() and "upstream card" not in (out / "README.md").read_text()
    hf = json.loads((out / HF_QUANT_CONFIG_NAME).read_text())
    assert hf["producer"]["name"] == "foldquant" and hf["quantization"]["quant_algo"] == "W8A8"

    replaced = quantized_keys(out)
    assert replaced and all(k.endswith(".weight") for k in replaced)
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        keys = set(f.keys())
    base_keys = set(dit.state_dict())
    assert not (set(replaced) & keys), "the quantized projections' bf16 weights are gone"
    assert base_keys - set(replaced) <= keys, "everything unquantized is kept"
    for k in replaced:
        head = k[: -len(".weight")]
        assert {f"{head}.qweight", f"{head}.weight_scale"} <= keys
    assert (out / "model.safetensors").stat().st_size < (ckpt / "model.safetensors").stat().st_size

    state = quantized.load_quantized_model(out)
    assert quantized.resolve_model_path(state, out, None) == str(out)
    quantized.verify_base_checkpoint(state, out)

    full = quantized.push_to_hub(out, "org/m", dry_run=True)
    assert full["self_contained"]
    assert {f["path"] for f in full["files"]} == {
        "README.md", "foldquant_quant.json", HF_QUANT_CONFIG_NAME, "model.safetensors", "config.json",
        "tokenizer_config.json", "processor_config.json", "experiment_cfg/conf.yaml",
    }

    # the Hub client's own collector sees the same files (README excluded: its check needs the network)
    from huggingface_hub import HfApi

    ops = HfApi()._prepare_upload_folder_additions(
        out, path_in_repo="", allow_patterns=[f["path"] for f in full["files"] if f["path"] != "README.md"]
    )
    assert {o.path_in_repo for o in ops} == {f["path"] for f in full["files"]} - {"README.md"}



def test_a_loader_of_the_checkpoint_misses_only_the_quantized_weights(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    """What a family loader sees: every unquantized tensor loads, the quantized projections' weights
    are missing (to be replaced by fake-quant layers), the codes are the unexpected extras."""
    import copy

    from safetensors.torch import load_file

    from foldquant.quantized_checkpoint import quantized_keys

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, _, _ = tiny_dit
    _, out = _saved_model(tiny_dit, tmp_path)
    fresh = copy.deepcopy(dit)
    result = fresh.load_state_dict(load_file(str(out / "model.safetensors")), strict=False)
    assert set(result.missing_keys) == set(quantized_keys(out))
    assert all(k.endswith((".qweight", ".weight_scale")) or k.startswith("foldquant.") for k in result.unexpected_keys)


def test_the_bf16_arm_is_refused_from_a_quantized_checkpoint(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    from foldquant.quantized import quantized_arm

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    ckpt, out = _saved_model(tiny_dit, tmp_path)
    assert quantized_arm(out, None) == str(out)
    with pytest.raises(SystemExit, match="no bf16 arm"):
        quantized_arm(out, None, no_fakequant=True)
    assert quantized_arm(ckpt, None, no_fakequant=True) is None  # the base checkpoint is the bf16 arm


# The family tools' fake-quant arm


def test_quantized_arm_detects_a_fakequant_model_unless_another_arm_is_asked_for(tmp_path: Path) -> None:
    from foldquant.quantized import MANIFEST_NAME, quantized_arm

    plain = tmp_path / "plain"
    plain.mkdir()
    fq = tmp_path / "fq"
    fq.mkdir()
    (fq / MANIFEST_NAME).write_text('{"format": "foldquant-quantized-checkpoint"}')
    assert quantized_arm(plain, None) is None
    assert quantized_arm(fq, None) == str(fq)
    with pytest.raises(SystemExit, match="no bf16 arm"):  # a quantized checkpoint has no bf16 weights to run
        quantized_arm(fq, None, no_fakequant=True)
    assert quantized_arm(plain, None, no_fakequant=True) is None
    assert quantized_arm(fq, None, other_arms=("engines",)) is None
    assert quantized_arm(fq, None, other_arms=(None, "")) == str(fq)
    assert quantized_arm(plain, "state") == "state"
    assert quantized_arm("org/hub-id", None) is None


_TOOLS = ("eval_libero.py", "serve.py", "verify.py")
_FAMILIES = ("groot_n1_7", "groot_n1_6", "groot_n1_5", "pi05")


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("tool", _TOOLS)
def test_every_family_tool_runs_a_quantized_arm(family: str, tool: str) -> None:
    """Every eval / serve / verify takes --quantized-model and --no-fakequant and detects a fake-quant model."""
    import ast

    path = Path(__file__).resolve().parents[1] / "models" / family / "foldquant_integration" / tool
    tree = ast.parse(path.read_text())
    fields = {
        t.target.id
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name.endswith("Config")
        for t in c.body
        if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
    }
    assert {"quantized_model", "no_fakequant"} <= fields, f"{family}/{tool}: config lacks the fake-quant flags"
    calls = {
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert "quantized_arm" in calls, f"{family}/{tool}: does not detect a fake-quant model"
    if calls & {"install_on_policy", "install_fake_quant", "install_fakequant"}:
        return
    # A tool that runs the policy in a server process must hand the arm to it.
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {"--quantized-model", "--no-fakequant"} <= strings, f"{family}/{tool}: never installs the fake-quant"


def test_re_emitting_the_llm_graph_over_an_old_export_gives_the_same_bytes(tmp_path: Path) -> None:
    """The LLM sidecar is rewritten, not appended to: an earlier export at the same path
    (an interrupted run, a rerun) must not shift the tensor offsets the graph records."""
    import onnx

    from foldquant.onnx_io import save_plugin_onnx

    from onnx import helper as oh, numpy_helper
    import numpy as np

    def model():  # a fresh proto per write: saving moves its tensors out in place
        w = numpy_helper.from_array(np.arange(4096, dtype=np.float32).reshape(64, 64), "w")
        g = oh.make_graph([oh.make_node("MatMul", ["x", "w"], ["y"])], "g",
                          [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 64])],
                          [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 64])], initializer=[w])
        return oh.make_model(g)

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    save_plugin_onnx(model(), tmp_path / "a" / "llm_bf16.onnx", size_threshold=1024)
    save_plugin_onnx(model(), tmp_path / "b" / "llm_bf16.onnx", size_threshold=1024)
    save_plugin_onnx(model(), tmp_path / "b" / "llm_bf16.onnx", size_threshold=1024)
    for name in ("llm_bf16.onnx", "llm_bf16.onnx.data"):
        assert filecmp.cmp(tmp_path / "a" / name, tmp_path / "b" / name, shallow=False), name


def test_the_llm_emitter_writes_through_save_plugin_onnx() -> None:
    import inspect

    from foldquant import llm

    src = inspect.getsource(llm)
    assert "onnx.save(" not in src and "convert_model_to_external_data" not in src



def test_recording_without_a_path_writes_no_graph_and_exports_the_same_bytes(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    """Quantizing builds the graph in memory to record the codes and its digest; no graph is
    written. The export from the quantized checkpoint equals a direct export byte for byte."""
    from foldquant import quantized
    from foldquant.export import export_dit

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, loop, _ = tiny_dit
    res = export_dit(dit, None, scheme=schemes.W4A4_SH, forward_loop=loop, record=True)
    assert res.onnx_path is None and res.graph_digest and res.state is not None
    assert list(tmp_path.rglob("*.onnx")) == []
    ckpt = _base_checkpoint(tmp_path / "ckpt", dit)
    _save(tmp_path / "s", ckpt, dit, res, "dit")
    state = quantized.load_quantized_model(tmp_path / "s")
    assert state.manifest["graph_digest"] == "inline"
    assert state.manifest["graphs"]["dit"] == res.graph_digest
    out = quantized.export_onnx(state, {"dit": _without_quantized_weights(dit, tmp_path / "s")}, tmp_path / "r")
    export_dit(dit, tmp_path / "d" / "dit_bf16.onnx", scheme=schemes.W4A4_SH, forward_loop=loop)
    _same_export(tmp_path / "d" / "dit_bf16.onnx", out / "dit_bf16.onnx")

def test_export_completes_the_directory_through_the_family_adapter(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    """A family whose engine set holds float components exports them in `to-onnx` too, through
    its adapter's complete_onnx hook, into the same directory as the FoldQuant graphs."""
    from types import SimpleNamespace

    from foldquant import quantized
    from foldquant.export import export_dit

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, loop, _ = tiny_dit
    ckpt = _base_checkpoint(tmp_path / "ckpt", dit)
    res = export_dit(dit, None, scheme=schemes.W8A8_SH, forward_loop=loop, record=True)
    _save(tmp_path / "q", ckpt, dit, res, "dit", family="stub")
    calls = []
    stripped = _without_quantized_weights(dit, tmp_path / "q")

    def complete_onnx(policy, onnx_dir, state):
        calls.append((policy, Path(onnx_dir), state))
        (Path(onnx_dir) / "vit_fp32.onnx").write_bytes(b"float component")

    adapter = SimpleNamespace(
        FAMILY="stub",
        load_policy=lambda model_path, tag, device, **kw: stripped,
        module_paths=lambda policy: {"dit": policy},
        checkpoint_root=lambda policy: policy,
        complete_onnx=complete_onnx,
    )
    monkeypatch.setattr(quantized, "load_adapter", lambda family=None: adapter)
    out = quantized.to_onnx(quantized.ToOnnx(quantized_model=str(tmp_path / "q"), output_dir=str(tmp_path / "o"), device="cpu"))
    assert calls and calls[0][0] is stripped and calls[0][1] == out
    assert (out / "dit_bf16.onnx").is_file() and (out / "vit_fp32.onnx").is_file()


def test_an_old_state_directory_no_longer_loads(tmp_path: Path) -> None:
    from foldquant import quantized

    (tmp_path / "foldquant_quant.json").write_text('{"format": "foldquant-quant-state", "modules": {}}')
    (tmp_path / "quant_state.safetensors").write_bytes(b"")
    assert not quantized.is_quantized_model(tmp_path)
    with pytest.raises(ValueError, match="quantized checkpoint"):
        quantized.load_quantized_model(tmp_path)


def test_the_float_baseline_trace_matches_the_direct_module_call(tiny_dit, tmp_path: Path) -> None:
    """`export` on an unquantized checkpoint traces each tower on one captured call, under the same
    bindings as the quantized graph and with no plugin nodes."""
    import onnx

    from foldquant.trace_export import Binding, trace_module

    dit, loop, inputs = tiny_dit
    ones = lambda kw: torch.ones(kw["encoder_hidden_states"].shape[:2], dtype=torch.bool)  # noqa: E731
    path = trace_module(
        dit, tmp_path / "dit_bf16.onnx", module_name="dit",
        bindings=[
            Binding("sa_embs", "hidden_states", torch.float32),  # the tiny DiT is fp32; the families bind bf16
            Binding("vl_embs", "encoder_hidden_states", torch.float32, {1: "vl_seq_len"}),
            Binding("timestep", "timestep", torch.int64),
            Binding("image_mask", "image_mask", torch.bool, {1: "vl_seq_len"}, default=ones),
            Binding("backbone_attention_mask", "backbone_attention_mask", torch.bool, {1: "vl_seq_len"}, default=ones),
        ],
        output_name="output", forward_loop=loop, extract=lambda o: o[0] if isinstance(o, (tuple, list)) else o,
    )
    model = onnx.load(str(path))
    assert [i.name for i in model.graph.input] == ["sa_embs", "vl_embs", "timestep", "image_mask", "backbone_attention_mask"]
    assert [o.name for o in model.graph.output] == ["output"]
    assert not any(n.domain == "trt.plugins" for n in model.graph.node), "a float graph carries no plugin nodes"
