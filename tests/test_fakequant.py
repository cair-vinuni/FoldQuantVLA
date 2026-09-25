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
from foldquant.quant_state import (
    ModuleQuantState,
    QuantState,
    ReplayError,
    ReplaySite,
    load_state,
    record_gptq,
    replay_sites,
    save_state,
)

_N17 = Path(__file__).resolve().parents[1] / "models" / "groot_n1_7"


def _hessian(k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4 * k, k, generator=g)
    return (x.T @ x).double()


# Quant state and GPTQ record / replay


def test_state_round_trips_scales_codes_and_config(tmp_path: Path) -> None:
    g = torch.Generator().manual_seed(1)
    ms = ModuleQuantState("dit", schemes.W4A4_SHG, config={"params": {"sq_alpha": 0.5}, "x": [1, 2]})
    ms.put_group("sq", {"block0_qkv": torch.rand(64, generator=g) + 0.5})
    c4 = torch.randint(-7, 8, (6, 64), generator=g, dtype=torch.int8)
    c8 = torch.randint(-127, 128, (5, 33), generator=g, dtype=torch.int8)
    ms.gptq = {"dit.block0_qkv": [(c4, torch.rand(6)), (c4.flip(0), torch.rand(6))], "dit.x": [(c8, torch.rand(5))]}
    save_state(QuantState(manifest={"family": "t"}, modules={"dit": ms}), tmp_path)
    back = load_state(tmp_path)
    got = back.modules["dit"]
    assert back.manifest["family"] == "t" and got.scheme == schemes.W4A4_SHG and got.config == ms.config
    assert torch.equal(got.tensor_group("sq")["block0_qkv"], ms.tensor_group("sq")["block0_qkv"])
    for site, entries in ms.gptq.items():
        for (a, sa), (b, sb) in zip(entries, got.gptq[site]):
            assert torch.equal(a, b) and torch.equal(sa, sb)


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
    "scheme", [schemes.W4A4_SHG, schemes.W4A4_SH, schemes.W4A4_SR, schemes.W8A8_SH, schemes.W8A8]
)
def test_a_dit_graph_replays_byte_identical_from_its_saved_state(tiny_dit, tmp_path: Path, scheme: str) -> None:
    from foldquant.export import export_dit

    dit, loop, _ = tiny_dit
    first = export_dit(dit, tmp_path / "a" / "dit_bf16.onnx", scheme=scheme, forward_loop=loop, record=True)
    save_state(QuantState(modules={"dit": first.state}), tmp_path / "state")
    state = load_state(tmp_path / "state")
    export_dit(dit, tmp_path / "b" / "dit_bf16.onnx", scheme=scheme, state=state.modules["dit"])
    _same_export(tmp_path / "a" / "dit_bf16.onnx", tmp_path / "b" / "dit_bf16.onnx")


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
def test_an_llm_graph_replays_byte_identical_from_its_saved_state(tiny_qwen3, tmp_path: Path, scheme: str) -> None:
    from foldquant.export import export_llm

    model, loop, _ = tiny_qwen3
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
    first = export_llm(model, tmp_path / "a" / "llm_bf16.onnx", scheme=scheme, forward_loop=loop, record=True, max_seq_len=64)
    save_state(QuantState(modules={"llm": first.state}), tmp_path / "state")
    state = load_state(tmp_path / "state")
    export_llm(model, tmp_path / "b" / "llm_bf16.onnx", scheme=scheme, state=state.modules["llm"], max_seq_len=64)
    _same_export(tmp_path / "a" / "llm_bf16.onnx", tmp_path / "b" / "llm_bf16.onnx")


@pytest.mark.parametrize("scheme", [schemes.W4A4_SRG, schemes.W8A8_SR])
def test_the_llm_fake_quant_uses_the_engine_codes(tiny_qwen3, tmp_path: Path, scheme: str) -> None:
    """Every projection runs on the codes and scales the graph packs, and tracks the float model."""
    from foldquant.export import export_llm
    from foldquant.fake_quant_linear import FakeQuantLinear
    from foldquant.fakequant import install_llm_fake_quant

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


# Checkpoint guard and Hub upload (dry run)


def test_a_downloaded_copy_passes_and_changed_weights_do_not(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    """Extra files (a README, Hub metadata) are ignored; a changed or missing weight file is refused."""
    import os

    from foldquant import fakequant
    from foldquant.export import export_dit

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, loop, _ = tiny_dit
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for name, data in (("model-00001.safetensors", b"A" * 64), ("model-00002.safetensors", b"B" * 64), ("config.json", b"{}")):
        (ckpt / name).write_bytes(data)
    res = export_dit(dit, tmp_path / "d" / "dit_bf16.onnx", scheme=schemes.W8A8, forward_loop=loop, record=True)
    fakequant.save_arm_state(tmp_path / "s", family="t", model_path=str(ckpt), results=[res], export_manifest={})
    state = load_state(tmp_path / "s")
    assert set(state.manifest["base"]["file_hashes"]) == {"model-00001.safetensors", "model-00002.safetensors", "config.json"}

    copy = tmp_path / "copy"
    copy.mkdir()
    for f in ckpt.iterdir():
        os.link(f, copy / f.name)
    (copy / "README.md").write_text("model card")
    (copy / "LICENSE").write_text("licence")
    (copy / ".cache").mkdir()
    (copy / ".cache" / "x").write_text("etag")
    fakequant.verify_base_checkpoint(state, copy)

    (copy / "model-00002.safetensors").unlink()
    (copy / "model-00002.safetensors").write_bytes(b"C" * 64)
    with pytest.raises(ValueError, match="different content"):
        fakequant.verify_base_checkpoint(state, copy)
    (copy / "model-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="missing"):
        fakequant.verify_base_checkpoint(state, copy)


def test_a_state_refuses_a_checkpoint_it_was_not_calibrated_on(tmp_path: Path, monkeypatch) -> None:
    from foldquant.eval_protocol import artifact_digest
    from foldquant.fakequant import verify_base_checkpoint

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"A" * 64)
    state = QuantState(manifest={"base": {"digest": artifact_digest(ckpt)["digest"]}})
    verify_base_checkpoint(state, ckpt)
    (ckpt / "model.safetensors").write_bytes(b"B" * 64)
    with pytest.raises(ValueError, match="not the checkpoint"):
        verify_base_checkpoint(state, ckpt)


def test_push_dry_run_lists_the_state_without_contacting_the_hub(tmp_path: Path) -> None:
    from foldquant.fakequant import push_to_hub, write_model_card

    ms = ModuleQuantState("dit", schemes.W8A8_SH, config={"params": {}})
    ms.put_group("sq", {"encoder": torch.ones(8)})
    state = QuantState(manifest={"family": "groot_n1_7", "base": {"model_id": "org/base", "digest": "0" * 64}}, modules={"dit": ms})
    save_state(state, tmp_path)
    card = write_model_card(tmp_path, state, repo_id="org/fq")
    assert "base_model: org/base" in card.read_text()
    report = push_to_hub(tmp_path, "org/fq", dry_run=True)
    assert report["private"] and sorted(f["path"] for f in report["files"]) == [
        "README.md", "foldquant_quant.json", "quant_state.safetensors"
    ]
    with pytest.raises(FileNotFoundError):
        push_to_hub(tmp_path / "missing", "org/fq", dry_run=True)


def test_the_pipeline_saves_a_state_and_rebuilds_the_same_graph(tiny_dit, tmp_path: Path, monkeypatch, capsys) -> None:
    """save_arm_state -> push dry run -> export_onnx, through foldquant.fakequant alone."""
    import json

    from foldquant import fakequant
    from foldquant.export import export_dit

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, loop, _ = tiny_dit
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"weights")
    direct = tmp_path / "direct" / "onnx"
    res = export_dit(dit, direct / "dit_bf16.onnx", scheme=schemes.W4A4_SHG, forward_loop=loop, record=True)
    manifest = {"files": {"dit": "dit_bf16.onnx"}, "calibration": {"seed": 0, "num_samples": 3}, "plugin_libs": []}
    fakequant.save_arm_state(
        tmp_path / "state", family="test_family", model_path=str(ckpt), results=[res],
        export_manifest=manifest, extra_files={"export_metadata.json": {"sa_seq_len": 9}},
    )
    fakequant.main(["info", "--fakequant-dir", str(tmp_path / "state")])
    assert json.loads(capsys.readouterr().out)["modules"]["dit"]["scheme"] == schemes.W4A4_SHG

    state = load_state(tmp_path / "state")
    fakequant.verify_base_checkpoint(state, ckpt)
    out = fakequant.export_onnx(state, {"dit": dit}, tmp_path / "replayed", source=tmp_path / "state")
    _same_export(direct / "dit_bf16.onnx", out / "dit_bf16.onnx")
    written = json.loads((out / fakequant.EXPORT_MANIFEST_NAME).read_text())
    assert written["family"] == "test_family" and written["plugin_libs"] == res.plugin_libs
    assert json.loads((out / "export_metadata.json").read_text()) == {"sa_seq_len": 9}


def test_the_pipeline_rebuilds_an_llm_graph_too(tiny_qwen3, tmp_path: Path, monkeypatch) -> None:
    from foldquant import fakequant
    from foldquant.export import export_llm

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    model, loop, _ = tiny_qwen3
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"weights")
    direct = tmp_path / "direct"
    direct.mkdir()
    res = export_llm(model, direct / "llm_bf16.onnx", scheme=schemes.W4A4_SRG, forward_loop=loop, record=True, max_seq_len=64)
    fakequant.save_arm_state(
        tmp_path / "state", family="t", model_path=str(ckpt), results=[res],
        export_manifest={"files": {"llm": "llm_bf16.onnx"}},
    )
    out = fakequant.export_onnx(load_state(tmp_path / "state"), {"llm": model}, tmp_path / "replayed")
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
    from foldquant import fakequant
    from foldquant.export import export_dit

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, loop, _ = tiny_dit
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "w").write_bytes(b"w")
    res = export_dit(dit, tmp_path / "d" / "dit_bf16.onnx", scheme=schemes.W8A8_SH, forward_loop=loop, record=True)
    fakequant.save_arm_state(tmp_path / "s", family="t", model_path=str(ckpt), results=[res],
                             export_manifest={"files": {"dit": "dit_bf16.onnx"}})
    state = load_state(tmp_path / "s")
    assert state.manifest["graphs"]["dit"]
    assert state.manifest["graph_digest"] == "content"
    state.manifest["graphs"]["dit"] = "0" * 64
    with pytest.raises(fakequant.GraphMismatchError):
        fakequant.export_onnx(state, {"dit": dit}, tmp_path / "r")


def test_push_uploads_only_the_state_files(tmp_path: Path) -> None:
    from foldquant.fakequant import push_to_hub

    save_state(QuantState(modules={"dit": ModuleQuantState("dit", schemes.W8A8)}), tmp_path)
    (tmp_path / "scratch.log").write_text("x")
    names = [f["path"] for f in push_to_hub(tmp_path, "org/x", dry_run=True)["files"]]
    assert "scratch.log" not in names and "quant_state.safetensors" in names


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


def test_the_graph_digest_ignores_the_header_but_not_the_weights(tiny_dit, tmp_path: Path) -> None:
    import onnx

    from foldquant.export import export_dit
    from foldquant.fakequant import _graph_digest

    dit, loop, _ = tiny_dit
    path = tmp_path / "dit_bf16.onnx"
    export_dit(dit, path, scheme=schemes.W8A8, forward_loop=loop)
    before = _graph_digest(path)
    m = onnx.load(str(path), load_external_data=False)
    m.ir_version, m.producer_name, m.producer_version = 9, "another-onnx", "0.0"
    onnx.save_model(m, str(path))
    assert _graph_digest(path) == before
    node = next(n for n in m.graph.node if n.domain == "trt.plugins" and any(a.name.startswith("weight_") for a in n.attribute))
    attr = next(a for a in node.attribute if a.name.startswith("weight_") and a.s)
    attr.s = bytes([attr.s[0] ^ 1]) + attr.s[1:]
    onnx.save_model(m, str(path))
    assert _graph_digest(path) != before


# Self-contained fake-quant models


def _saved_model(tiny_dit, tmp_path: Path, **kw):
    from foldquant import fakequant
    from foldquant.export import export_dit

    dit, loop, _ = tiny_dit
    ckpt = tmp_path / "ckpt"
    (ckpt / "experiment_cfg").mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"W" * 256)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "experiment_cfg" / "conf.yaml").write_text("a: 1")
    (ckpt / "README.md").write_text("upstream card")
    res = export_dit(dit, tmp_path / "d" / "dit_bf16.onnx", scheme=schemes.W8A8_SH, forward_loop=loop, record=True)
    out = tmp_path / "fq"
    fakequant.save_arm_state(out, family="t", model_path=str(ckpt), results=[res],
                             export_manifest={"files": {"dit": "dit_bf16.onnx"}}, **kw)
    return ckpt, out


def test_a_saved_fake_quant_model_is_self_contained(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    import os

    from foldquant import fakequant

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    ckpt, out = _saved_model(tiny_dit, tmp_path)
    assert fakequant.is_fakequant_model(out)
    for rel in ("model.safetensors", "config.json", "experiment_cfg/conf.yaml"):
        assert (out / rel).is_symlink() and (out / rel).read_bytes() == (ckpt / rel).read_bytes()
    assert not (out / "experiment_cfg").is_symlink()  # a real directory: the Hub client walks it
    assert "self-contained" in (out / "README.md").read_text()
    state = load_state(out)
    assert state.manifest["base"]["bundled"] == "link"
    assert fakequant.resolve_model_path(state, out, None) == str(out)
    fakequant.verify_base_checkpoint(state, out)

    full = fakequant.push_to_hub(out, "org/m", dry_run=True)
    assert full["self_contained"]
    assert {f["path"] for f in full["files"]} == {
        "README.md", "foldquant_quant.json", "quant_state.safetensors",
        "model.safetensors", "config.json", "experiment_cfg/conf.yaml",
    }
    assert next(f for f in full["files"] if f["path"] == "model.safetensors")["bytes"] == 256
    only = fakequant.push_to_hub(out, "org/m", dry_run=True, state_only=True)
    assert {f["path"] for f in only["files"]} == {"README.md", "foldquant_quant.json", "quant_state.safetensors"}

    # the Hub client's own collector reads through the file links (README excluded: its check needs the network)
    from huggingface_hub import HfApi

    ops = HfApi()._prepare_upload_folder_additions(
        out, path_in_repo="", allow_patterns=[f["path"] for f in full["files"] if f["path"] != "README.md"]
    )
    assert {o.path_in_repo for o in ops} == {f["path"] for f in full["files"]} - {"README.md"}
    assert next(o for o in ops if o.path_in_repo == "model.safetensors").upload_info.size == 256
    os.unlink(out / "config.json")
    with pytest.raises(FileNotFoundError, match="bundled base files missing"):
        fakequant.push_to_hub(out, "org/m", dry_run=True)


def test_a_state_only_model_needs_its_base_and_can_be_bundled_later(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    from foldquant import fakequant

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    ckpt, out = _saved_model(tiny_dit, tmp_path, bundle=False)
    state = load_state(out)
    assert not (out / "model.safetensors").exists()
    with pytest.raises(SystemExit, match="--model-path"):
        fakequant.resolve_model_path(state, out, None)
    fakequant.main(["bundle", "--fakequant-dir", str(out), "--model-path", str(ckpt), "--copy"])
    assert (out / "model.safetensors").is_file() and not (out / "model.safetensors").is_symlink()
    assert load_state(out).manifest["base"]["bundled"] == "copy"
    fakequant.verify_base_checkpoint(load_state(out), out)


# The family tools' fake-quant arm


def test_fakequant_arm_detects_a_fakequant_model_unless_another_arm_is_asked_for(tmp_path: Path) -> None:
    from foldquant.fakequant import MANIFEST_NAME, fakequant_arm

    plain = tmp_path / "plain"
    plain.mkdir()
    fq = tmp_path / "fq"
    fq.mkdir()
    (fq / MANIFEST_NAME).write_text("{}")
    assert fakequant_arm(plain, None) is None
    assert fakequant_arm(fq, None) == str(fq)
    assert fakequant_arm(fq, None, no_fakequant=True) is None
    assert fakequant_arm(fq, None, other_arms=("engines",)) is None
    assert fakequant_arm(fq, None, other_arms=(None, "")) == str(fq)
    assert fakequant_arm(plain, "state") == "state"
    assert fakequant_arm("org/hub-id", None) is None


_TOOLS = ("eval_libero.py", "serve.py", "verify.py")
_FAMILIES = ("groot_n1_7", "groot_n1_6", "groot_n1_5", "pi05")


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("tool", _TOOLS)
def test_every_family_tool_runs_a_fakequant_arm(family: str, tool: str) -> None:
    """Every eval / serve / verify takes --fakequant-dir and --no-fakequant and detects a fake-quant model."""
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
    assert {"fakequant_dir", "no_fakequant"} <= fields, f"{family}/{tool}: config lacks the fake-quant flags"
    calls = {
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert "fakequant_arm" in calls, f"{family}/{tool}: does not detect a fake-quant model"
    if calls & {"install_on_policy", "install_fake_quant", "install_fakequant"}:
        return
    # A tool that runs the policy in a server process must hand the arm to it.
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {"--fakequant-dir", "--no-fakequant"} <= strings, f"{family}/{tool}: never installs the fake-quant"


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
