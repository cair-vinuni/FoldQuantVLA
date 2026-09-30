# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The quantized checkpoint round-trips every module kind and scheme, and the graphs
export from it byte-identically with the bf16 weights of the quantized projections gone."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from foldquant import schemes
from test_expert_fakequant import _TinyExpert, _fake_capture  # noqa: E402
from test_quantized import _base_checkpoint, _same_export, tiny_dit, tiny_qwen3  # noqa: F401,E402


def _roundtrip(module, res, tmp_path: Path, name: str):
    from foldquant import quantized
    from foldquant.quantized_checkpoint import quantized_keys

    ckpt = _base_checkpoint(tmp_path / "ckpt", module)
    out = tmp_path / "q"
    quantized.save_arm_state(out, family="t", model_path=str(ckpt), results=[res], modules={name: module},
                             checkpoint_root=module, export_manifest={"files": {name: f"{name}_bf16.onnx"}})
    state = quantized.load_quantized_model(out)
    ms, rec = state.modules[name], res.state
    assert set(ms.gptq) == set(rec.gptq)
    for key, entries in rec.gptq.items():
        assert len(ms.gptq[key]) == len(entries)
        for (codes, scale), (c2, s2) in zip(entries, ms.gptq[key]):
            assert torch.equal(codes.to(torch.int8), c2.to(torch.int8)) and torch.equal(scale.float(), s2.float()), key
    assert set(ms.tensors) == set(rec.tensors)
    for key in rec.tensors:
        assert torch.equal(torch.as_tensor(rec.tensors[key]).float().cpu(), ms.tensors[key].float()), key
    stripped = copy.deepcopy(module)
    with torch.no_grad():
        for key in quantized_keys(out):
            stripped.get_parameter(key).zero_()
    return state, stripped, out


@pytest.mark.parametrize("scheme", [schemes.W4A4_SR, schemes.W4A4_SH, schemes.W4A4_SHG, schemes.W8A8_SH, schemes.W8A8])
def test_dit_checkpoint_round_trips_and_exports_the_same_graph(tiny_dit, tmp_path: Path, scheme: str, monkeypatch) -> None:
    from foldquant import quantized
    from foldquant.export import export_dit

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, loop, _ = tiny_dit
    res = export_dit(dit, tmp_path / "direct" / "dit_bf16.onnx", scheme=scheme, forward_loop=loop, record=True)
    state, stripped, out = _roundtrip(dit, res, tmp_path, "dit")
    if scheme == schemes.W4A4_SR:
        assert state.modules["dit"].tensor_group("rot"), "a dense rotation is recorded, not recomputed"
    replayed = quantized.export_onnx(state, {"dit": stripped}, tmp_path / "replayed")
    _same_export(tmp_path / "direct" / "dit_bf16.onnx", replayed / "dit_bf16.onnx")


@pytest.mark.parametrize("scheme", [schemes.W4A4_SRG, schemes.W8A8_SR])
def test_llm_checkpoint_round_trips_and_exports_the_same_graph(tiny_qwen3, tmp_path: Path, scheme: str, monkeypatch) -> None:
    from foldquant import quantized
    from foldquant.export import export_llm

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    model, loop, _ = tiny_qwen3
    params = {"site_bits": {"o": 8, "down": 8}} if scheme == schemes.W4A4_SRG else None
    (tmp_path / "direct").mkdir()
    res = export_llm(model, tmp_path / "direct" / "llm_bf16.onnx", scheme=scheme, forward_loop=loop, record=True,
                     max_seq_len=64, params=params)
    state, stripped, out = _roundtrip(model, res, tmp_path, "llm")
    replayed = quantized.export_onnx(state, {"llm": stripped}, tmp_path / "replayed")
    _same_export(tmp_path / "direct" / "llm_bf16.onnx", replayed / "llm_bf16.onnx")


@pytest.mark.parametrize("scheme", [schemes.W4A4_SR, schemes.W4A4_SHG, schemes.W8A8_SH])
def test_expert_checkpoint_round_trips_and_exports_the_same_graph(tmp_path: Path, scheme: str, monkeypatch) -> None:
    from foldquant import quantized, gemma_expert
    from foldquant.export import export_expert

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(gemma_expert, "compute_gemma_expert_sq_scales", _fake_capture)
    module = _TinyExpert()
    (tmp_path / "direct").mkdir()
    res = export_expert(module, tmp_path / "direct" / "expert_bf16.onnx", scheme=scheme, forward_loop=lambda m: None, record=True)
    state, stripped, out = _roundtrip(module, res, tmp_path, "expert")
    replayed = quantized.export_onnx(state, {"expert": stripped}, tmp_path / "replayed")
    _same_export(tmp_path / "direct" / "expert_bf16.onnx", replayed / "expert_bf16.onnx")


def test_fake_quant_from_the_checkpoint_matches_fake_quant_from_the_weights(tiny_dit, tmp_path: Path, monkeypatch) -> None:
    """The fake-quant layers installed on a loader's view of the checkpoint (no bf16 weights in
    the quantized projections) compute what they compute on the original module."""
    from foldquant.dit_fake_quant import install_dit_fake_quant
    from foldquant.export import export_dit

    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    dit, loop, inputs = tiny_dit
    res = export_dit(dit, None, scheme=schemes.W4A4_SHG, forward_loop=loop, record=True)
    state, stripped, _ = _roundtrip(dit, res, tmp_path, "dit")
    kw = inputs[0]
    with torch.no_grad():
        h1 = install_dit_fake_quant(copy.deepcopy(dit), res.state)
        ref_module = copy.deepcopy(dit)
        h_ref = install_dit_fake_quant(ref_module, res.state)
        ref = ref_module(**kw)
        h_new = install_dit_fake_quant(stripped, state.modules["dit"])
        got = stripped(**kw)
    ref = ref[0] if isinstance(ref, tuple) else ref
    got = got[0] if isinstance(got, tuple) else got
    assert torch.equal(got, ref)
    h1.remove(); h_ref.remove(); h_new.remove()
