# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The Pi action expert's fake-quant: recorded state, byte-identical replay, and a
PyTorch reader that tracks the float expert.

The expert is a tiny vanilla-Gemma one (``use_adarms=False``) built from
``transformers``; the fake-quant touches only the decoder's projections, which
are the same modules under AdaRMS. Calibration is replaced by fixed SmoothQuant
vectors and Hessians, so every scheme exports without a forward loop.
"""

from __future__ import annotations

import filecmp
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")
pytest.importorskip("onnx")

from foldquant import export as fq_export  # noqa: E402
from foldquant import gemma_expert, schemes  # noqa: E402
from foldquant.expert_fake_quant import install_expert_fake_quant  # noqa: E402
from foldquant.fake_quant_linear import FakeQuantLinear  # noqa: E402
from foldquant.quant_state import QuantState, load_state, save_state  # noqa: E402

HIDDEN, FF, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 64, 128, 2, 2, 1, 32


class _TinyExpert(nn.Module):
    """The emitter's layout: ``expert_model`` (Gemma decoder) plus the Pi0 suffix projections."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        cfg = transformers.GemmaConfig(
            vocab_size=16,
            hidden_size=HIDDEN,
            intermediate_size=FF,
            num_hidden_layers=LAYERS,
            num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            max_position_embeddings=64,
        )
        self.expert_model = transformers.GemmaForCausalLM(cfg).eval()
        self.action_in_proj = nn.Linear(8, HIDDEN)
        self.action_out_proj = nn.Linear(HIDDEN, 8)
        self.action_time_mlp_in = nn.Linear(2 * HIDDEN, HIDDEN)
        self.action_time_mlp_out = nn.Linear(HIDDEN, HIDDEN)
        self.state_proj = nn.Linear(8, HIDDEN)
        self.config = SimpleNamespace(use_adarms=False, action_horizon=4, action_dim=8)
        self._variant = SimpleNamespace(
            width=HIDDEN, depth=LAYERS, num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM, mlp_dim=FF
        )
        # A few outlier input channels, the thing SmoothQuant exists to move.
        with torch.no_grad():
            for layer in self.expert_model.model.layers:
                for lin in (layer.self_attn.q_proj, layer.mlp.gate_proj, layer.mlp.down_proj):
                    lin.weight.mul_(1.0 + 3.0 * (torch.rand(lin.in_features) > 0.9).float())


def _site_ks() -> Dict[str, int]:
    out = {}
    for i in range(LAYERS):
        out.update({f"G{i}_qkv": HIDDEN, f"G{i}_o": HEADS * HEAD_DIM, f"G{i}_gu": HIDDEN, f"G{i}_dn": FF})
    return out


def _fake_capture(module, forward_loop, *, gptq_scales=None, **_kw):
    """Fixed SmoothQuant vectors, or (second pass) fixed per-site Hessians."""
    g = torch.Generator().manual_seed(1)
    out = {}
    for site, k in _site_ks().items():
        if gptq_scales is None:
            out[site] = torch.rand(k, generator=g) * 2.0 + 0.25
        else:
            x = torch.randn(4 * k, k, generator=g)
            out[site] = (x.T @ x).double()
    return out


_CASES = [
    (schemes.W8A8, {}),
    (schemes.W8A8_SH, {}),
    (schemes.W8A8_SH, {"sq_fold_order": "after"}),
    (schemes.W4A4_SR, {}),
    (schemes.W4A4_SH, {}),
    (schemes.W4A4_SH, {"sq_fold_order": "after"}),
    (schemes.W4A4_SHG, {}),
]


def _ids(case) -> str:
    scheme, params = case
    return scheme + ("_" + "_".join(f"{v}" for v in params.values()) if params else "")


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr(gemma_expert, "compute_gemma_expert_sq_scales", _fake_capture)


def _record(module, scheme, params, path):
    return fq_export.export_expert(
        module, path, scheme=scheme, forward_loop=lambda m: None, params=params, record=True
    ).state


@pytest.mark.parametrize("case", _CASES, ids=_ids)
def test_expert_state_replays_to_the_same_graph(case, capture, tmp_path: Path) -> None:
    scheme, params = case
    module = _TinyExpert()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ms = _record(module, scheme, params, tmp_path / "a" / "expert.onnx")
    save_state(QuantState({"family": "test"}, {"expert": ms}), tmp_path / "state")
    loaded = load_state(tmp_path / "state").modules["expert"]
    fq_export.export_expert(module, tmp_path / "b" / "expert.onnx", scheme=scheme, state=loaded)
    for name in ("expert.onnx", "expert.onnx.data"):
        assert filecmp.cmp(tmp_path / "a" / name, tmp_path / "b" / name, shallow=False), name


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))


@pytest.mark.parametrize("case", _CASES, ids=_ids)
def test_expert_fake_quant_tracks_the_float_expert(case, capture, tmp_path: Path) -> None:
    scheme, params = case
    module = _TinyExpert()
    ms = load_state(save_state(QuantState({"family": "test"}, {"expert": _record(module, scheme, params, tmp_path / "a.onnx")}), tmp_path / "s")).modules["expert"]
    int4 = scheme in schemes.ACT_W4A4_SCHEMES
    decoder = module.expert_model.model
    x = torch.randn(1, 6, HIDDEN, generator=torch.Generator().manual_seed(2))
    originals = {n: m for n, m in decoder.named_modules() if isinstance(m, nn.Linear)}
    with torch.no_grad():
        ref = decoder(inputs_embeds=x).last_hidden_state
        handle = install_expert_fake_quant(module, ms)
        swapped = {n: m for n, m in decoder.named_modules() if isinstance(m, FakeQuantLinear)}
        assert len(swapped) == 7 * LAYERS and len(handle) == 7 * LAYERS
        assert all(m.weight.dtype == torch.float32 for m in swapped.values())

        # Weight side alone (activation left in float): the codes, scales and the
        # activation transform the node declares must reproduce W x almost exactly.
        for name, fq in swapped.items():
            xi = torch.randn(5, fq.in_features, generator=torch.Generator().manual_seed(3))
            a_qmax, fq.a_qmax = fq.a_qmax, None
            got = fq(xi)
            fq.a_qmax = a_qmax
            # 4-bit weight error under the fixed test scales sits at 0.97-0.99; a transform
            # on the wrong side, or a transposed rotation, scores 0.81 or lower.
            assert _cos(got, originals[name](xi)) > (0.95 if int4 else 0.999), name

        out = decoder(inputs_embeds=x).last_hidden_state
        handle.remove()
        back = decoder(inputs_embeds=x).last_hidden_state
    assert _cos(out, ref) > (0.9 if int4 else 0.995)
    assert torch.equal(back, ref)
    assert not any(isinstance(m, FakeQuantLinear) for m in decoder.modules())


def test_expert_fake_quant_reads_the_scale_on_the_side_the_node_names(capture, tmp_path: Path) -> None:
    module = _TinyExpert()
    before = _record(module, schemes.W4A4_SH, {}, tmp_path / "a.onnx")
    after = _record(module, schemes.W4A4_SH, {"sq_fold_order": "after"}, tmp_path / "b.onnx")
    h = install_expert_fake_quant(module, before)
    q = module.expert_model.model.layers[0].self_attn.q_proj
    assert q.s_pre is not None and q.s_post is None
    h.remove()
    h = install_expert_fake_quant(module, after)
    q = module.expert_model.model.layers[0].self_attn.q_proj
    assert q.s_pre is None and q.s_post is not None
    h.remove()


def test_install_fake_quant_routes_the_expert(capture, tmp_path: Path) -> None:
    from foldquant.fakequant import install_fake_quant

    module = _TinyExpert()
    ms = _record(module, schemes.W8A8_SH, {}, tmp_path / "a.onnx")
    handle = install_fake_quant({"expert": module}, QuantState({"family": "test"}, {"expert": ms}))
    assert isinstance(module.expert_model.model.layers[1].mlp.down_proj, FakeQuantLinear)
    handle.remove()
    assert isinstance(module.expert_model.model.layers[1].mlp.down_proj, nn.Linear)
