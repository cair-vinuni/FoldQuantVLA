# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Unit tests for the INT4 groupwise-GEMM ONNX graph surgery (issue #48).

All graphs here are synthetic, built directly with ``onnx``/``onnx_graphsurgeon``
in-memory — no GPU, no TensorRT, and no real ModelOpt export are needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from foldquant.int4_groupwise import (
    _fix_strongly_typed_mismatches,
    apply_int4_modelopt_surgery,
    rewrite_int4_modelopt_dq,
)


def _build_synthetic_modelopt_int4_graph(
    n: int = 128, k: int = 64, block_size: int = 64
) -> tuple[gs.Graph, np.ndarray, np.ndarray]:
    """A minimal graph shaped like ModelOpt's ``modelopt_w4a16_awq`` weight-only export:

    ``trt::DequantizeLinear(Wb, scale) -> Reshape([N, K]) -> Transpose -> MatMul(act, .)``

    ``Wb`` is built as ``q_true * scale`` with ``scale == 1`` so the recovered
    ``q = round(Wb / scale)`` equals ``q_true`` exactly (no floating-point
    rounding to account for in assertions).
    """
    nblk = k // block_size
    rng = np.random.default_rng(0)
    q_true = rng.integers(-8, 8, size=(n * nblk, block_size)).astype(np.float32)
    scale = np.ones((n * nblk, 1), dtype=np.float32)
    wb = q_true * scale

    act = gs.Variable("activation", dtype=np.float32, shape=(1, k))
    wb_const = gs.Constant("Wb", wb)
    sb_const = gs.Constant("scale_b", scale)
    dq_out = gs.Variable("dq_out")
    dq_node = gs.Node(
        op="DequantizeLinear",
        name="dq0",
        domain="trt",
        inputs=[wb_const, sb_const],
        outputs=[dq_out],
        attrs={"block_size": block_size},
    )

    reshape_shape = gs.Constant("reshape_shape", np.array([n, k], dtype=np.int64))
    reshape_out = gs.Variable("reshape_out")
    reshape_node = gs.Node(op="Reshape", name="reshape0", inputs=[dq_out, reshape_shape], outputs=[reshape_out])

    transpose_out = gs.Variable("transpose_out")
    transpose_node = gs.Node(
        op="Transpose", name="transpose0", inputs=[reshape_out], outputs=[transpose_out], attrs={"perm": [1, 0]}
    )

    # dtype is float32 here purely so the pre-surgery graph is itself a valid
    # ONNX export target for the end-to-end apply_int4_modelopt_surgery() test;
    # the surgery repoints this same Variable's dtype to bfloat16 either way.
    mm_out = gs.Variable("mm_out", dtype=np.float32, shape=(1, n))
    mm_node = gs.Node(op="MatMul", name="matmul0", inputs=[act, transpose_out], outputs=[mm_out])

    graph = gs.Graph(nodes=[dq_node, reshape_node, transpose_node, mm_node], inputs=[act], outputs=[mm_out])
    return graph, q_true, scale


# --------------------------------------------------------------------------- rewrite_int4_modelopt_dq()
def test_rewrite_replaces_dq_chain_with_plugin_node() -> None:
    graph, _, _ = _build_synthetic_modelopt_int4_graph()
    graph, replaced, materialized = rewrite_int4_modelopt_dq(graph)

    assert replaced == 1
    assert materialized == 0
    op_types = [n.op for n in graph.nodes]
    assert op_types == ["Int4GroupwiseGemmPlugin"]


def test_rewrite_plugin_node_attrs_match_gemm_shape() -> None:
    graph, _, _ = _build_synthetic_modelopt_int4_graph(n=128, k=64, block_size=64)
    graph, _, _ = rewrite_int4_modelopt_dq(graph, plugin_namespace="gr00t::v1")

    plugin_node = graph.nodes[0]
    assert plugin_node.op == "Int4GroupwiseGemmPlugin"
    assert plugin_node.attrs["gemm_n"] == 128
    assert plugin_node.attrs["gemm_k"] == 64
    assert plugin_node.attrs["group_size"] == 64
    assert plugin_node.attrs["plugin_namespace"] == "gr00t::v1"


def test_rewrite_plugin_node_preserves_activation_and_graph_output() -> None:
    graph, _, _ = _build_synthetic_modelopt_int4_graph()
    graph, _, _ = rewrite_int4_modelopt_dq(graph)

    plugin_node = graph.nodes[0]
    assert plugin_node.inputs[0].name == "activation"
    assert graph.outputs[0] is plugin_node.outputs[0]


def test_rewrite_packed_weight_and_scale_shapes() -> None:
    n, k, block_size = 128, 64, 64
    graph, _, _ = _build_synthetic_modelopt_int4_graph(n=n, k=k, block_size=block_size)
    graph, _, _ = rewrite_int4_modelopt_dq(graph)

    plugin_node = graph.nodes[0]
    packed_weight = plugin_node.inputs[1]
    packed_scale = plugin_node.inputs[2]
    assert packed_weight.values.shape == (n // 2, k)
    assert packed_weight.values.dtype == np.int8
    assert packed_scale.values.shape == (k // block_size, n)
    assert packed_scale.values.dtype == np.float16


def test_rewrite_recovers_exact_int4_values_through_the_pack() -> None:
    """``pack_intweights`` is a pure nibble permutation (see test_weights_quant.py) —
    unpacking the plugin's packed weight must reproduce the same +8-biased
    nibble multiset the surgery computed from ``q_true``."""
    n, k, block_size = 128, 64, 64
    graph, q_true, _ = _build_synthetic_modelopt_int4_graph(n=n, k=k, block_size=block_size)
    graph, _, _ = rewrite_int4_modelopt_dq(graph)

    packed8 = graph.nodes[0].inputs[1].values  # [N/2, K] int8
    packed16 = packed8.view(np.int16).reshape(n // 4, k)
    nibbles_out: list[int] = []
    for word in packed16.astype(np.uint16).flatten():
        nibbles_out.extend([int(word) >> shift & 0xF for shift in (0, 4, 8, 12)])

    expected_unsigned = (q_true.astype(np.int16) + 8).flatten()
    assert sorted(nibbles_out) == sorted(int(x) for x in expected_unsigned)


def test_rewrite_out_of_range_shape_falls_back_to_bypass_not_plugin() -> None:
    """gemm_n not divisible by 128 fails the plugin's divisibility requirement —
    the DQ must be bypassed to its fake-dequant constant, not force-converted."""
    graph, _, _ = _build_synthetic_modelopt_int4_graph(n=96, k=64, block_size=64)  # 96 % 128 != 0
    graph, replaced, materialized = rewrite_int4_modelopt_dq(graph)

    assert replaced == 0
    assert materialized == 1
    assert all(n.op != "Int4GroupwiseGemmPlugin" for n in graph.nodes)


def test_rewrite_raises_nothing_and_no_ops_on_graph_with_no_trt_dq() -> None:
    act = gs.Variable("activation", dtype=np.float32, shape=(1, 4))
    out = gs.Variable("out", dtype=np.float32, shape=(1, 4))
    node = gs.Node(op="Identity", inputs=[act], outputs=[out])
    graph = gs.Graph(nodes=[node], inputs=[act], outputs=[out])

    graph, replaced, materialized = rewrite_int4_modelopt_dq(graph)
    assert (replaced, materialized) == (0, 0)


# --------------------------------------------------------------------------- apply_int4_modelopt_surgery()
def test_apply_surgery_rejects_input_with_no_trt_dequantize_linear(tmp_path: Path) -> None:
    act = gs.Variable("activation", dtype=np.float32, shape=(1, 4))
    out = gs.Variable("out", dtype=np.float32, shape=(1, 4))
    node = gs.Node(op="Identity", inputs=[act], outputs=[out])
    graph = gs.Graph(nodes=[node], inputs=[act], outputs=[out])

    in_path = tmp_path / "not_int4.onnx"
    onnx.save(gs.export_onnx(graph), str(in_path))

    with pytest.raises(ValueError, match="no trt::DequantizeLinear weights"):
        apply_int4_modelopt_surgery(in_path, tmp_path / "out.onnx")


def test_apply_surgery_end_to_end_writes_plugin_node_onnx(tmp_path: Path) -> None:
    graph, _, _ = _build_synthetic_modelopt_int4_graph()
    in_path = tmp_path / "modelopt_w4a16_awq.onnx"
    onnx.save(gs.export_onnx(graph), str(in_path))

    out_path = tmp_path / "int4_plugin.onnx"
    replaced, materialized = apply_int4_modelopt_surgery(in_path, out_path)

    assert (replaced, materialized) == (1, 0)
    assert out_path.is_file()
    saved = onnx.load(str(out_path), load_external_data=False)
    op_types = {n.op_type for n in saved.graph.node}
    assert "Int4GroupwiseGemmPlugin" in op_types
    assert any(o.domain == "trt" for o in saved.opset_import)


# --------------------------------------------------------------------------- _fix_strongly_typed_mismatches()
def _make_mismatched_matmul_model() -> onnx.ModelProto:
    """A MatMul whose two inputs are FLOAT and FLOAT16 - STRONGLY_TYPED-illegal."""
    a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [1, 4])
    b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT16, [4, 1])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [1, 1])
    node = onnx.helper.make_node("MatMul", ["a", "b"], ["y"])
    graph = onnx.helper.make_graph([node], "mismatched", [a, b], [y])
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])


def test_fix_strongly_typed_mismatches_inserts_cast_for_mixed_float_inputs() -> None:
    model = _make_mismatched_matmul_model()
    n_fixed = _fix_strongly_typed_mismatches(model)

    assert n_fixed == 1
    cast_nodes = [n for n in model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 1
    matmul_node = next(n for n in model.graph.node if n.op_type == "MatMul")
    # The minority-typed input (FLOAT, outnumbered 1-vs-1 -> Counter.most_common
    # is stable and keeps the first-seen type, "a"'s FLOAT here) gets cast;
    # whichever input changed, it must now point at the inserted Cast's output.
    assert cast_nodes[0].output[0] in set(matmul_node.input)


def test_fix_strongly_typed_mismatches_is_a_noop_for_uniform_dtypes() -> None:
    a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT16, [1, 4])
    b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT16, [4, 1])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [1, 1])
    node = onnx.helper.make_node("MatMul", ["a", "b"], ["y"])
    graph = onnx.helper.make_graph([node], "uniform", [a, b], [y])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])

    n_fixed = _fix_strongly_typed_mismatches(model)
    assert n_fixed == 0
    assert not any(n.op_type == "Cast" for n in model.graph.node)


def test_fix_strongly_typed_mismatches_pins_int4_plugin_activation_to_bf16() -> None:
    act = onnx.helper.make_tensor_value_info("act", onnx.TensorProto.FLOAT, [1, 64])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.BFLOAT16, [1, 128])
    node = onnx.helper.make_node(
        "Int4GroupwiseGemmPlugin", ["act", "w", "s"], ["y"], domain="trt", gemm_n=128, gemm_k=64, group_size=64
    )
    graph = onnx.helper.make_graph([node], "int4plugin", [act], [y])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("trt", 1)]
    )

    n_fixed = _fix_strongly_typed_mismatches(model)
    assert n_fixed == 1
    plugin_node = next(n for n in model.graph.node if n.op_type == "Int4GroupwiseGemmPlugin")
    cast_node = next(n for n in model.graph.node if n.op_type == "Cast")
    assert plugin_node.input[0] == cast_node.output[0]
    assert cast_node.attribute[0].i == onnx.TensorProto.BFLOAT16
