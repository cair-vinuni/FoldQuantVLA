# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""The ModelOpt INT8 SmoothQuant baseline: layer exclusions, graph repairs, scheme routing.

None of these need ModelOpt or a GPU; the quantize / export / build path is
exercised by the GR00T N1.7 and Pi0.5 exports on a device.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from foldquant import modelopt_int8, schemes


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.input_layernorm = nn.LayerNorm(8)
        self.norm_out = nn.Linear(8, 8)  # a Linear whose name matches *norm*
        self.action_proj = nn.Linear(8, 4)
        self.final_action_head = nn.Linear(4, 4)
        self.conv = nn.Conv1d(8, 8, 1)


def test_layer_exclusions_list_config() -> None:
    cfg = {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": "smoothquant"}
    excluded = modelopt_int8.apply_layer_exclusions(cfg, _Block())
    # Only quantizable leaves are listed: the LayerNorm itself is not a Linear / Conv.
    assert excluded == ["action_proj", "final_action_head", "norm_out"]
    assert cfg["quant_cfg"][1:] == [
        {"quantizer_name": "*norm_out*", "enable": False},
        {"quantizer_name": "*action_proj*", "enable": False},
        {"quantizer_name": "*final_action_head*", "enable": False},
    ]


class _AdaRMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.dense = nn.Linear(width, 3 * width)  # the adaRMS modulation


class _ExpertLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(width, width, bias=False))
        self.mlp = nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self.mlp, name, nn.Linear(width, width, bias=False))
        self.input_layernorm = _AdaRMSNorm(width)
        self.post_attention_layernorm = _AdaRMSNorm(width)


class _Pi05ExpertTree(nn.Module):
    """The leaf names of Pi0.5's action expert (openpi's module tree seen through ``Pi05ExpertView``,
    the framework's ``action_expert``)."""

    def __init__(self, width: int = 4, action_dim: int = 2) -> None:
        super().__init__()
        self.expert_model = nn.Module()
        self.expert_model.model = nn.Module()
        self.expert_model.model.layers = nn.ModuleList([_ExpertLayer(width) for _ in range(2)])
        self.expert_model.model.norm = _AdaRMSNorm(width)
        self.action_in_proj = nn.Linear(action_dim, width)
        self.action_out_proj = nn.Linear(width, action_dim)
        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)


def test_layer_exclusions_pi05_expert() -> None:
    cfg = {"quant_cfg": [{"quantizer_name": "*", "enable": False}]}
    excluded = modelopt_int8.apply_layer_exclusions(cfg, _Pi05ExpertTree())
    # Only the adaRMS modulation layers stay float: the action and time projections are
    # quantized, because neither action_in_proj nor action_out_proj contains "action_proj".
    assert excluded == [
        "expert_model.model.layers.0.input_layernorm.dense",
        "expert_model.model.layers.0.post_attention_layernorm.dense",
        "expert_model.model.layers.1.input_layernorm.dense",
        "expert_model.model.layers.1.post_attention_layernorm.dense",
        "expert_model.model.norm.dense",
    ]


def test_layer_exclusions_dict_config() -> None:
    cfg = {"quant_cfg": {"*weight_quantizer": {"num_bits": 8}}}
    excluded = modelopt_int8.apply_layer_exclusions(cfg, _Block())
    assert excluded == ["action_proj", "final_action_head", "norm_out"]
    assert cfg["quant_cfg"]["*action_proj*"] == {"enable": False}
    assert "*q_proj*" not in cfg["quant_cfg"]


def test_layer_exclusions_refuse_unknown_shape() -> None:
    with pytest.raises(TypeError):
        modelopt_int8.apply_layer_exclusions({"quant_cfg": "nope"}, _Block())


def test_scheme_is_routed_not_validated() -> None:
    key = modelopt_int8.MODELOPT_W8A8_SMOOTHQUANT
    assert modelopt_int8.is_modelopt_scheme(key)
    assert not modelopt_int8.is_modelopt_scheme(schemes.W8A8_SR)
    assert key in schemes.MODELOPT_SCHEMES and key not in schemes.ALL_SCHEMES
    assert {"llm", "dit", "expert"} <= schemes.MODELOPT_MODULES
    for module in ("dit", "expert"):
        with pytest.raises(ValueError, match="ModelOpt Q/DQ baseline"):
            schemes.validate(module, key)


def _save(model, tmp_path):
    import onnx

    path = tmp_path / "g.onnx"
    onnx.save(model, str(path))
    return path


def test_repair_scatternd_updates_cast_to_data(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 2])
    idx = numpy_helper.from_array(np.array([[0], [2]], dtype=np.int64), "idx")
    upd = numpy_helper.from_array(np.ones((2, 2), dtype=np.float16), "upd")
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [4, 2])
    node = helper.make_node("ScatterND", ["data", "idx", "upd"], ["out"])
    graph = helper.make_graph([node], "g", [data], [out], initializer=[idx, upd])
    path = _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), tmp_path)

    fixed = modelopt_int8.repair_onnx_dtypes(path, "g")
    assert fixed["scatternd"] == 1
    model = onnx.load(str(path))
    cast, scatter = model.graph.node
    assert cast.op_type == "Cast" and cast.input[0] == "upd"
    assert cast.attribute[0].i == TensorProto.FLOAT
    assert scatter.input[2] == cast.output[0]


def test_repair_layernorm_scale_and_complex_cast(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    scale = numpy_helper.from_array(np.ones(3, dtype=np.float16), "scale")
    y = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    norm = helper.make_node("LayerNormalization", ["x", "scale"], ["y"], axis=-1)
    cast = helper.make_node("Cast", ["y"], ["z"], to=TensorProto.COMPLEX128)
    graph = helper.make_graph([norm, cast], "g", [x], [y], initializer=[scale])
    path = _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), tmp_path)

    fixed = modelopt_int8.repair_onnx_dtypes(path, "g")
    assert fixed == {"cast_complex128": 1, "scatternd": 0, "layernorm": 1}
    model = onnx.load(str(path))
    ops = [n.op_type for n in model.graph.node]
    assert ops == ["Cast", "LayerNormalization", "Cast"]
    assert model.graph.node[2].attribute[0].i == TensorProto.BFLOAT16


def test_require_qdq_refuses_float_graph(tmp_path) -> None:
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [x], [y])
    path = _save(helper.make_model(graph), tmp_path)
    assert not modelopt_int8.onnx_has_qdq(path)
    with pytest.raises(ValueError, match="no QuantizeLinear"):
        modelopt_int8.require_qdq(path, "g")


def test_cast_graph_outputs_to_bf16(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [2])
    mask = helper.make_tensor_value_info("mask", TensorProto.BOOL, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    nodes = [
        helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT),
        helper.make_node("Not", ["mask"], ["mask_out"]),
    ]
    mask_out = helper.make_tensor_value_info("mask_out", TensorProto.BOOL, [2])
    graph = helper.make_graph(nodes, "g", [x, mask], [y, mask_out])
    path = _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), tmp_path)

    assert modelopt_int8.cast_graph_outputs(path, "g", TensorProto.BFLOAT16) == ["y"]
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.BFLOAT16
    last = model.graph.node[-1]
    assert (last.op_type, list(last.output)) == ("Cast", ["y"])
    assert model.graph.node[0].output[0] == last.input[0]
    # already bf16: nothing to do
    assert modelopt_int8.cast_graph_outputs(path, "g", TensorProto.BFLOAT16) == []


def test_strip_default_scatternd_reduction(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [4])
    idx = numpy_helper.from_array(np.array([[1]], dtype=np.int64), "idx")
    upd = numpy_helper.from_array(np.ones(1, dtype=np.float32), "upd")
    nodes = [
        helper.make_node("ScatterND", ["data", "idx", "upd"], ["a"], reduction="none"),
        helper.make_node("ScatterND", ["a", "idx", "upd"], ["b"], reduction="add"),
    ]
    out = helper.make_tensor_value_info("b", TensorProto.FLOAT, [4])
    graph = helper.make_graph(nodes, "g", [data], [out], initializer=[idx, upd])
    path = _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), tmp_path)

    assert modelopt_int8.strip_default_scatternd_reduction(path, "g") == 1
    default, add = onnx.load(str(path)).graph.node
    assert list(default.attribute) == []
    assert [(a.name, a.s) for a in add.attribute] == [("reduction", b"add")]


def test_consolidate_external_data_merges_only_this_graph(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    from onnx.external_data_helper import convert_model_to_external_data

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])
    inits = [numpy_helper.from_array(np.full(3, i, dtype=np.float32), f"w{i}") for i in range(3)]
    nodes = [
        helper.make_node("Add", ["x", "w0"], ["t0"]),
        helper.make_node("Add", ["t0", "w1"], ["t1"]),
        helper.make_node("Add", ["t1", "w2"], ["y"]),
    ]
    model = helper.make_model(helper.make_graph(nodes, "g", [x], [y], initializer=inits))
    convert_model_to_external_data(model, all_tensors_to_one_file=False, size_threshold=0)
    path = tmp_path / "llm_bf16.onnx"
    onnx.save(model, str(path))
    scattered = {p.name for p in tmp_path.iterdir()} - {path.name}
    assert len(scattered) == 3
    sibling = tmp_path / "expert_bf16.onnx.data"
    sibling.write_bytes(b"keep")

    assert modelopt_int8.consolidate_external_data(path, "llm") == 3
    assert {p.name for p in tmp_path.iterdir()} == {path.name, "llm_bf16.onnx.data", sibling.name}
    assert sibling.read_bytes() == b"keep"
    loaded = onnx.load(str(path))
    assert [numpy_helper.to_array(t).tolist() for t in loaded.graph.initializer] == [[i] * 3 for i in range(3)]
    # already on one sidecar: nothing to do
    assert modelopt_int8.consolidate_external_data(path, "llm") == 0
