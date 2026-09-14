# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""The ModelOpt INT8 SmoothQuant baseline: layer exclusions, graph repairs, scheme routing.

None of these need ModelOpt or a GPU; the quantize / export / build path is
exercised by the GR00T N1.7 export on a device.
"""

from __future__ import annotations

import numpy as np
import pytest
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
    with pytest.raises(ValueError, match="ModelOpt Q/DQ baseline"):
        schemes.validate("dit", key)


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
