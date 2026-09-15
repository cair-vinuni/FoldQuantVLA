# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""``profiles_from_onnx`` reads static dims and ranges every symbolic one explicitly."""

import onnx
import pytest
from onnx import TensorProto, helper

from foldquant.runtime.builder import ShapeProfile, profiles_from_onnx


def _graph(tmp_path, seq_dim="seq_len"):
    x = helper.make_tensor_value_info("embeds", TensorProto.BFLOAT16, ["batch", seq_dim, 2048])
    m = helper.make_tensor_value_info("mask", TensorProto.INT64, ["batch", seq_dim])
    y = helper.make_tensor_value_info("out", TensorProto.BFLOAT16, ["batch", seq_dim, 2048])
    node = helper.make_node("Identity", ["embeds"], ["out"])
    graph = helper.make_graph([node], "g", [x, m], [y])
    path = tmp_path / "g.onnx"
    onnx.save(helper.make_model(graph), str(path))
    return path


def test_static_and_ranged_dims(tmp_path):
    path = _graph(tmp_path)
    profiles = profiles_from_onnx(path, {"batch": 1, "seq_len": (1, 151, 2048)})
    assert profiles == {
        "embeds": ShapeProfile((1, 1, 2048), (1, 151, 2048), (1, 2048, 2048)),
        "mask": ShapeProfile((1, 1), (1, 151), (1, 2048)),
    }


def test_missing_symbolic_dim_raises(tmp_path):
    path = _graph(tmp_path)
    with pytest.raises(ValueError, match="seq_len"):
        profiles_from_onnx(path, {"batch": 1})


def test_unordered_range_raises(tmp_path):
    path = _graph(tmp_path)
    with pytest.raises(ValueError, match="min <= opt <= max"):
        profiles_from_onnx(path, {"batch": 1, "seq_len": (151, 1, 2048)})


def test_unnamed_dynamic_dim_raises(tmp_path):
    path = _graph(tmp_path, seq_dim="")
    with pytest.raises(ValueError, match="unnamed"):
        profiles_from_onnx(path, {"batch": 1})
