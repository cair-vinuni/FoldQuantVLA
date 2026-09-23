# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""check_records must fail on missing, empty or inconsistent records."""

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("check_records", Path(__file__).resolve().parents[1] / "scripts" / "check_records.py")
check_records = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_records)


def _record(n=4, held_out=True):
    cos = [0.99, 0.5, 0.9, 1.0][:n]
    return {
        "dataset_path": "libero_4suites_calib",
        "seed": 42,
        "num_samples": n,
        "held_out": held_out,
        "schemes": {"llm": "w4a4_srg"},
        "actions": {"cos_min": min(cos), "cos_median": statistics.median(cos), "cos_mean": sum(cos) / n},
        "samples": [{"episode": i, "step": 3, "action_cos": c, "action_max_abs": 0.1} for i, c in enumerate(cos)],
    }


def _write(root, fam, arm, record):
    p = root / fam / arm / "verify.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record))


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(check_records, "RESULTS", tmp_path)
    for fam in check_records.FAMILIES:
        for arm in check_records.ARMS:
            _write(tmp_path, fam, arm, _record())
        (tmp_path / fam / "benchmark.json").write_text("{}")
    return tmp_path


def test_complete_records_pass(results):
    assert check_records.check() == []


def test_empty_results_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(check_records, "RESULTS", tmp_path)
    problems = check_records.check()
    assert len(problems) >= 8 and all("missing" in p for p in problems if "verify" in p)


def test_sample_count_and_held_out_are_enforced(results):
    r = _record()
    r["samples"] = []
    _write(results, "pi05", "w4a4", r)
    assert any("pi05/w4a4" in p and "samples recorded" in p for p in check_records.check())

    r = _record()
    del r["held_out"]
    _write(results, "pi05", "w4a4", r)
    assert any("pi05/w4a4" in p and "held_out" in p for p in check_records.check())


def test_aggregates_must_match_samples(results):
    r = _record()
    r["actions"]["cos_min"] = 0.8
    _write(results, "groot_n1_6", "w4a4", r)
    assert any("cos_min" in p for p in check_records.check())


def test_missing_or_nan_aggregate_fails(results):
    r = _record()
    r["actions"]["cos_min"] = float("nan")
    _write(results, "groot_n1_6", "w4a4", r)
    assert any("cos_min" in p and "finite" in p for p in check_records.check())
    r = _record()
    del r["actions"]["cos_median"]
    _write(results, "groot_n1_6", "w4a4", r)
    assert any("cos_median" in p for p in check_records.check())
    r = _record()
    r["samples"][0]["action_max_abs"] = -1.0
    _write(results, "groot_n1_6", "w4a4", r)
    assert any("action_max_abs" in p for p in check_records.check())


def test_operator_paths_are_rejected(results):
    r = _record()
    r["dataset_path"] = "/home/someone/data"
    _write(results, "groot_n1_5", "w8a8", r)
    assert any("operator path" in p for p in check_records.check())
