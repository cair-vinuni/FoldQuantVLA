# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Unit tests for ``kernels.locator``.

Device/TensorRT-version resolution is exercised by monkeypatching
``current_target_tuple()`` — no CUDA device or TensorRT installation is
required for these tests. ``load_required_plugins()``'s actual ``ctypes``
load path is covered separately by the GPU integration test; here it is only
exercised up to (not including) ``load_plugin_libs()``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from foldquant.kernels import locator as loc

_SOURCE_PATH = Path(loc.__file__)


# --------------------------------------------------------------------------- import hygiene invariant
def test_kernel_locator_has_no_top_level_heavy_or_optimization_imports() -> None:
    """``locator`` must stay import-safe for a bare runtime install.

    No top-level (module-level) import may reference the export code
    (the runtime -> build-time boundary this module exists to keep intact) or
    ``torch``/``tensorrt`` (so importing this module never requires a CUDA/TRT
    install; those imports live inside functions instead).
    """
    tree = ast.parse(_SOURCE_PATH.read_text())
    top_level_module_names: list[str] = []
    for node in tree.body:  # only the module's top level, not nested in functions
        if isinstance(node, ast.Import):
            top_level_module_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            top_level_module_names.append(node.module)

    forbidden_prefixes = ("foldquant.export", "foldquant.calibrate", "torch", "tensorrt")
    violations = [
        name
        for name in top_level_module_names
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
    ]
    assert violations == [], f"locator.py has forbidden top-level import(s): {violations}"


# --------------------------------------------------------------------------- resolve_plugin_so()
@pytest.fixture
def fake_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """Point committed/cache dirs at tmp_path and pin the target tuple to sm86/x86_64/trt10.15."""
    committed_dir = tmp_path / "prebuilt"
    committed_dir.mkdir()
    cache_dir = tmp_path / "cache" / "sm86-x86_64-trt10.15"
    cache_dir.mkdir(parents=True)

    monkeypatch.setattr(loc, "current_target_tuple", lambda: ("sm86", "x86_64", "trt10.15"))
    monkeypatch.setattr(loc, "committed_plugin_dir", lambda: committed_dir)
    monkeypatch.setattr(loc, "plugin_cache_dir", lambda: cache_dir)
    return committed_dir, cache_dir


def test_resolve_exact_slug_match(fake_target: tuple[Path, Path]) -> None:
    committed_dir, _ = fake_target
    exact = committed_dir / "foldquant_int8_per_row.sm86-x86_64-trt10.15.so"
    exact.write_bytes(b"fake-so")

    resolved = loc.resolve_plugin_so("foldquant_int8_per_row")
    assert resolved == exact


def test_resolve_minor_tolerant_match_below_host_minor(
    fake_target: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    committed_dir, _ = fake_target
    older = committed_dir / "foldquant_int8_per_row.sm86-x86_64-trt10.13.so"
    older.write_bytes(b"fake-so")

    with caplog.at_level("WARNING"):
        resolved = loc.resolve_plugin_so("foldquant_int8_per_row")
    assert resolved == older
    assert any("forward-compatible" in rec.message for rec in caplog.records)


def test_resolve_minor_tolerant_picks_highest_eligible_minor(fake_target: tuple[Path, Path]) -> None:
    committed_dir, _ = fake_target
    for minor in (10, 12, 13):
        (committed_dir / f"foldquant_int8_per_row.sm86-x86_64-trt10.{minor}.so").write_bytes(b"fake-so")

    resolved = loc.resolve_plugin_so("foldquant_int8_per_row")
    assert resolved is not None
    assert resolved.name.endswith("trt10.13.so")


def test_resolve_ignores_minor_above_host(fake_target: tuple[Path, Path]) -> None:
    committed_dir, _ = fake_target
    # Only a *newer*-minor binary is committed — not usable on this (older) host.
    (committed_dir / "foldquant_int8_per_row.sm86-x86_64-trt10.20.so").write_bytes(b"fake-so")

    assert loc.resolve_plugin_so("foldquant_int8_per_row") is None


def test_resolve_ignores_mismatched_sm_or_machine(fake_target: tuple[Path, Path]) -> None:
    committed_dir, _ = fake_target
    (committed_dir / "foldquant_int8_per_row.sm87-aarch64-trt10.15.so").write_bytes(b"fake-so")

    assert loc.resolve_plugin_so("foldquant_int8_per_row") is None


def test_resolve_falls_back_to_cache(fake_target: tuple[Path, Path]) -> None:
    _, cache_dir = fake_target
    cached = cache_dir / "foldquant_int8_per_row.so"
    cached.write_bytes(b"rebuilt-so")

    resolved = loc.resolve_plugin_so("foldquant_int8_per_row")
    assert resolved == cached


def test_resolve_prefers_committed_over_cache(fake_target: tuple[Path, Path]) -> None:
    committed_dir, cache_dir = fake_target
    exact = committed_dir / "foldquant_int8_per_row.sm86-x86_64-trt10.15.so"
    exact.write_bytes(b"fake-so")
    (cache_dir / "foldquant_int8_per_row.so").write_bytes(b"rebuilt-so")

    assert loc.resolve_plugin_so("foldquant_int8_per_row") == exact


def test_resolve_returns_none_when_nothing_matches(fake_target: tuple[Path, Path]) -> None:
    assert loc.resolve_plugin_so("foldquant_int8_per_row") is None


# --------------------------------------------------------------------------- load_required_plugins()
def test_load_required_plugins_raises_missing_plugin_error_with_search_paths(
    fake_target: tuple[Path, Path],
) -> None:
    committed_dir, cache_dir = fake_target
    with pytest.raises(loc.MissingPluginError) as exc_info:
        loc.load_required_plugins(["foldquant_int8_per_row", "foldquant_int4_groupwise"])

    message = str(exc_info.value)
    assert "foldquant_int8_per_row" in message
    assert "foldquant_int4_groupwise" in message
    assert str(committed_dir) in message
    assert str(cache_dir) in message


def test_load_required_plugins_reports_only_the_missing_libs(fake_target: tuple[Path, Path]) -> None:
    committed_dir, _ = fake_target
    (committed_dir / "foldquant_int8_per_row.sm86-x86_64-trt10.15.so").write_bytes(b"fake-so")

    with pytest.raises(loc.MissingPluginError) as exc_info:
        loc.load_required_plugins(["foldquant_int8_per_row", "foldquant_int4_groupwise"])

    message = str(exc_info.value)
    assert "foldquant_int4_groupwise" in message
    assert "foldquant_int8_per_row" not in message  # resolved — must not be reported as missing


# --------------------------------------------------------------------------- target_slug() / sm_arch_number()
def test_target_slug_and_sm_arch_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loc, "current_target_tuple", lambda: ("sm86", "x86_64", "trt10.15"))
    assert loc.target_slug() == "sm86-x86_64-trt10.15"
    assert loc.sm_arch_number() == "86"


# --------------------------------------------------------------------------- KNOWN_PLUGIN_LIBS scope
def test_known_plugin_libs_covers_every_shipped_plugin_set() -> None:
    """All three plugin libs participate in default resolution and rebuild.

    ``foldquant_int4_per_row`` (the W4A4 DiT set) was previously excluded here as
    out of scope; it now ships, so leaving it out would mean
    the kernel build silently skips the library a W4A4 preset needs.
    """
    assert set(loc.KNOWN_PLUGIN_LIBS) == {
        "foldquant_int8_per_row",
        "foldquant_int4_groupwise",
        "foldquant_int4_per_row",
    }
    # The named constants and the tuple must not drift apart.
    assert loc.INT4_PER_ROW_LIB in loc.KNOWN_PLUGIN_LIBS
    assert loc.INT8_PER_ROW_LIB in loc.KNOWN_PLUGIN_LIBS
    assert loc.INT4_GROUPWISE_LIB in loc.KNOWN_PLUGIN_LIBS


def test_declared_plugin_libs_accepts_string_list_and_empty() -> None:
    from foldquant.kernels.locator import declared_plugin_libs

    assert declared_plugin_libs(None) == []
    assert declared_plugin_libs({}) == []
    assert declared_plugin_libs({"plugin_lib": ""}) == []
    assert declared_plugin_libs({"plugin_lib": "foldquant_int8_per_row"}) == ["foldquant_int8_per_row"]
    assert declared_plugin_libs({"plugin_lib": ["foldquant_int4_per_row", "foldquant_int8_per_row"]}) == [
        "foldquant_int4_per_row",
        "foldquant_int8_per_row",
    ]
