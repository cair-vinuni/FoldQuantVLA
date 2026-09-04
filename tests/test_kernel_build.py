# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Unit tests for `optimization/targets/tensorrt/kernel_build.py`'s stale-cache guard.

``_invalidate_stale_cmake_cache`` is pure filesystem logic (no cmake/subprocess
involved), so it is testable without a real toolchain: a plugin build cache
directory outlives repo checkouts (it lives under ``~/.cache/foldquant/``), and a
kernel-source-tree reorg leaves behind a ``CMakeCache.txt`` pointing at a source
dir that no longer matches — CMake then refuses to reconfigure in place, which
is otherwise a permanent, non-self-healing failure.
"""

from __future__ import annotations

import os
from pathlib import Path

from foldquant.kernels import build as kernel_build
from foldquant.kernels.build import (
    _install_built_lib,
    _invalidate_stale_cmake_cache,
    ensure_plugins_for_current_device,
)


def _write_cmake_cache(build_tmp: Path, home_dir: str) -> None:
    build_tmp.mkdir(parents=True, exist_ok=True)
    (build_tmp / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={home_dir}\n")


def test_no_cache_file_is_a_no_op(tmp_path: Path) -> None:
    build_tmp = tmp_path / "_build"
    build_tmp.mkdir()
    marker = build_tmp / "keep.txt"
    marker.write_text("still here")

    _invalidate_stale_cmake_cache(build_tmp, tmp_path / "src")

    assert marker.exists()


def test_matching_cached_source_is_kept(tmp_path: Path) -> None:
    build_tmp = tmp_path / "_build"
    src = tmp_path / "src"
    src.mkdir()
    _write_cmake_cache(build_tmp, str(src))
    marker = build_tmp / "CMakeFiles" / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("still here")

    _invalidate_stale_cmake_cache(build_tmp, src)

    assert marker.exists()


def test_mismatched_cached_source_wipes_the_build_dir(tmp_path: Path) -> None:
    build_tmp = tmp_path / "_build"
    old_src = tmp_path / "modules" / "kernels"
    old_src.mkdir(parents=True)
    _write_cmake_cache(build_tmp, str(old_src))
    marker = build_tmp / "CMakeFiles" / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("stale")

    new_src = tmp_path / "kernels" / "tensorrt"
    new_src.mkdir(parents=True)
    _invalidate_stale_cmake_cache(build_tmp, new_src)

    assert not build_tmp.exists()


def test_install_built_lib_replaces_the_inode_and_keeps_open_mappings_valid(tmp_path: Path) -> None:
    """A process that already mapped the old .so must keep reading the old bytes (no in-place overwrite)."""
    built = tmp_path / "build" / "libx.so"
    built.parent.mkdir()
    built.write_bytes(b"new")
    dest = tmp_path / "cache" / "x.so"
    dest.parent.mkdir()
    dest.write_bytes(b"old")
    old_ino = dest.stat().st_ino
    with open(dest, "rb") as old_handle:  # stands in for a dlopen()-ed mapping
        _install_built_lib(built, dest)
        assert old_handle.read() == b"old"
    assert dest.read_bytes() == b"new"
    assert dest.stat().st_ino != old_ino
    assert not list(dest.parent.glob(".*.tmp"))
    assert dest.stat().st_mtime >= built.stat().st_mtime


def _stub_plugin_layout(
    monkeypatch, tmp_path: Path, *, stale: set[str], missing: set[str]
) -> tuple[Path, list[list[str]]]:
    """A fake cache holding every lib not in *missing*; libs in *stale* predate the source tree."""
    cache = tmp_path / "cache"
    cache.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (src / "kernel.cu").write_text("x")
    src_mtime = (src / "kernel.cu").stat().st_mtime
    for name in ("foldquant_int4_per_row", "foldquant_int8_per_row"):
        if name in missing:
            continue
        so = cache / f"{name}.so"
        so.write_bytes(b"so")
        os.utime(so, (src_mtime - 100, src_mtime - 100) if name in stale else (src_mtime + 100, src_mtime + 100))
    calls: list[list[str]] = []

    def fake_rebuild(out_dir: Path, names) -> None:
        calls.append(list(names))
        for name in names:
            (out_dir / f"{name}.so").write_bytes(b"rebuilt")

    monkeypatch.setattr(kernel_build.loc, "plugin_cache_dir", lambda: cache)
    monkeypatch.setattr(
        kernel_build.loc,
        "resolve_plugin_so",
        lambda name: (cache / f"{name}.so") if (cache / f"{name}.so").exists() else None,
    )
    monkeypatch.setattr(kernel_build, "_kernels_source_dir", lambda: src)
    monkeypatch.setattr(kernel_build, "rebuild_plugins", fake_rebuild)
    return cache, calls


def test_only_stale_or_missing_libs_are_rebuilt(monkeypatch, tmp_path: Path) -> None:
    cache, calls = _stub_plugin_layout(monkeypatch, tmp_path, stale={"foldquant_int8_per_row"}, missing=set())
    monkeypatch.setattr(kernel_build.loc, "is_plugin_loaded", lambda path: False)

    paths = ensure_plugins_for_current_device(["foldquant_int4_per_row", "foldquant_int8_per_row"])

    assert calls == [["foldquant_int8_per_row"]]
    assert paths == [cache / "foldquant_int4_per_row.so", cache / "foldquant_int8_per_row.so"]


def test_a_lib_already_loaded_in_this_process_is_never_rebuilt(monkeypatch, tmp_path: Path) -> None:
    cache, calls = _stub_plugin_layout(
        monkeypatch, tmp_path, stale={"foldquant_int4_per_row"}, missing={"foldquant_int8_per_row"}
    )
    monkeypatch.setattr(kernel_build.loc, "is_plugin_loaded", lambda path: path.name == "foldquant_int4_per_row.so")

    ensure_plugins_for_current_device(["foldquant_int4_per_row", "foldquant_int8_per_row"])

    assert calls == [["foldquant_int8_per_row"]]


def test_fresh_cache_triggers_no_rebuild(monkeypatch, tmp_path: Path) -> None:
    _, calls = _stub_plugin_layout(monkeypatch, tmp_path, stale=set(), missing=set())
    monkeypatch.setattr(kernel_build.loc, "is_plugin_loaded", lambda path: False)

    ensure_plugins_for_current_device(["foldquant_int4_per_row", "foldquant_int8_per_row"])

    assert calls == []
