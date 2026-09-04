# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""The kernel registry must agree with the sources it describes.

``registry.py`` exists so that "which kernels need CUTLASS" is stated once,
beside the code. That only helps if the declaration is true — a wrong
``needs_cutlass`` is exactly the bug the registry was introduced to prevent
(``foldquant_int4_per_row`` was treated as CUTLASS-free while its GEMM includes
``<cutlass/cutlass.h>``, which surfaced as a bare "No such file" mid-compile).

So these tests check the declarations against the actual ``#include`` lines and
the actual directory layout, rather than restating them.
"""

from __future__ import annotations

import pytest

from foldquant.kernels.locator import KNOWN_INIT_SYMBOLS, KNOWN_PLUGIN_LIBS
from foldquant.kernels.registry import TENSORRT_KERNELS, kernel_spec, kernels_needing

_SOURCE_SUFFIXES = {".cu", ".cuh", ".h", ".hpp", ".cpp"}


def _sources(spec) -> list:
    return [p for p in spec.source_dir.rglob("*") if p.suffix in _SOURCE_SUFFIXES]


@pytest.mark.parametrize("lib", sorted(TENSORRT_KERNELS))
class TestDeclarationsMatchSources:
    def test_source_dir_exists_with_the_expected_layout(self, lib: str) -> None:
        spec = kernel_spec(lib)
        assert spec.source_dir.is_dir(), f"{lib}: {spec.source_dir} missing"
        assert (spec.source_dir / "kernel.cmake").is_file(), f"{lib}: no kernel.cmake beside its sources"
        assert (spec.source_dir / "cuda").is_dir(), f"{lib}: no cuda/ dir"
        assert (spec.source_dir / "plugin").is_dir(), f"{lib}: no plugin/ dir"

    def test_needs_cutlass_matches_the_includes(self, lib: str) -> None:
        """The declaration must reflect what the code actually includes."""
        spec = kernel_spec(lib)
        includes_cutlass = any("cutlass/" in p.read_text(errors="ignore") for p in _sources(spec))
        assert spec.needs_cutlass == includes_cutlass, (
            f"{lib}: registry says needs_cutlass={spec.needs_cutlass} but "
            f"{'a source includes' if includes_cutlass else 'no source includes'} a cutlass/ header"
        )

    def test_kernel_cmake_declares_the_same_cutlass_dependency(self, lib: str) -> None:
        """CMake and the Python registry must not drift apart."""
        spec = kernel_spec(lib)
        cmake = (spec.source_dir / "kernel.cmake").read_text()
        assert ("NEEDS_CUTLASS" in cmake) == spec.needs_cutlass, (
            f"{lib}: kernel.cmake and registry.py disagree about CUTLASS"
        )

    def test_declared_init_symbols_exist_in_the_plugin_sources(self, lib: str) -> None:
        """A symbol we probe after dlopen must actually be defined somewhere."""
        spec = kernel_spec(lib)
        blob = "\n".join(p.read_text(errors="ignore") for p in _sources(spec))
        missing = [s for s in spec.init_symbols if s not in blob]
        assert not missing, f"{lib}: declares init symbols absent from its sources: {missing}"

    def test_every_cmake_listed_source_exists(self, lib: str) -> None:
        """A kernel.cmake naming a moved/renamed file fails the build, not a test."""
        spec = kernel_spec(lib)
        cmake = (spec.source_dir / "kernel.cmake").read_text()
        missing = []
        for line in cmake.splitlines():
            entry = line.strip()
            if not entry.startswith(("cuda/", "plugin/", "../")):
                continue
            if not (spec.source_dir / entry).resolve().is_file():
                missing.append(entry)
        assert not missing, f"{lib}: kernel.cmake lists sources that do not exist: {missing}"


class TestRegistryIsTheSingleSourceOfTruth:
    def test_locator_lib_names_come_from_the_registry(self) -> None:
        assert set(KNOWN_PLUGIN_LIBS) == set(TENSORRT_KERNELS)

    def test_locator_init_symbols_are_the_registry_union(self) -> None:
        expected = {s for spec in TENSORRT_KERNELS.values() for s in spec.init_symbols}
        assert set(KNOWN_INIT_SYMBOLS) == expected

    def test_init_symbols_are_unique_across_kernels(self) -> None:
        """Two libraries claiming one symbol would make dlopen order matter."""
        seen: dict = {}
        for lib, spec in TENSORRT_KERNELS.items():
            for sym in spec.init_symbols:
                assert sym not in seen, f"{sym} claimed by both {seen[sym]} and {lib}"
                seen[sym] = lib

    def test_kernels_needing_partitions_the_registry(self) -> None:
        assert set(kernels_needing(cutlass=True)) | set(kernels_needing(cutlass=False)) == set(TENSORRT_KERNELS)
        assert not set(kernels_needing(cutlass=True)) & set(kernels_needing(cutlass=False))

    def test_unknown_kernel_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown kernel"):
            kernel_spec("foldquant_not_a_kernel")
