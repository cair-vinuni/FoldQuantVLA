# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The fold has one implementation, and every emitter must reach it.

The action-module emitters are deliberately split — by architecture, and for the
DiT also by bit width, because those are different designs
rather than one design at two widths. Splitting is what let the INT8 halves stop
folding: each improvement had to be applied twice and the second time was missed,
which cost one expert 0.6005 cosine and went unnoticed because that family had no
INT4 arm to compare against.

These tests make that failure mode loud. They do not forbid the split; they
forbid a second copy of the fold.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "foldquant"
EXPORT = PACKAGE / "export.py"

#: Emitters that quantize an action module. Each may live in its own file; none
#: may carry its own copy of the fold.
ACTION_EMITTERS = (
    "gemma_expert.py",
    "dit_int4.py",
)

#: Fold primitives that belong to omega_rotation/foldq. An emitter naming one of
#: these directly is re-deriving the fold instead of asking for it.
_FOLD_PRIMITIVES = {
    "fold_rotation_sq",
    "fold_rotation_sq_before",
    "fold_weight_sq",
    "pack_int4_colmajor_sq",
    "pack_int4_colmajor_sq_before",
}

#: Rotation constructors. Choosing between the butterfly and the learned dense
#: rotation is a fold decision, so it is made in foldq.site_rotation and nowhere
#: else — an emitter that calls a constructor directly can only build ONE of the
#: two, and a capture stuck on the dense rotation measures the scale in a frame
#: the engine never enters.
_ROTATION_CONSTRUCTORS = {"build_rotation", "hadamard_blocks"}

#: The per-architecture calibration captures. Each lives beside the emitter that
#: consumes it and must take the same knobs, because the scale it measures and
#: the weight the emitter folds have to agree on rotation AND frame.
CAPTURES = {
    "gemma_expert.py": "compute_gemma_expert_sq_scales",
    "dit_int4.py": "compute_dit_sq_scales",
}


def _source(name: str) -> str:
    p = PACKAGE / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    return p.read_text()


@pytest.mark.parametrize("name", ACTION_EMITTERS)
def test_every_action_emitter_offers_both_fold_orders(name: str) -> None:
    """A site's scale lands on the axis fold_order names; every emitter must offer it."""
    assert "fold_order" in _source(name), f"{name} cannot express the fold order"


@pytest.mark.parametrize("name", ACTION_EMITTERS)
def test_every_action_emitter_can_use_the_butterfly(name: str) -> None:
    """The butterfly is the shipped rotation; an emitter stuck on the dense one
    silently misses every improvement made to it."""
    src = _source(name)
    assert "fwht" in src or "hadamard_blocks" in src, f"{name} has no butterfly path"


@pytest.mark.parametrize("name", ACTION_EMITTERS)
def test_no_emitter_reaches_past_foldq_into_the_fold_primitives(name: str) -> None:
    """Fold maths belongs to foldq/omega_rotation, not to each emitter.

    Calling a primitive directly is how two copies drift apart. Emitters may still
    import omega_rotation for packing and byte helpers; what they may not do is
    re-derive which axis the scale lands on.
    """
    src = _source(name)
    tree = ast.parse(src)
    used = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr in _FOLD_PRIMITIVES
    } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in _FOLD_PRIMITIVES}
    assert not used, (
        f"{name} derives the fold itself: {sorted(used)}. Call foldq.fold_site (weights) or "
        "foldq.fold_rotation (a baked matrix) — which axis the SmoothQuant scale lands on is "
        "decided in one place, and a second copy is how the INT8 halves stopped folding."
    )


def test_foldq_is_the_only_module_that_packs_for_both_widths() -> None:
    """One module decides what a folded site looks like at 4 and at 8 bit."""
    src = (PACKAGE / "foldq.py").read_text()
    assert "bits" in src and "quant_weight_per_row" in src and "pack_int4_colmajor" in src


@pytest.mark.parametrize("name", ACTION_EMITTERS)
def test_no_emitter_builds_its_own_rotation(name: str) -> None:
    """Which rotation a site gets is foldq's call, not each emitter's.

    This is the check that would have caught the real defect: an expert capture
    hardcoded the dense constructor and had no ``fwht`` parameter at all, so the
    butterfly arm asked for a rotation it could not build. It only stayed silent
    because the shipped arms fold BEFORE the rotation, where the measured frame
    is the raw one and the rotation goes unused.
    """
    tree = ast.parse(_source(name))
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in _ROTATION_CONSTRUCTORS} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in _ROTATION_CONSTRUCTORS
    }
    assert not used, (
        f"{name} builds its own rotation: {sorted(used)}. Call foldq.site_rotation(weight, "
        "block_size, fwht) so the butterfly and the dense rotation stay one decision."
    )


@pytest.mark.parametrize(("name", "fn"), sorted(CAPTURES.items()))
def test_every_capture_takes_the_same_knobs(name: str, fn: str) -> None:
    """A capture missing a knob measures in a frame the emitter does not fold in.

    Signature-level, not text-level: the file already mentions ``fwht`` for its
    emitter, which is exactly how a capture without it went unnoticed.
    """
    tree = ast.parse(_source(name))
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn]
    assert fns, f"{name} has no {fn}"
    kwonly = {a.arg for a in fns[0].args.kwonlyargs} | {a.arg for a in fns[0].args.args}
    missing = {"fwht", "block_size", "fold_order"} - kwonly
    assert not missing, (
        f"{fn} cannot express {sorted(missing)}. The capture and the emitter it feeds must take "
        "the same knobs, or the scale is measured through a rotation the engine never applies."
    )


def test_no_dispatch_matches_the_dense_w4a4_key_alone() -> None:
    """Every W4A4 dispatch must accept the butterfly key too.

    Comparing a module's scheme against the DENSE key alone silently routes the
    butterfly arm to the INT8 emitter while the plugin-library decision has already
    declared the INT4 library — every node then fails TensorRT import with "Plugin
    not found". This bit an action head after it had already been fixed on the
    DiT and expert branches, which is why it is pinned rather than reviewed.

    Comparing against the GPTQ key alone stays legal: that is how a branch asks
    "is this the GPTQ arm?" after the W4A4 set is already inside.
    """
    tree = ast.parse(EXPORT.read_text())
    bad = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, ast.Eq) for op in node.ops)
        and any(
            isinstance(side, ast.Attribute) and side.attr in ("W4A4_SR", "W4A4_SH")
            for side in [node.left, *node.comparators]
        )
    ]
    assert not bad, (
        f"export.py lines {bad} test a single W4A4 key with ==. Use "
        "`in schemes.ACT_W4A4_SCHEMES` so every W4A4 arm reaches the same emitter, or the "
        "graph and the declared plugin library disagree."
    )


def test_expert_builder_gets_the_capture_fold_kwargs() -> None:
    """The expert emitter must be called with the SAME knobs as its capture.

    The capture and the emitter share one dispatch block, so a kwarg dropped on
    one side is invisible: the capture still measures with it while the emitter
    falls back to its default. An expert lost ``fold_order`` exactly this way —
    scales measured in the raw frame, weights folded in the rotated one, expert
    cosine 0.7837.

    export.py binds the emitter to ONE local name and makes ONE call with
    ``**kwargs`` — there is no second call site to drift.
    """
    tree = ast.parse(EXPORT.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "export_expert")
    bound = {
        alias.asname
        for node in ast.walk(fn)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name == "build_gemma_expert_plugin_onnx"
    }
    assert bound == {"build"}, f"the expert emitter must be imported under the one name `build`, got {bound}"
    calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "build"
    ]
    assert len(calls) == 1, f"expected exactly one emitter call in export_expert, found {len(calls)}"
    starred = [k for k in calls[0].keywords if k.arg is None]
    assert starred and all(isinstance(k.value, ast.Name) for k in starred), (
        "the expert emitter must be called with one **kwargs mapping, not a dict rebuilt inline: "
        "the emitter and the capture take the same knobs, or they disagree on rotation or frame."
    )


def test_every_scheme_is_also_routed() -> None:
    """A key the validator accepts must reach an emitter, not die at dispatch.

    This is the general shape of the bug family this file guards: a scheme is
    added to the vocabulary and to an emitter branch, the validator learns it,
    and the dispatch is never updated — so the arm passes every check and then
    fails at export time. ``w8a8_sh`` hid from every family this way.
    """
    import foldquant.schemes as S
    from foldquant import export

    assert set(export._EXPORTERS) == set(S.MODULES), "every module kind in the vocabulary needs an exporter"
    for scheme in sorted(S.ALL_SCHEMES):
        if scheme == S.W8A8:
            modules = S.MODULES
        elif scheme in S.ACT_FOLDED_SCHEMES:
            modules = S.ACT_MODULES
        else:
            modules = S.LLM_MODULES
        for module in sorted(modules):
            S.validate(module, scheme)  # must not raise
        assert S.plugin_libs(scheme), f"{scheme} names no plugin library"


def test_every_foldquant_scheme_name_follows_the_grammar() -> None:
    """A FoldQuant key spells out its own fold, and bit width is the only prefix.

    ``_s`` scale, ``_r`` dense rotation, ``_h`` butterfly, ``_g`` GPTQ, in that
    order, with ``_h`` and ``_r`` mutually exclusive. The butterfly arm was
    originally keyed ``_hr``, which reads as "butterfly AND dense rotation" —
    impossible — and hid the fact that it applies SmoothQuant too. Same fold,
    two widths, one name: w4a4_sh and w8a8_sh.
    """
    import re

    import foldquant.schemes as S

    bad = []
    for key in sorted(S.ALL_SCHEMES):
        m = re.fullmatch(r"w[48]a[48](?:_([a-z]+))?", key)
        assert m, f"{key!r} is not w<bits>a<bits>[_variant]"
        variant = m.group(1) or ""
        if variant and not re.fullmatch(r"s?[rh]?g?", variant):
            bad.append((key, "not _s/_r|_h/_g in that order"))
        elif "r" in variant and "h" in variant:
            bad.append((key, "_r and _h are alternatives, never both"))
    assert not bad, f"off-grammar FoldQuant keys: {bad}"
