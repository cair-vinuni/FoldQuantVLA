# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""``public_path`` keeps a committed result free of the operator's filesystem."""

from __future__ import annotations

import pytest

from foldquant.provenance import public_path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # An absolute checkpoint keeps the name the operator gave it, nothing above it.
        ("/home/someone/vr_repos/weights/nvidia/GR00T-N1.7-LIBERO-4suite", "GR00T-N1.7-LIBERO-4suite"),
        ("/data/shared/pi05_libero_pytorch/", "pi05_libero_pytorch"),
        ("~/weights/pi05_libero_pytorch", "pi05_libero_pytorch"),
        # A Hugging Face cache carries the repo in its own directory name, so the
        # reduction keeps the identifier rather than the snapshot hash.
        ("/home/someone/.cache/huggingface/hub/models--MINT-SJTU--Evo1_LIBERO/snapshots/3ddc6c9c", "MINT-SJTU/Evo1_LIBERO"),
        ("/x/.cache/huggingface/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots/869589", "HuggingFaceVLA/libero"),
        # Already portable: hub ids and repo-relative paths are left alone.
        ("HuggingFaceVLA/smolvla_libero", "HuggingFaceVLA/smolvla_libero"),
        ("exports/w4a4/engines", "exports/w4a4/engines"),
        ("examples.Libero.custom_data_config:LiberoDataConfig", "examples.Libero.custom_data_config:LiberoDataConfig"),
        # Pass-through, so a caller need not guard an optional field.
        (None, None),
        ("", ""),
        (5, 5),
    ],
)
def test_public_path(raw, expected):
    assert public_path(raw) == expected


def test_repo_internal_paths_stay_relative_and_named():
    """An engine directory must keep the arm: every one of them is called "engines"."""
    from foldquant.provenance import _REPO_ROOT

    raw = str(_REPO_ROOT / "models" / "pi05" / "exports" / "w4a4" / "engines")
    assert public_path(raw) == "models/pi05/exports/w4a4/engines"


def test_no_home_directory_survives():
    """The property that matters: nothing under a user's home reaches a result file."""
    for raw in (
        "/home/alice/ckpt/model",
        "/Users/bob/data/libero_calib",
        "/home/alice/.cache/huggingface/hub/models--org--name/snapshots/deadbeef",
    ):
        out = public_path(raw)
        assert "/home/" not in out and "/Users/" not in out, out
        assert "alice" not in out and "bob" not in out, out
