# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""LIBERO reads the pinned checkout's paths, whatever ``~/.libero`` says."""

from __future__ import annotations

from pathlib import Path

import pytest

from foldquant import libero_config
from foldquant.libero_config import CONFIG_ENV, use_checkout


@pytest.fixture
def env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.delenv(libero_config._OWNED_ENV, raising=False)
    monkeypatch.setenv("FOLDQUANT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def _read(config_dir: Path) -> dict:
    return dict(line.split(": ", 1) for line in (config_dir / "config.yaml").read_text().splitlines())


def test_the_config_names_the_checkout_and_lives_in_the_cache(env: Path, monkeypatch) -> None:
    other = env / "home" / ".libero"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text("benchmark_root: /somebody/else/libero/libero\n")
    root = env / "pinned" / "libero" / "libero"
    root.mkdir(parents=True)
    config_dir = use_checkout(root)
    assert config_dir.is_relative_to(env / "cache")
    import os

    assert os.environ[CONFIG_ENV] == str(config_dir)
    assert _read(config_dir)["benchmark_root"] == str(root.resolve())
    assert _read(config_dir)["init_states"].startswith(str(root.resolve()))
    # The shared file is not touched.
    assert (other / "config.yaml").read_text() == "benchmark_root: /somebody/else/libero/libero\n"


def test_two_checkouts_get_two_configs_and_a_second_call_switches(env: Path) -> None:
    a, b = env / "a" / "libero" / "libero", env / "b" / "libero" / "libero"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    da, db = use_checkout(a), use_checkout(b)
    assert da != db
    assert _read(db)["benchmark_root"] == str(b.resolve())


def test_a_user_set_config_path_is_honoured(env: Path, monkeypatch) -> None:
    mine = env / "mine"
    mine.mkdir()
    (mine / "config.yaml").write_text("benchmark_root: /my/choice\n")
    monkeypatch.setenv(CONFIG_ENV, str(mine))
    root = env / "pinned" / "libero" / "libero"
    root.mkdir(parents=True)
    assert use_checkout(root) == mine
    assert (mine / "config.yaml").read_text() == "benchmark_root: /my/choice\n"
