# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Run fingerprints, resume refusal and the protocol table."""

import json

import pytest

from foldquant.eval_protocol import PROTOCOLS, artifact_digest, prepare_summary, run_fingerprint


def test_p3_is_the_paper_protocol():
    p3 = PROTOCOLS["p3"]
    assert (p3.max_episode_steps, p3.n_action_steps, p3.n_episodes, p3.settle_steps) == (520, 8, 20, 10)
    assert p3.fixed_init_states
    up = PROTOCOLS["upstream"]
    assert not up.fixed_init_states and up.settle_steps == 0


def test_resolve_prefers_explicit_then_protocol_then_family():
    p3, up = PROTOCOLS["p3"], PROTOCOLS["upstream"]
    assert p3.resolve(None, "max_episode_steps", 504) == 520
    assert p3.resolve(720, "max_episode_steps", 504) == 720
    assert up.resolve(None, "max_episode_steps", 504) == 504


def test_artifact_digest_tracks_content_and_names(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text('{"a": 1}')
    (ckpt / "model.safetensors").write_bytes(b"\0" * 100)
    a = artifact_digest(ckpt)
    (ckpt / "config.json").write_text('{"a": 2}')
    b = artifact_digest(ckpt)
    (ckpt / "config.json").write_text('{"a": 1}')
    (ckpt / "model.safetensors").write_bytes(b"\0" * 101)
    c = artifact_digest(ckpt)
    assert a["digest"] != b["digest"] != c["digest"] and a["digest"] != c["digest"]
    assert artifact_digest(None) is None
    assert artifact_digest(tmp_path / "nope")["missing"] is True


def _run(**over):
    run = {"protocol": {"name": "p3"}, "model": {"digest": "m1"}, "engines": None, "suites": ["libero_10"], "n": 20}
    run.update(over)
    return run


def test_resume_only_into_the_same_run(tmp_path):
    path = tmp_path / "summary.json"
    summary = prepare_summary(path, _run())
    summary["tasks"]["libero_sim/a"] = {"status": "ok", "successes": 1, "num_episodes": 1}
    path.write_text(json.dumps(summary))

    same = prepare_summary(path, _run())
    assert same["tasks"]["libero_sim/a"]["status"] == "ok"
    assert same["fingerprint"] == run_fingerprint(_run())

    for change in ({"model": {"digest": "m2"}}, {"n": 1}, {"protocol": {"name": "upstream"}}, {"engines": {"digest": "e"}}):
        with pytest.raises(SystemExit) as exc:
            prepare_summary(path, _run(**change))
        assert "different run" in str(exc.value)

    fresh = prepare_summary(path, _run(n=1), resume=False)
    assert fresh["tasks"] == {}


def test_legacy_summary_without_fingerprint_is_refused(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"tasks": {"libero_sim/a": {"status": "ok"}}, "n_episodes": 20}))
    with pytest.raises(SystemExit) as exc:
        prepare_summary(path, _run())
    assert "no fingerprint" in str(exc.value)
