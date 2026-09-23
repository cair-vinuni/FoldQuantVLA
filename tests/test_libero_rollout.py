# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""The fixed-initial-state rollout against a fake LIBERO env and policy."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from foldquant.eval_protocol import LIBERO_DUMMY_ACTION  # noqa: E402
from foldquant.libero_rollout import fixed_init_state_env, rollout_task  # noqa: E402


class FakeRawEnv:
    """LIBERO's OffScreenRenderEnv, reduced to what the wrapper touches."""

    def __init__(self):
        self.log = []
        self.state = None
        self.t = 0

    def reset(self):
        self.log.append("reset")
        self.t = 0
        return {"pos": np.zeros(3)}

    def set_init_state(self, state):
        self.log.append(("init", int(state)))
        self.state = int(state)
        return {"pos": np.zeros(3)}

    def step(self, action):
        self.log.append(("step", list(action)))
        self.t += 1
        return {"pos": np.full(3, self.t, dtype=np.float32)}, 0.0, False, {}

    def check_success(self):
        # state s succeeds after s + 1 policy-driven steps; state 3 never does
        return self.state != 3 and self.t >= 10 + self.state + 1


class FakeLiberoEnv(gym.Env):
    def __init__(self):
        self._env = FakeRawEnv()
        self.observation_space = gym.spaces.Dict({"state.pos": gym.spaces.Box(-1, 1, shape=(3,)),
                                                  "annotation.human.action.task_description": gym.spaces.Text(max_length=8)})
        self.action_space = gym.spaces.Dict({"action.pos": gym.spaces.Box(-1, 1, shape=(3,))})

    def _process_observation(self, obs):
        return {"state.pos": obs["pos"], "annotation.human.action.task_description": "task"}

    def reset(self, seed=None, options=None):
        obs = self._env.reset()
        return self._process_observation(obs), {"success": False}

    def step(self, action):
        obs, r, done, info = self._env.step(action["action.pos"])
        info["success"] = self._env.check_success()
        return self._process_observation(obs), r, done, False, info


class FakeMultiStep(gym.Wrapper):
    """Upstream's MultiStepWrapper contract: execute n steps, cap, terminate on success."""

    def __init__(self, env, video_delta_indices, state_delta_indices, n_action_steps, max_episode_steps, terminate_on_success):
        super().__init__(env)
        self.n_action_steps, self.max_episode_steps, self.terminate_on_success = n_action_steps, max_episode_steps, terminate_on_success
        self.reward = []

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.reward = []
        return {k: (np.asarray(v)[None] if not isinstance(v, str) else v) for k, v in obs.items()}, {k: [v] for k, v in info.items()}

    def step(self, action):
        successes = []
        done = False
        for i in range(self.n_action_steps):
            obs, r, _d, _t, info = self.env.step({k: v[i] for k, v in action.items()})
            self.reward.append(r)
            successes.append(bool(info["success"]))
            if len(self.reward) >= self.max_episode_steps:
                done = True
                break
        if self.terminate_on_success and any(successes):
            done = True
        return {k: (np.asarray(v)[None] if not isinstance(v, str) else v) for k, v in obs.items()}, 0.0, done, False, {"success": successes}


class FakePolicy:
    def __init__(self):
        self.calls = []

    def get_action(self, obs):
        self.calls.append(obs)
        assert obs["state.pos"].shape[0] == 1 and obs["annotation.human.action.task_description"] == ["task"]
        return {"action.pos": np.zeros((1, 16, 3), dtype=np.float32)}, {}


def test_episode_i_starts_from_state_i_after_settle_steps():
    raw_log = []

    def make_env():
        env = FakeLiberoEnv()
        raw_log.append(env._env.log)
        return env

    policy = FakePolicy()
    records = rollout_task(policy, make_env, FakeMultiStep, list(range(5)), n_episodes=5,
                           max_episode_steps=24, n_action_steps=8, settle_steps=10)
    assert [r["init_state_id"] for r in records] == [0, 1, 2, 3, 4]
    log = raw_log[0]
    # each episode: reset, set_init_state(i), ten no-op settle steps with the gripper open
    firsts = [i for i, e in enumerate(log) if e == "reset"]
    assert len(firsts) == 5
    for start, ep in zip(firsts, range(5)):
        assert log[start + 1] == ("init", ep)
        assert log[start + 2 : start + 12] == [("step", LIBERO_DUMMY_ACTION)] * 10
    # success arrives after (i+1) policy steps, so within the first chunk; state 3 runs to the cap
    assert [r["success"] for r in records] == [True, True, True, False, True]
    assert records[3]["steps"] == 24
    assert all(r["steps"] == 8 for r in records if r["success"])


def test_wrapper_reports_init_state_and_wraps_around():
    env = fixed_init_state_env(FakeLiberoEnv(), [10, 11], settle_steps=0)
    _obs, info = env.reset()
    assert info["init_state_id"] == 0 and env.init_state_id == 0
    _obs, info = env.reset(options={"init_state_id": 5})
    assert info["init_state_id"] == 1
