# Upstream

This directory is the Hugging Face **LeRobot** repository at the **v0.6.1**
release, commit `7e241bd630a3719a56157a497ce5d08f244784f1`, with the FoldQuant
integration added under
[`foldquant_integration/`](foldquant_integration/README.md). Upstream code is
unchanged and keeps its own Apache-2.0 [LICENSE](LICENSE).

SmolVLA is one policy inside that release
(`src/lerobot/policies/smolvla/`), and the release is kept whole rather than
reduced to it: the LIBERO environment (`src/lerobot/envs/libero.py`), the
evaluator (`src/lerobot/scripts/lerobot_eval.py`), the dataset reader and the
processor pipelines the integration drives are all elsewhere in the same tree.

## What was trimmed

Kept: `src/`, `examples/`, `docs/`, `scripts/`, `docker/`, `tests/` (its
source), `pyproject.toml`, `uv.lock`, `setup.py`, `Makefile`, `MANIFEST.in`,
`docs-requirements.txt`, `.gitignore`, the README, LICENSE, security,
contributing and code-of-conduct files.

Dropped, to keep the tree small and text-only:

- `tests/artifacts/` — 73 MB of binary fixtures (sample datasets and their
  `.safetensors` shards, encoded `.mp4` clips, camera `.png`s, pretrained
  policy shards). The test *source* is kept; the tests that read a fixture do
  not run here,
- `media/` (README images — their links in `README.md` are broken by design),
- repository automation, editor and agent-guide files (`.github/`,
  `.pre-commit-config.yaml`, `.dockerignore`),
- the `.gitattributes` LFS filters (nothing LFS-tracked remains).

No submodule is needed: upstream reaches LIBERO through the `hf-libero`
package its `libero` extra pins, not through a checkout.

## Local edits to upstream files

None. The release's `pyproject.toml` and `uv.lock` resolve as shipped
(`uv sync --extra smolvla --extra libero`, Python 3.12); the integration adds
`tensorrt-cu12` and the `foldquant` package on top (see its README).

## Re-syncing

Check out the release tag, copy the kept paths over this directory, and
re-apply the trims above. `foldquant_integration/` imports only public
upstream symbols (`SmolVLAPolicy`, `SmolVLAConfig`, `LeRobotDataset`,
`make_pre_post_processors`, `make_env`, `make_env_pre_post_processors`,
`LiberoEnv`, `lerobot_eval.eval_policy_all`, the `OBS_*` constants), reads
the loaded model's public structure (`model.vlm_with_expert`, its `lm_expert`,
`expert_hidden_size` and `self_attn_every_n_layers`, and the four action
projections), and rebinds two methods on the loaded instance
(`vlm_with_expert.forward`, `denoise_step`); a later release that keeps those
needs no change here.
