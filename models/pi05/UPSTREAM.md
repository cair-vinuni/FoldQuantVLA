# Upstream

This directory is the Physical Intelligence **openpi** repository at commit
`215abfb217dbac7d5f1273282331b9b1866c0479` (`main`, September 2025), with the
FoldQuant integration added under
[`foldquant_integration/`](foldquant_integration/README.md). Upstream code is
unchanged and keeps its own Apache-2.0 [LICENSE](LICENSE).

## What was trimmed

Kept: `src/openpi/` (JAX and PyTorch models, transforms, policies, serving,
training, the `transformers_replace` Gemma patch), `packages/openpi-client/`,
`scripts/`, `examples/` (every example's code, README, Dockerfile and
requirements), `docs/`, `pyproject.toml`, `uv.lock`, `.python-version`,
`.gitignore`, the README, LICENSE and contributing notes.

Dropped, to keep the tree small and text-only:

- repository automation and editor settings (`.github/`, `.vscode/`,
  `.pre-commit-config.yaml`, `.dockerignore`),
- the ALOHA simulator submodule (`third_party/aloha`) — not used here.

The LIBERO submodule is kept and re-pinned at the top level of this
repository (`models/pi05/third_party/libero`,
`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`, upstream's own pin); initialise
it with `git submodule update --init models/pi05/third_party/libero` before
setting up the LIBERO client environment described in
`examples/libero/README.md`.

## Local edits to upstream files

None. The release's `pyproject.toml` and `uv.lock` resolve as shipped
(`GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11`); the integration adds
`tensorrt-cu12` and the `foldquant` package on top (see its README).

## Re-syncing

Check out the commit, copy the kept paths over this directory, and re-apply
the trims above. `foldquant_integration/` imports only public upstream
symbols (`openpi.training.config.get_config`,
`openpi.policies.policy_config.create_trained_policy`,
`openpi.models.model.Observation`, `openpi.models.gemma.get_config`,
`openpi.serving.websocket_policy_server.WebsocketPolicyServer`,
`openpi_client.image_tools.resize_with_pad`, the pinned `lerobot` dataset
class), reads the loaded `Policy`'s private state (`_model`,
`_input_transform`, `_sample_kwargs`, `_sample_actions`, `_pytorch_device`),
clears `Pi0Config.pytorch_compile_mode` for the quantized arms, and rebinds
two methods on the loaded model (`paligemma_with_expert.forward`,
`denoise_step`); a later release that keeps those needs no change here.
