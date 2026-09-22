# Upstream

This directory is the NVIDIA Isaac GR00T repository at the **N1.7 release**,
commit `23ace64f17aa5015259b8609d371eb61a357c776` (branch `n1.7-release`),
with the FoldQuant integration added under
[`foldquant_integration/`](foldquant_integration/README.md). Upstream code is
unchanged and keeps its own Apache-2.0 [LICENSE](LICENSE).

## What was trimmed

Kept: `gr00t/`, `scripts/`, `getting_started/`, `examples/`, `tests/`,
`docker/`, `demo_data/libero_demo`, `pyproject.toml`, `uv.lock`, the README,
FAQ, ATTRIBUTIONS and CONTRIBUTING files.

Dropped, to keep the tree small and text-only:

- `media/` (README images; their links in `README.md` are broken by design),
- the 405 MB prebuilt aarch64 `flash_attn` wheel under
  `scripts/deployment/dgpu/wheels` (the small `torchcodec` aarch64 wheels the
  Orin and Spark installers look for are kept; the per-device `uv.lock` files
  stay),
- repository automation and agent-guide files (`.github/`, editor configs),
- the `.gitattributes` LFS filters (nothing LFS-tracked remains),
- `demo_data/` other than `libero_demo`,
- the `SimplerEnv` and `robocasa` submodule gitlinks under
  `external_dependencies/`.

The LIBERO submodule is declared at the repository top level (`.gitmodules`,
path `models/groot_n1_7/external_dependencies/LIBERO`) at the commit upstream
pins, `8f1084e3132a39270c3a13ebe37270a43ece2a01`.

## Local edits to upstream files

`pyproject.toml`: `[tool.uv] environments` / `required-environments` are
x86_64 Linux only and the aarch64 `flash-attn`/`torchcodec` path sources are
removed: they referenced the
dropped wheel, and `uv sync` refuses to resolve a path source it cannot read
even on another platform. `uv.lock` is re-locked accordingly. Nothing else
under upstream paths is modified.

## Re-syncing

Check out the release commit, copy the kept paths over this directory, and
re-apply the trims above. `foldquant_integration/` imports only public
upstream symbols (`Gr00tPolicy`, `LeRobotEpisodeLoader`, `extract_step_data`,
`parse_observation_gr00t`, `EmbodimentTag`, the `scripts/deployment` builders and
`trt_model_forward.setup_tensorrt_engines`, `gr00t.eval.rollout_policy`); a
later release that keeps them needs no change here.
