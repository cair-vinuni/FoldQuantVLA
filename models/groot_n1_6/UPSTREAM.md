# Upstream

This directory is the NVIDIA Isaac GR00T repository at the **N1.6.1 release**,
commit `5dc80c4afd726b34faad1d8f7e007a13b34e4c88` (branch `n1.6.1-release`),
with the FoldQuant integration added under
[`foldquant_integration/`](foldquant_integration/README.md). Upstream code is
unchanged and keeps its own Apache-2.0 [LICENSE](LICENSE).

## What was trimmed

Kept: `gr00t/`, `scripts/`, `getting_started/`, `examples/`, `tests/`,
`test_support/`, `docker/`, `pyproject.toml`, `uv.lock`, `.gitignore`, the
README and CONTRIBUTING files.

Dropped, to keep the tree small and text-only:

- `media/` (README images and clips; their links in `README.md` are broken
  by design),
- `demo_data/` entirely (every dataset there is LFS-only in this release;
  the tools under `foldquant_integration/` take `--dataset-path`),
- repository automation (`.github/`) and the `.gitattributes` LFS filters
  (nothing LFS-tracked remains),
- the `robocasa`, `robocasa-gr1-tabletop-tasks`, `SimplerEnv` and
  `GR00T-WholeBodyControl` submodule gitlinks under `external_dependencies/`,
- `examples/LIBERO/patches/episode_000082.mp4`, a replacement clip for one
  corrupted episode of the LIBERO *fine-tuning* dataset (`examples/LIBERO/README.md`
  still shows the `cp` step); inference and calibration never need it.

The LIBERO submodule is declared at the repository top level (`.gitmodules`,
path `models/groot_n1_6/external_dependencies/LIBERO`) at the commit upstream
pins, `8f1084e3132a39270c3a13ebe37270a43ece2a01`, the same commit the N1.7
folder uses, so `git submodule update --init --reference` against that
checkout avoids a second download.

## Local edits to upstream files

`.gitignore`: two agent-tool ignore patterns removed. Nothing else; this
release's `pyproject.toml` carries no path sources, so it resolves as shipped
(`uv sync --python 3.10`; the `flash-attn` wheel is fetched from the GitHub
release URL upstream pins).

## Re-syncing

Check out the release commit, copy the kept paths over this directory, and
re-apply the trims above. `foldquant_integration/` imports only public
upstream symbols (`Gr00tPolicy`, `Gr00tSimPolicyWrapper`, `LeRobotEpisodeLoader`,
`extract_step_data`, `parse_observation_gr00t`, `EmbodimentTag`,
`gr00t.eval.rollout_policy`, the `scripts/deployment/benchmark_inference.py`
helpers) and rebinds two module forwards (`policy.model.backbone.model.language_model`,
`policy.model.action_head.model`); a later release that keeps those needs no
change here.
