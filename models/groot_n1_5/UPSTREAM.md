# Upstream

This directory is the NVIDIA Isaac GR00T repository at the **N1.5 release**,
commit `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` (tag `n1.5-release`, branch
`n1d5`), with the FoldQuant integration added under
[`foldquant_integration/`](foldquant_integration/README.md). Upstream code is
unchanged and keeps its own Apache-2.0 [LICENSE](LICENSE).

## What was trimmed

Kept: `gr00t/`, `scripts/`, `deployment_scripts/`, `getting_started/`,
`examples/`, `tests/`, `reference_architecture/reference_architecture.md`,
`pyproject.toml`, `Makefile`, the three Dockerfiles, `.gitignore`, the
README and CONTRIBUTING files. The four `getting_started/*.ipynb` notebooks
are kept with their outputs cleared.

Dropped, to keep the tree small and text-only:

- `media/` (README images and clips — their links in `README.md` are broken
  by design), `reference_architecture/*.png`,
  `examples/SO-100/tictac_bot_setup.jpg`,
- `demo_data/` entirely (the tools under `foldquant_integration/` take
  `--dataset-path`),
- `tests/labeled_frames_video.mp4`, the fixture of `tests/test_load_video.py`
  (that test therefore does not run here),
- repository automation (`.github/`) and the `.gitattributes` LFS filters
  (nothing LFS-tracked remains).

This release declares no submodules and ships no lockfile. LIBERO is not
part of the tree: install it as upstream's `examples/Libero/README.md`
describes (`robosuite==1.4.0` + the LIBERO package), or point the install
at the checkout the N1.7 folder pins
(`models/groot_n1_7/external_dependencies/LIBERO`).

## Local edits to upstream files

None. The release's `pyproject.toml` resolves as shipped
(`uv pip install -e ".[base]"`, then the `flash-attn` wheel and
`tensorrt-cu12` pin from its `deploy` extra — see the integration README).

## Re-syncing

Check out the release commit, copy the kept paths over this directory, and
re-apply the trims above. `foldquant_integration/` imports only public
upstream symbols (`Gr00tPolicy`, `load_data_config`, `LeRobotSingleDataset`,
`EmbodimentTag`, `RobotInferenceServer`, `gr00t.model.policy.COMPUTE_DTYPE`
and `unsqueeze_dict_values`, the LIBERO client's `GR00TPolicy` and
`examples.Libero.eval.utils`) and rebinds two module forwards
(`policy.model.backbone.eagle_model.language_model`,
`policy.model.action_head.model`); a later release that keeps those needs no
change here.
