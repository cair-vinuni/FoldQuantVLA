# Upstream

This directory is the MINT-SJTU **Evo-1** repository at commit
`5fd14b015013c4fd0aacf5f8f48f868ca9b870a2` (`main`), with the FoldQuant
integration added under
[`foldquant_integration/`](foldquant_integration/README.md). Upstream code is
unchanged and keeps its own MIT [LICENSE](LICENSE).

## What was trimmed

Kept: `Evo_1/` (the model, its config, dataset tooling and the training /
serving scripts), `LIBERO_evaluation/`, `libero-plus-eval/`,
`MetaWorld_evaluation/`, `RoboTwin_evaluation/`, `deepspeed_setup_example.txt`,
the README and LICENSE.

Dropped, to keep the tree small and text-only:

- `readme_pics/` (README images — their links in `README.md` are broken by
  design),
- `so100_evo1/`, which vendors a second copy of LeRobot for the SO-100 arm:
  9 MB of duplicated upstream, none of it on the LIBERO path this integration
  evaluates, and this repository already carries LeRobot at
  [`models/smolvla`](../smolvla/UPSTREAM.md).

## Local edits to upstream files

None.

## Re-syncing

Check out the commit, copy the kept paths over this directory, and re-apply the
trims above. `foldquant_integration/` puts `Evo_1/` on `sys.path` — which is
what upstream's own modules do before importing `config` and `model.*` — and
then imports only public upstream symbols
(`scripts.Evo1_server.load_model_and_normalizer`, `.infer_from_json_dict`,
`scripts.Evo1.EVO1`, the `FlowmatchingActionHead` reached through the loaded
model). `Evo1_server.py` guards its entry point with
`if __name__ == "__main__"`, so importing it starts no server.

Two upstream details the integration depends on, both re-checked at runtime:
the action head's `CategorySpecificLinear` collapses to a plain `nn.Linear`
when `num_categories <= 1` (the emitter reads those weights directly), and
`FlowmatchingActionHead.get_action` inlines its denoise step inside the Euler
loop, so the runtime reimplements that loop around upstream's own pieces — see
that module's docstring.
