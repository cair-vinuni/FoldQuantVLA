# Repository-level scripts

Everything under `models/<family>/foldquant_integration/` runs one family in
that family's own upstream environment. The scripts here sit above that layer:
they either drive several families in turn or tune the knobs the per-family
`export_foldquant` then consumes.

| script | needs | does |
|---|---|---|
| `bench_all.sh` | built engines, each family's `.venv` | latency for every family into `results/<family>/benchmark.json`; exits non-zero if any family fails (`BENCH_ALLOW_BUSY_GPU=1` runs on a shared GPU, timings then unusable) |
| `smoke_family.sh` | each family's `.venv`, checkpoint and dataset paths | export -> build -> verify on 8 calibration / 8 held-out observations: does the chain run here |
| `smoke_serve.sh` | as above | starts each family's server on a spare port and checks it binds (a bound port, not a served request) |
| `smoke_eval.sh` | the LIBERO simulator stack in the family `.venv` | one LIBERO episode per task through `eval_libero`: does the rollout run |
| `deploy_groot_n17_jetson.sh` | a Jetson AGX Orin, GR00T N1.7 `.venv` | plugins, float pipeline, export, engines, verify and serve in one resumable run; see [`docs/deploy/jetson_serve.md`](../docs/deploy/jetson_serve.md) |
| `_gpu_busy.sh`, `_family_env.sh` | - | sourced helpers: which processes hold the GPU (Tegra-aware), and the `PYTHONPATH` a family's CLI needs |
| `results_tables.py` | `results/` only | regenerates the tables in `results/README.md` from the records |
| `sweep_llm_quant_knobs.py` | a GR00T `.venv`, GPU | RTN grid then GPTQ rescoring over the LLM fold's `sq_alpha` × `act_clip_ratio` |
| `llm_learn_calib.py` | a GR00T `.venv`, GPU | learns per-layer SmoothQuant scales, activation clips and weight clips for the INT4 LLM fold |
| `_groot_family.py` | - | loader shared by the two tuning scripts (policy, decoder, calibration and held-out samples) |

The two tuning scripts are GR00T-only (N1.5 / N1.6 / N1.7). The three
integrations expose the same `calibration.load_policy` / `load_dataset` /
`sample_observations` / `make_forward_loop` surface and the same
`export_foldquant._module_paths`, which is all `_groot_family.Loaded` uses;
openpi goes through `calibration.infer(...)` instead and is
refused with a message rather than half-supported. Run them **from the family
directory, in its environment**: the loader imports `foldquant_integration`
from `models/<family>/`, and the upstream `gr00t` package has to resolve:

```bash
cd models/groot_n1_6
.venv/bin/python ../../scripts/sweep_llm_quant_knobs.py --family groot_n1_6 \
    --model-path <checkpoint> --embodiment-tag libero_panda --dataset-path <dataset> \
    --output ../../results/groot_n1_6/w4a4/sweep_llm_quant_knobs.json
```

## Tuning the LLM fold: `sweep_llm_quant_knobs.py`

The shipped W4A4 defaults (`sq_alpha 0.4`, `act_clip_ratio 1.0`, rotation
block 64) were chosen on one family. The sweep asks whether another point of
the same two knobs does better on a given checkpoint, and does so cheaply:

1. **RTN stage**: every `(alpha, clip)` of the grid (default 5 × 4) is scored
   with round-to-nearest weights on `--max-samples` observations (default 16),
   without GPTQ, so the grid costs seconds per cell.
2. **GPTQ stage**: the top `--top` RTN cells plus the shipped point are
   re-scored with the real GPTQ rounding; the Hessian is what the final export
   uses, so this stage ranks the way the deployed engine will behave.

Two objectives: `seam` (cosine of the decoder's output hidden state against
BF16, the LLM's own damage) and `actions` (cosine of the decoded action chunk,
what the policy does with it). `--per-site` adds a coordinate descent over
per-site clip ratios after the grid. The result records both stages, the best
cell, the gain over the shipped point, and a ready-to-paste `llm_params`.

## Learning the calibration: `llm_learn_calib.py`

An OmniQuant-style alternative to the grid: per decoder layer, the SmoothQuant
scale `s`, the per-site activation clip, and a per-row weight clip `gamma` are
reconstructed by gradient descent through a straight-through estimator of the
INT4 fold, against the BF16 layer's output (`--objective layer`); an optional
end-to-end stage (`e2e`, `both`) fine-tunes the same tensors on the decoder's
final hidden state. `--objective score` only scores an existing file. The
output is a `.pt` of `{sq_scales, act_clip, weight_clip, config}` consumed by
`foldquant.calibrate.load_learned_calib`, which the export takes as

```bash
python -m foldquant_integration.export_foldquant ... \
    --llm-scheme w4a4_srg --llm-params '{"learned_calib": "path/to/calib.pt"}'
```

Scoring is on `--score-samples` **held-out** observations (episodes disjoint
from the calibration set) with GPTQ emulation, and reports the paired
Wilcoxon test against the grid point given by `--alpha` / `--clip`, so a gain
that does not survive 32 held-out observations is reported as such.

## Tuned-arm recipes

The tuned arms of the evaluation are the default schemes with explicit knob
overrides; nothing else changes. Every override is a `--llm-params` /
`--dit-params` JSON (`--expert-params` on pi), and cascade calibration is the `--cascade` flag. The values below are
the ones the reported arms were built with.

| arm | family | `--llm-scheme` | `--llm-params` | action scheme | action params | `--cascade` |
|---|---|---|---|---|---|---|
| arc | N1.5 | `w4a4_srg` | `{"sq_alpha": 0.5, "act_clip_ratio": 0.85}` | `w4a4_sr` | - | no |
| arc | N1.6 | `w4a4_srg` | `{"sq_alpha": 0.6, "act_clip_ratio": 0.85}` | `w4a4_sr` | - | no |
| arc | N1.7 | `w4a4_srg` | `{"sq_alpha": 0.5, "act_clip_ratio": 0.85}` | `w4a4_sr` | - | no |
| arc + fb | N1.6 | `w4a4_srg` | `{"sq_alpha": 0.6, "act_clip_ratio": 0.85}` | `w4a4_sr` / `w4a4_sh` / `w4a4_shg` | `{"sq_fold_order": "before", "sq_alpha": 0.5}` | no |
| arc + fb | N1.7 | `w4a4_srg` | `{"sq_alpha": 0.5, "act_clip_ratio": 0.85}` | `w4a4_sr` / `w4a4_sh` / `w4a4_shg` | `{"sq_fold_order": "before", "sq_alpha": 0.5}` | no |
| arc + fb + res8 | N1.6 | `w4a4_srg` | `{"sq_alpha": 0.6, "act_clip_ratio": 0.85, "site_bits": {"o": 8, "down": 8}}` | `w4a4_sr` | `{"sq_fold_order": "before", "sq_alpha": 0.5}` | no |
| arc + fb, cascade | N1.6 | `w4a4_srg` | `{"sq_alpha": 0.6, "act_clip_ratio": 0.85}` | `w4a4_sr` | `{"sq_fold_order": "before", "sq_alpha": 0.5}` | yes |
| W4A8 arc + fb | N1.6 | `w4a8_srg` | `{"sq_alpha": 0.6}` | `w4a4_sr` | `{"sq_fold_order": "before", "sq_alpha": 0.5}` | no / yes |
| cascade | N1.6, N1.7, pi0.5 | `w4a4_srg` | - | `w4a4_sr` | - | yes |
| cascade | pi0.5 | `w4a4_srg` | - | `w4a4_sh` | `{"sq_fold_order": "before"}` | yes |

Reading notes:

- **arc** = the LLM `(sq_alpha, act_clip_ratio)` the sweep above picked for
  that checkpoint (`results/<family>/w4a4/sweep_llm_quant_knobs.json`); the
  clip is a W4A4-only knob and `w4a8_srg` therefore carries the alpha alone.
  The values are **selections on the LIBERO checkpoints**; applied to a
  different checkpoint of the same family they are borrowed constants, not
  selections, and should be re-swept (on a fine-tuned N1.7 checkpoint outside
  the paper's families the LIBERO clip of 0.85 improved one task's held-out
  drift and worsened another's, while adding `site_bits` recovered both).
- **Private fine-tunes: pass a local path, not the hub id.** Every record
  runs its paths through `foldquant.provenance.public_path`, which strips an
  absolute path to its basename but passes an `org/name` hub id through
  verbatim, which is the point for upstream releases. For a personal or institutional
  fine-tune the org half of that id is an identity, and it would be committed
  with the record.
- **fb** = fold-before on the action module. Butterfly arms (`sh`, `shg`)
  already default to fold-before with `sq_alpha 0.5` (`foldquant.export._act_fold_knobs`),
  so for them the params are a restatement; for the dense `sr` head, whose
  default is fold-after with an absmax scale, they are the change.
- **res8** = `site_bits`, the two residual-writing projections (`o_proj`,
  `down_proj`) at 8-bit weights and activations while `qkv` and `gate/up`
  stay at 4.
- The default `--dit-scheme` in the GR00T integrations is `w4a4_shg`; the
  tuned arms above name `w4a4_sr` explicitly.
- Every arm was calibrated on a manifest of 128 observations spanning all four
  LIBERO suites; the tuning scripts default to 16 (sweep) and 128 (learned
  calibration) samples of the same dataset.
