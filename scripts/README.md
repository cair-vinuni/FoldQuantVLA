# Repository-level scripts

These scripts run checks across model families, generate result tables, and
tune calibration parameters. Each model runs in its own upstream environment
through `models/<family>/foldquant_integration/`.

| script | needs | does |
|---|---|---|
| `bench_all.sh` | built engines, each family's `.venv` | latency for every family into `results/<family>/benchmark.json`; exits non-zero if any family fails (`BENCH_ALLOW_BUSY_GPU=1` runs on a shared GPU, timings then unusable) |
| `smoke_family.sh` | each family's `.venv`, checkpoint and dataset paths | export -> build -> verify on 8 calibration / 8 held-out observations: does the chain run here |
| `smoke_serve.sh` | as above | starts each family's server on a spare port and checks it binds (a bound port, not a served request) |
| `smoke_eval.sh` | the LIBERO simulator stack in the family `.venv` | one LIBERO episode per task through `eval_libero`: does the rollout run |
| `deploy_groot_n17_jetson.sh` | a Jetson AGX Orin, GR00T N1.7 `.venv` | plugins, float pipeline, export, engines, verify and serve in one resumable run; see [`docs/deploy/jetson_serve.md`](../docs/deploy/jetson_serve.md) |
| `_gpu_busy.sh`, `_family_env.sh` | - | sourced helpers: which processes hold the GPU (Tegra-aware), and the `PYTHONPATH` a family's CLI needs |
| `results_tables.py` | `results/` only | prints the paper's drift and latency tables from the committed records |
| `sweep_llm_quant_knobs.py` | a GR00T `.venv`, GPU | RTN grid then GPTQ rescoring over the LLM fold's `sq_alpha` × `act_clip_ratio` |
| `llm_learn_calib.py` | a GR00T `.venv`, GPU | learns per-layer SmoothQuant scales, activation clips and weight clips for the INT4 LLM fold |
| `_groot_family.py` | - | loader shared by the two tuning scripts (policy, decoder, calibration and held-out samples) |

The tuning scripts support GR00T N1.5, N1.6, and N1.7. Run them from the
family directory in its environment:

```bash
cd models/groot_n1_6
.venv/bin/python ../../scripts/sweep_llm_quant_knobs.py --family groot_n1_6 \
    --model-path <checkpoint> --embodiment-tag libero_panda --dataset-path <dataset> \
    --output exports/n16_w4a4/sweep_llm_quant_knobs.json
```

## Tuning the LLM fold: `sweep_llm_quant_knobs.py`

The sweep tunes `sq_alpha` and `act_clip_ratio` for a checkpoint. Defaults are
0.4 and 1.0 respectively, with rotation block 64.

1. **RTN stage**: every `(alpha, clip)` of the grid (default 5 × 4) is scored
   with round-to-nearest weights on `--max-samples` observations (default 16),
   without GPTQ.
2. **GPTQ stage**: the top `--top` RTN cells plus the shipped point are
   rescored with GPTQ using Hessians measured in the transformed activation frame.

`seam` scores decoder-output cosine against BF16; `actions` scores decoded
action cosine. `--per-site` adds coordinate descent over site clip ratios.
Results include both stages, the best parameters, the gain over defaults,
and an `llm_params` value for export.

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

Scoring uses `--score-samples` held-out observations from disjoint episodes
with GPTQ emulation. It reports a paired Wilcoxon test against the baseline
selected by `--alpha` and `--clip`.

## Tuned-arm recipes

The measured configurations below use JSON overrides in `--llm-params` and
`--dit-params` (`--expert-params` on pi). `--cascade` enables cascade calibration.

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
  that checkpoint (written beside the arm's export); the
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
