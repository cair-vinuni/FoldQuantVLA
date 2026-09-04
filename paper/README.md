# Evaluation protocol and results

Every number reported for FoldQuant is produced by the tools under
`models/<family>/foldquant_integration/`, which drive the **upstream**
evaluation harness of that family — the same rollout loop, wrappers,
episode caps and success definition the model's authors used to report
their own numbers. Nothing in this repository re-implements an evaluator.

## Arms

| arm | LLM | action expert | notes |
|---|---|---|---|
| bf16 | PyTorch | PyTorch | upstream checkpoint, no export |
| float TRT | TensorRT bf16 | TensorRT bf16 | upstream pipeline, unquantized |
| W8A8 | `w8a8_sr` | `w8a8_sh` | dynamic per-row INT8, folded |
| W4A4 | `w4a4_srg` | `w4a4_shg` | INT4 weights and activations, folded, GPTQ |
| W4A4 cascade | `w4a4_srg` | `w4a4_shg` (`--cascade`) | action expert calibrated under the quantized LLM |

The untouched modules of every TensorRT arm (vision tower, VL self-attention,
state / action encoders, action decoder) are the upstream float engines.

## Calibration

128 observations, seeded (`--seed 0`), episode-balanced (round-robin over a
shuffled episode order, one uniform step per visit), drawn through the upstream
data path from the deployment distribution — for LIBERO, all four suites, not
one. The export manifest (`foldquant_export.json`) records every `(episode,
step)` used.

## Drift (`verify`)

Held-out observations from episodes the calibration never saw; per observation
the flow-matching noise is seeded identically for the PyTorch and the engine
pass. Reported: cosine (mean / min) of the LLM output the action head consumes
and of the decoded action chunk, plus action max-abs error. The PyTorch
repeatability under the same seeds is reported alongside so the drift is read
against the sampler's own floor.

## Success rate (`eval_libero`)

LIBERO `spatial`, `object`, `goal`, `10` — 10 tasks each, `n` episodes per
task with the upstream `MultiStepWrapper` (8-step action chunks, 504-step cap,
terminate on success) and the upstream unseeded `reset()`. Reported per suite
and pooled. TensorRT arms run one environment at a time (engines pin batch 1);
PyTorch arms may batch — the protocol is otherwise identical.

## Latency (`benchmark`)

Upstream `benchmark_inference.py` (5 warm-up, 20 timed chunks by default),
end-to-end from observation to action chunk, on

- an RTX 4070 Ti SUPER (sm89, 16 GB, TensorRT 10.15), and
- a Jetson AGX Orin 64 GB (sm87, JetPack TensorRT 10.3),

float TRT and FoldQuant arms with identical pipelines apart from the two
quantized engines.

## Results

Measured outputs are committed under `results/<family>/<arm>/`:
`verify.json`, `libero/summary.json`, `benchmark.log`, and the arm's
`foldquant_export.json`. Tables in this file are regenerated from those files.

### GR00T N1.7 — LIBERO

_Pending: re-measured with the upstream harness (see the integration README)._
