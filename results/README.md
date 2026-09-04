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

For GR00T N1.7 / N1.6 the untouched modules of every TensorRT arm (vision
tower, VL self-attention, state / action encoders, action decoder) are the
upstream float engines, and the float TRT arm is the upstream pipeline with
nothing quantized. GR00T N1.5's upstream engines use a different DiT contract
(fp16, `sa_embs`/`vl_embs` inputs), and openpi, LeRobot and Evo-1 ship no
TensorRT path at all, so for those four families only the two FoldQuant
engines run under TensorRT, the rest of the policy stays in PyTorch, there is
no float TRT arm, and drift is read against the bf16 PyTorch policy alone.

The second quantized module is the family's action generator, whatever its
shape: a DiT for GR00T, a Gemma-300M expert for pi, SmolVLA's dual-stream
expert, Evo-1's cross-attention flow-matching head.

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

**This is the harder of the two protocols in circulation, and the two are not
interchangeable.** A drift figure depends as much on which observations it is
measured over as on the arm under test. Held-out episodes at mid-trajectory
steps — what `verify` samples — put the policy in states the calibration never
saw, partway through a motion, where the scene has moved and the action is
least constrained. A validation suite that instead draws its frames from the
*build* dataset, round-robin over episodes and early in each, is measuring
calibration-adjacent near-initial states, and the same engine scores materially
higher there. Where a number from another harness appears beside one of these,
its sampling protocol is stated with it; a column that mixes the two compares
observation sets, not arms.

For the same reason the tables report the **median** action cosine beside the
mean and the min. These distributions have tails — a handful of observations
carry nearly all of the mean's deficit, while the median sits with the bulk —
so a mean alone reads as a uniform degradation, which is not what the arm does.
Per-observation drift is in each arm's `verify.json`, so a tail can be examined
rather than inferred.

`--components` (`llm`, and the family's action module) installs one engine and
leaves the other in PyTorch, which is how a drift figure is attributed to a
seam rather than to the sum of both. GR00T N1.7 does the same through
upstream's own `--mode` (`vit_llm_only`, `dit_only`).

## Success rate (`eval_libero`)

LIBERO `spatial`, `object`, `goal`, `10` — 10 tasks each, `n` episodes per
task, each family under its own upstream rollout loop, reported per suite and
pooled:

- **GR00T N1.7 / N1.6** — upstream `MultiStepWrapper` (8-step action chunks,
  504-step cap, terminate on success) and the upstream unseeded `reset()`.
  TensorRT arms run one environment at a time (engines pin batch 1); PyTorch
  arms may batch — the protocol is otherwise identical.
- **GR00T N1.5** — upstream's `examples/Libero` client loop (`num_steps_wait`
  no-op steps, per-suite step budgets, 8-step chunks), run in-process around
  the served policy.
- **π₀.₅** — upstream's `examples/libero/main.py` unchanged, against
  `foldquant_integration.serve` (the upstream websocket server over the
  engines): `replan_steps 5`, per-suite step budgets (220 / 280 / 300 / 520),
  50 trials per task, `seed 7`.
- **SmolVLA** — upstream's own evaluator (`lerobot_eval.eval_policy_all`)
  over upstream's `LiberoEnv`, one environment per task, `start_seed 7`, run
  suite by suite from the integration's driver.
- **Evo-1** — upstream's `LIBERO_evaluation/libero_client_4tasks.py`
  unchanged, against the integration's copy of upstream's websocket server
  with the engines installed: upstream's per-suite step budgets, its action
  horizon and its `SEED`.

## Latency (`benchmark`)

Upstream `benchmark_inference.py` for GR00T N1.7 / N1.6, and the
integration's own component timer where the release ships none (N1.5,
π₀.₅, SmolVLA, Evo-1) — 5 warm-up, 20 timed chunks by default, end-to-end
from observation to action chunk, on

- an RTX 4070 Ti SUPER (sm89, 16 GB, TensorRT 10.15), and
- a Jetson AGX Orin 64 GB (sm87, JetPack TensorRT 10.3),

float TRT and FoldQuant arms with identical pipelines apart from the two
quantized engines. Where there is no float TRT arm the reference is the
upstream PyTorch serving configuration: bf16 eager for N1.5, SmolVLA and
Evo-1 — all three serve eagerly upstream — and for π₀.₅ both eager and
upstream's `torch.compile(max-autotune)` default (the FoldQuant seams need the
eager model, so the compiled arm is timed end to end only).

## Results

Measured outputs are committed beside this file as `<family>/<arm>/`:
`verify.json`, `libero/summary.json`, `benchmark.log`, and the arm's
`foldquant_export.json`. Tables in this file are regenerated from those files.

Each integration README carries a **smoke check** — a 16-observation
calibration, 8 held-out observations, one RTX 4070 Ti SUPER — that exercises
the whole export → build → verify → benchmark chain on that family. Those
numbers are sanity gates, not the paper's: the tables below are filled from
the 128-observation exports and the full LIBERO sweeps.

### GR00T N1.7 — LIBERO

_Pending: re-measured with the upstream harness (see the integration README)._

### GR00T N1.6 — LIBERO

_Pending: re-measured with the upstream harness (see the integration README)._

### GR00T N1.5 — LIBERO

_Pending: re-measured with the upstream harness (see the integration README)._

### π₀.₅ — LIBERO

_Pending: re-measured with the upstream harness (see the integration README)._

### SmolVLA — LIBERO

_Pending: re-measured with the upstream harness (see the integration README)._

### Evo-1 — LIBERO

_Pending: re-measured with the upstream harness (see the integration README)._
