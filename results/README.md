# Evaluation protocol and results

Every number here comes from the tools under
`models/<family>/foldquant_integration/`, which drive that family's
**upstream** evaluation harness: the same rollout loop, wrappers, episode caps
and success definition its authors used. Nothing in this repository
re-implements an evaluator.

Each family therefore runs under *its own* loop, and those loops differ (see
the settle-step note under Success rate). These are **per-family
reproductions**: "what does this arm do to this policy, run the way its authors
run it". Comparing success rates across families needs one harness applied to
all four; where a paper table does that, it says so.

## Arms

| arm | LLM | action expert | notes |
|---|---|---|---|
| bf16 | PyTorch | PyTorch | upstream checkpoint, no export |
| float TRT | TensorRT bf16 | TensorRT bf16 | upstream pipeline, unquantized |
| W8A8 | `w8a8_sr` | `w8a8_sh` | dynamic per-row INT8, folded |
| W4A4 | `w4a4_srg` | `w4a4_shg` | INT4 weights and activations, folded, GPTQ |
| W4A4 cascade | `w4a4_srg` | `w4a4_shg` (`--cascade`) | action expert calibrated under the quantized LLM |
| W4A4 + o/d INT8 | `w4a4_srg` + `site_bits {o: 8, down: 8}` | as W4A4 | site-selective INT8: `o_proj` and `down_proj` held at INT8 inside W4A4 |

Arm names are used the same way in every table here: "float TRT" is the
unquantized TensorRT arm, "FoldQuant" prefixes this repository's engines where
a table also lists other methods, and "o/d INT8" is the site-selective arm
([`SITE_SELECTIVE_INT8.md`](SITE_SELECTIVE_INT8.md)).

For GR00T N1.7 the untouched modules of every TensorRT arm (vision tower, VL
self-attention, state / action encoders, action decoder) are the upstream float
engines, and the float TRT arm is upstream's full pipeline
(`n17_full_pipeline`) with nothing quantized: eight engines, text tower
included.

**N1.6's float arm has been two different things; the records say which.**
Upstream's `scripts/deployment/export_onnx_n1d6.py` exports only the DiT, so
the first float arm was upstream's pipeline as shipped: DiT under TensorRT,
text tower in eager PyTorch. The latency table below still uses it
(`components` is `["dit"]`); its backbone matches eager to within a fifth of a
millisecond while the quantized arms' backbone falls to 17-18 ms.

`--llm-scheme float` traces a float text tower too, so the drift table's float
row is the full-scope arm (both modules under TensorRT, nothing quantized) and
its record reads `schemes {"llm": "float", "dit": "float"}`. Check the
`components` or `schemes` field before comparing a number across families.

This is why the tables quote speedup against **eager PyTorch**, not the float
TRT arm. N1.6's float baseline leaves the larger module unaccelerated, so a
float-relative ratio flatters it: on one RTX 4070 Ti SUPER, W4A4 is 1.42x its
float arm on N1.6 and 1.27x on N1.7, while against eager the order reverses to
1.94x and 2.08x. Only the second pair compares the families.

GR00T N1.5's upstream engines use a different DiT contract (fp16,
`sa_embs`/`vl_embs` inputs), and openpi ships no TensorRT path. For those two
families only the two FoldQuant engines run under TensorRT, there is no float
TRT arm, and drift is read against the bf16 PyTorch policy alone.

The second quantized module is the family's action generator: a DiT for GR00T,
a Gemma-300M expert for pi.

## Calibration

128 observations, seeded (`--seed 0`), episode-balanced (round-robin over a
shuffled episode order, one uniform step per visit), drawn through the upstream
data path from the deployment distribution (for LIBERO, all four suites). The
export manifest (`foldquant_export.json`) records every `(episode, step)` used.

## Drift (`verify`)

Held-out observations from episodes the calibration never saw, with the
flow-matching noise seeded identically for the PyTorch and engine passes.
Reported: cosine (median; mean and min stay in the record) of the LLM output
the action head consumes and of the decoded action chunk, plus action max-abs
error. PyTorch repeatability under the same seeds is reported alongside as the
sampler's floor.

**This is the harder of the two protocols in circulation; the two are not
interchangeable.** `verify` samples held-out episodes at mid-trajectory steps,
where the scene has moved and the action is least constrained. A suite that
draws frames from the *build* dataset, early in each episode, measures
calibration-adjacent states, and the same engine scores materially higher.
Where another harness's number appears beside these, its sampling protocol is
stated; a column mixing the two compares observation sets, not arms.

"Flip" counts depend on the action scale: a saturated channel is |Δ| ≈ 2 where
actions run [-1, 1], but not elsewhere. The cross-family column is therefore
the **median worst-|Δ|** (per observation, `action_worst`), not a count above a
threshold.

GR00T returns a named action dict, so its records name the worst channel
(`gripper`, `x`, `y`, `z`); π₀.₅ returns an unnamed vector, so
`action_worst.label` is null there and only index and magnitude are recorded.

### The drift figure is the median

The tables report only the **median** action cosine. Mean and minimum stay in
every `verify.json` but are not the number to read.

VLA actions are clipped. A chunk railed on every channel compares signs and
scores 1.000 whatever the arm did; a chunk with two channels railed and the rest
near zero loses most of its cosine to a small error on a near-zero channel. The
distribution is bimodal, so the mean tracks how often the policy was railed,
not engine fidelity.

Measured on a GR00T N1.6 Bridge W4A4 arm over 32 held-out observations: 20
observations have `|action| ≈ 2.830` (norm² = 8.01, `max|action|` exactly
1.0000, eight channels at the rail) and all score above 0.998. Six have
`|action| ≈ 1.416` (norm² = 2.01, two channels at the rail) and carry almost
the whole deficit, down to 0.022. Mean 0.842, median 0.9985. The same arm's
SimplerEnv success rate, 200 episodes over seven tasks, is 0.612 against bf16's
0.622.

The median does not predict success rate (on that checkpoint the best-median
arm has the lowest success rate, close to reversed order). It answers the
drift table's question, whether the typical action is the same action, where
the mean calls an arm broken that closed loop shows is not.

Per-observation drift is in each arm's `verify.json`, so a tail can be examined
rather than inferred.

`--components` (`llm`, and the family's action module) installs one engine and
leaves the other in PyTorch, attributing drift to a seam. GR00T N1.7 does the
same through upstream's `--mode` (`vit_llm_only`, `dit_only`).

That attribution is by seam, not by observation: **backbone damage does not
predict which observation's chunk breaks.** Each `verify.json` sample carries
the worst per-position cosine of what the action module consumes
(`backbone_token_cos_min` for GR00T, `kv_stack_position_cos_min` for π₀.₅).
Over the 32 held-out observations of each W4A4 arm, its Pearson correlation
with the action cosine is +0.30 (N1.7), +0.16 (N1.6), +0.08 (N1.5), +0.31
(π₀.₅): positive but weak everywhere, and in every family the most-damaged
prefix decodes to an action cosine of 0.9987 or better. See
[`groot_n1_7/README.md`](groot_n1_7/README.md) for the case that prompted the
check; `scripts/results_tables.py --table correlation` regenerates the four
figures from the committed records.

## Success rate (`eval_libero`)

LIBERO `spatial`, `object`, `goal`, `10` (10 tasks each, `n` episodes per
task), each family under its own upstream rollout loop, reported per suite and
pooled:

- **GR00T N1.7 / N1.6**: upstream `MultiStepWrapper` (8-step action chunks,
  504-step cap, terminate on success) and the upstream unseeded `reset()`, with
  **no settle steps**: the policy acts on the first frame after reset. N1.5 and
  π₀.₅ first wait ten steps issuing LIBERO's no-op `[0, 0, 0, 0, 0, 0, -1]`
  (last channel holds the gripper open). The asymmetry is upstream's, but it is
  a real difference in starting conditions, so do not read success rates across
  families as if it were absent. TensorRT arms run one environment at a time
  (engines pin batch 1); PyTorch arms may batch; the protocol is otherwise
  identical.
- **GR00T N1.5**: upstream's `examples/Libero` client loop (`num_steps_wait`
  no-op steps, per-suite step budgets, 8-step chunks), run in-process around
  the served policy.
- **π₀.₅**: upstream's `examples/libero/main.py` unchanged, against
  `foldquant_integration.serve` (the upstream websocket server over the
  engines): `replan_steps 5`, per-suite step budgets (220 / 280 / 300 / 520),
  50 trials per task, `seed 7`.

## Latency (`benchmark`)

Upstream `benchmark_inference.py` for GR00T N1.7 / N1.6, and the integration's
own component timer where the release ships none (N1.5, π₀.₅): 5 warm-up, 20
timed chunks by default, observation to action chunk, on

- an RTX 4070 Ti SUPER (sm89, 16 GB, TensorRT 10.15), and
- a Jetson AGX Orin 64 GB (sm87, JetPack TensorRT 10.3),

with float TRT and FoldQuant arms identical apart from the two quantized
engines. Every figure is measured without the runtime's opt-in CUDA-graph
replay; on π₀.₅ replay moves the ten-step denoise loop by at most 1.5 ms. Without a float TRT arm the reference is upstream's PyTorch serving
configuration: bf16 eager for N1.5 (served eagerly upstream), and for π₀.₅
both eager and upstream's `torch.compile(max-autotune)` default (timed end to
end only, since the FoldQuant seams need the eager model).

**Every latency figure in this repository is a median.** Upstream's N1.7 log
prints `median=`, `mean= ± sd`, `min` and `max` on one line (component rows are
already medians), so copying the wrong field is an easy, invisible error. With
20 timed chunks the mean moves with the tail; the tables take `median=`.

**A speedup against eager is not a quantization speedup**, and here most of it
is not. The float TRT arm (same graph, compiled, nothing quantized) separates
the two, and where it exists the split is consistent:

| module | eager | float engine | W4A4 | graph | precision |
|---|---|---|---|---|---|
| N1.7 text tower | 27.47 | 16.84 | 13.87 | 10.63 (78%) | 2.97 (22%) |
| N1.7 action head | 39.47 | 21.76 | 15.84 | 17.71 (75%) | 5.92 (25%) |
| N1.6 action head | 38.20 | 19.86 | 14.60 | 18.33 (78%) | 5.26 (22%) |

Each difference is taken within one process where possible: upstream's N1.7
script re-times eager on every run, so the graph term pairs the float arm with
the eager from *its own* log, and only the float-to-W4A4 term crosses
processes. Pairing with another log's eager moves the split by a few points
(73-78% across the four N1.7 logs, whose eager action head spans
37.74-39.47 ms) without changing the reading.

Three module-level measurements across two families land at 75-78% (73-78%
across every pairing): **about three quarters of what a FoldQuant arm saves
against eager is TensorRT compiling the graph, and about one quarter is the
precision.** Both are part of the deployment path's value, but only the float
arm tells them apart.

Compiling with nothing quantized isolates the graph too. π₀.₅'s upstream
`torch.compile(max-autotune)` arm recovers 69.07 ms of the 92.81 ms between
eager and W4A4: **74.4%**, inside the 73-78% the float engines give on another
family through another mechanism.

Outside GR00T no family has a float engine for its action module; π₀.₅ has the
compile-only arm and N1.5 neither. Every row can still bound the last step,
8-bit to 4-bit, which no graph change explains:

| | eager → W4A4 saved | of which 8→4 bit |
|---|---|---|
| N1.5 action head | 8.88 ms | 2.74 ms (31%) |
| π₀.₅ denoise loop | 48.70 ms | 3.46 ms (7%) |

The rest of each row is graph and 8-bit quantization together; for N1.5 these
records do not separate the two.

## Results

Measured outputs are committed beside this file as `<family>/<arm>/`:
`verify.json` (drift), the arm's `foldquant_export.json` (calibration
manifest), and latency as `<family>/benchmark.json` (all arms of one family in
one run) or, for GR00T N1.7, `<family>/<arm>/benchmark.log` (upstream's script
writes one log per arm). `groot_n1_7/bf16/libero/summary.json` is the
release-harness LIBERO check below, and `<family>/w4a4/sweep_llm_quant_knobs*.json`
are the LLM knob sweeps `scripts/README.md` describes. Tables are regenerated from those files,
so every cell traces to a committed artifact.

The notes beside this file carry the paper's remaining figures:

- [`SITE_SELECTIVE_INT8.md`](SITE_SELECTIVE_INT8.md): the o/d INT8 arm on
  every checkpoint (fidelity, closed loop, latency, engine bytes).
- [`EXPERT_PROFILE.md`](EXPERT_PROFILE.md): where a π₀.₅ expert step goes,
  and the measured direction of the attention fix the paper cites.
- [`groot_n1_7/HOLOQ_LIBERO.md`](groot_n1_7/HOLOQ_LIBERO.md) and
  [`pi05/HOLOQ_LIBERO.md`](pi05/HOLOQ_LIBERO.md): FoldQuant engines beside
  the W4A4 methods HoloQ-VLA tabulates.
- [`groot_n1_7/README.md`](groot_n1_7/README.md): per-arm N1.7 tables and the
  per-observation check behind the correlation figures above.

Each integration README has a **smoke check** (16-observation calibration, 8
held-out observations, one RTX 4070 Ti SUPER) that exercises export → build →
verify → benchmark. Those are sanity gates; the tables below come from the
128-observation exports and the full LIBERO sweeps.

`scripts/check_records.py` checks the invariants the records must satisfy: a
float arm no less faithful than the INT8 one from the same graph, a stated
scope, a held-out split, no operator paths. Each rule exists because breaking
it once produced a plausible wrong number. It needs only `results/`.

Every table below is emitted by `scripts/results_tables.py`, which reads only
the committed records (no GPU, engines or upstream environment), so a stale
cell shows up as a diff.

### Drift: all families

Held-out action cosine per arm, 32 observations, seeded
(`<family>/<arm>/verify.json`). The PyTorch repeatability floor is 1.000000
under the same seeds in every family, so the deficits are the arm's.

| family | arm | n | action cos (median) | median worst \|Δ\| |
|---|---|---|---|---|
| GR00T N1.7 | `float` | 32 | 0.99999 | 0.009 |
| GR00T N1.7 | `w8a8` | 32 | 0.99996 | 0.011 |
| GR00T N1.7 | `w4a4` | 32 | 0.99817 | 0.099 |
| GR00T N1.7 | `w4a4_cascade` | 32 | 0.99825 | 0.092 |
| GR00T N1.6 | `float` | 32 | 0.99999 | 0.005 |
| GR00T N1.6 | `w8a8` | 32 | 0.99998 | 0.007 |
| GR00T N1.6 | `w4a4` | 32 | 0.99876 | 0.074 |
| GR00T N1.6 | `w4a4_cascade` | 32 | 0.99874 | 0.071 |
| GR00T N1.5 | `float` | 32 | 0.99999 | 0.005 |
| GR00T N1.5 | `w8a8` | 32 | 0.99999 | 0.007 |
| GR00T N1.5 | `w4a4` | 32 | 0.99885 | 0.072 |
| GR00T N1.5 | `w4a4_cascade` | 32 | 0.99865 | 0.067 |
| π₀.₅ | `float` | 32 | 1.00000 | 0.002 |
| π₀.₅ | `w8a8` | 32 | 1.00000 | 0.005 |
| π₀.₅ | `w4a4` | 32 | 0.99942 | 0.054 |
| π₀.₅ | `w4a4_cascade` | 32 | 0.99945 | 0.054 |

These distributions have tails; where W4A4 breaks, a minority of observations
carries the deficit. Worst \|Δ\| follows each family's action scale: 1.000 is a
saturated 0/1 gripper on GR00T, ~2.0 a saturated channel where actions run
[-1, 1].

**Read the float row first.** It is the same graph with nothing quantized, so
whatever it costs is not quantization.

A control must be built like the arms. A traced float graph was first compiled
weakly typed, so TensorRT ran some layers in fp32, *more* exact than the bf16
reference: π₀.₅'s float engine read 0.98256 mean against an INT8 engine's
1.00000. Built STRONGLY_TYPED, honouring the ONNX bf16 dtypes, it reads
1.00000. A control that differs from the arm in two ways measures neither.

### Latency: all families

End-to-end median ms per action chunk, one RTX 4070 Ti SUPER (sm89, TensorRT
10.15), 20 timed chunks after 5 warm-up. GR00T N1.7 re-times eager in every
arm's log; the column shows the eager from the W4A4 log, as in the split above.

| family | eager | float TRT | W8A8 | W4A4 | W4A4 cascade | W4A4 vs eager |
|---|---|---|---|---|---|---|
| GR00T N1.7 | 67.80 | 41.50 | 36.50 | 32.60 | 33.20 | 2.08x |
| GR00T N1.6 | 69.26 | 50.71 | 40.93 | 35.79 | 35.79 | 1.94x |
| GR00T N1.5 | 54.54 | - | 37.22 | 32.76 | 33.53 | 1.66x |
| π₀.₅ | 169.36 | - | 89.85 | 76.55 | 76.54 | 2.21x |

The last column is the deployment-path result against upstream's own serving
configuration, **not** a quantization result: roughly three quarters of it is
the graph (see the split above).

The paper's latency figure, timed with three repeats of sixty iterations after
ten warm-ups (five repeats for the GR00T eight- and four-bit arms):

| family | eager | torch.compile | float TRT | W8A8 | W4A4 | W4A4 + o/d INT8 | W4A4 vs eager |
|---|---:|---:|---:|---:|---:|---:|---:|
| GR00T N1.7 | 67.8 | 58.05 | 41.5 | 36.5 | 32.6 | 34.0 | 2.08× |
| GR00T N1.6 | 74.2 | 59.85 | 44.1 | 37.7 | 33.9 | 35.3 | 2.19× |
| GR00T N1.5 | 57.4 | 53.46 | 40.2 | 33.7 | 30.2 | 33.5 | 1.90× |
| π₀.₅ | 169.4 | 100.3 | 111.8 | 89.9 | 76.6 | 82.3 | 2.21× |

RTX 4070 Ti SUPER, batch 1. The compiled control supplies 74.7, 74.5, 63.2 and
74.5% of each eager-to-W4A4 reduction, and W4A4 removes 10.7, 10.2, 10.5 and
14.8% of the W8A8 latency. N1.7's torch.compile bar comes from a runtime whose
eager is 71.8 ms and is not used for attribution; rebuilds of N1.6 and N1.5
through this release's export path give float engines of 44.0 and 41.2 ms. The
o/d INT8 bars are a re-timing in this release's export path against the
uniform W4A4 engine of the same run.

#### Jetson AGX Orin

The paper's Orin figure: sm87, JetPack TensorRT 10.3, batch 1,
observation-to-action milliseconds.

| family | eager | torch.compile | float TRT | W8A8 | W4A4 | W4A4 + o/d INT8 | ModelOpt W8A8 SQ | ModelOpt W4A16 AWQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GR00T N1.7 | 351 | 231 | 146 | 127 | 119 | 120 | 137 | 167 |
| GR00T N1.6 | 333 | 217 | 150 | 137 | 125 | 127 | 142 | 177 |
| GR00T N1.5 | 249 | 199 | 135 | 123 | 111 | 114 | 134 | 174 |
| π₀.₅ | 861 | 861 | 229 | 226 | 172 | 203 | 211 | 326 |

Uniform W4A4 reaches 1.23, 1.20, 1.22 and 1.33× the float engine, and the
weight-only four-bit baseline is 13-30% slower than that engine on every
checkpoint. Holding `o_proj`/`down_proj` at INT8 costs 1-3 ms on GR00T and
31 ms on π₀.₅. The per-family scripts have also run on an Orin
(`docs/deploy/jetson.md` lists which), but those runs only checked that each
path completes and recorded no number.

### LIBERO success rate: all families

The paper's closed-loop campaigns: one H100 MIG 3g.40gb partition, four suites,
ten tasks, twenty initial states per task, 800 episodes per arm, seed 7;
serving-time flow noise unseeded. Success rate in percent; Wilson 95%
intervals describe each arm on its own and do not establish equivalence.

| family (K) | arm | spatial | object | goal | long | all | 95% CI |
|---|---|---:|---:|---:|---:|---:|---|
| GR00T N1.7 (8) | BF16 PyTorch | 97.5 | 100.0 | 97.5 | 90.0 | 96.25 | [94.7, 97.4] |
| | float TRT | 96.5 | 99.5 | 97.5 | 88.5 | 95.50 | [93.8, 96.7] |
| | ModelOpt W8A8 SQ | 97.0 | 98.5 | 97.5 | 90.0 | 95.75 | [94.1, 96.9] |
| | ModelOpt W4A16 AWQ | 98.0 | 99.0 | 98.0 | 90.5 | 96.38 | [94.8, 97.5] |
| | FoldQuant W8A8 | 97.0 | 99.5 | 98.0 | 87.0 | 95.38 | [93.7, 96.6] |
| | FoldQuant W4A4 | 96.5 | 99.0 | 97.0 | 89.0 | 95.38 | [93.7, 96.6] |
| | FoldQuant W4A4 + o/d INT8 | 93.5 | 99.5 | 98.0 | 89.0 | 95.00 | [93.3, 96.3] |
| GR00T N1.6 (8) | BF16 PyTorch | 97.0 | 100.0 | 97.0 | 91.5 | 96.38 | [94.8, 97.5] |
| | float TRT | 98.0 | 100.0 | 96.5 | 96.5 | 97.75 | [96.5, 98.6] |
| | ModelOpt W8A8 SQ | 98.0 | 100.0 | 97.5 | 89.5 | 96.25 | [94.7, 97.4] |
| | ModelOpt W4A16 AWQ | 94.5 | 99.5 | 96.0 | 95.0 | 96.25 | [94.7, 97.4] |
| | FoldQuant W8A8 | 95.5 | 100.0 | 94.0 | 93.5 | 95.75 | [94.1, 96.9] |
| | FoldQuant W4A4 | 95.5 | 96.5 | 96.0 | 95.0 | 95.75 | [94.1, 96.9] |
| | FoldQuant W4A4 + o/d INT8 | 95.0 | 100.0 | 99.5 | 92.0 | 96.62 | [95.1, 97.7] |
| GR00T N1.5 (1) | BF16 PyTorch | 92.0 | 95.5 | 89.5 | 68.5 | 86.38 | [83.8, 88.6] |
| | float TRT | 93.0 | 95.5 | 87.0 | 68.5 | 86.00 | [83.4, 88.2] |
| | ModelOpt W8A8 SQ | 91.0 | 93.5 | 87.5 | 63.5 | 83.88 | [81.2, 86.3] |
| | ModelOpt W4A16 AWQ | 91.5 | 96.0 | 85.0 | 72.5 | 86.25 | [83.7, 88.5] |
| | FoldQuant W8A8 | 90.5 | 98.0 | 88.5 | 71.5 | 87.12 | [84.6, 89.3] |
| | FoldQuant W4A4 | 92.5 | 96.5 | 88.0 | 72.5 | 87.38 | [84.9, 89.5] |
| | FoldQuant W4A4 + o/d INT8 | 91.5 | 97.5 | 90.5 | 68.5 | 87.00 | [84.5, 89.2] |
| π₀.₅ (5) | BF16 PyTorch | 99.5 | 98.0 | 98.0 | 90.5 | 96.50 | [95.0, 97.6] |
| | float TRT | 100.0 | 100.0 | 99.0 | 93.0 | 98.00 | [96.8, 98.8] |
| | ModelOpt W8A8 SQ | 98.5 | 98.5 | 97.0 | 93.0 | 96.75 | [95.3, 97.8] |
| | ModelOpt W4A16 AWQ | 99.5 | 98.5 | 98.5 | 95.5 | 98.00 | [96.8, 98.8] |
| | FoldQuant W8A8 | 99.5 | 99.5 | 96.5 | 94.0 | 97.38 | [96.0, 98.3] |
| | FoldQuant W4A4 | 98.5 | 99.0 | 98.0 | 93.0 | 97.12 | [95.7, 98.1] |
| | FoldQuant W4A4 + o/d INT8 | 98.5 | 99.5 | 98.5 | 94.0 | 97.62 | [96.3, 98.5] |

The FoldQuant W4A4 expert uses GPTQ and a butterfly rotation with fold-before. The N1.6
and N1.7 o/d INT8 arms were built on the dense-rotation calibration preset, whose
matched uniform-W4A4 partners scored 95.38% and 94.62%; the N1.5 arm shares the
preset of its printed W4A4 row. H100 executes four-bit operands through an INT8
lowering, so these rows are closed-loop outcomes, not native INT4 latency.

Paired over the 40 tasks with a two-sided t test and Holm correction across the
41 arm-versus-reference comparisons, **no comparison survives**: no loss is
detected at this campaign size, which does not establish equivalence.

#### A release-harness check

GR00T N1.7 BF16 through this release's `eval_libero` (upstream's
`MultiStepWrapper` rollout), one RTX 4070 Ti SUPER, 1h54m. Different harness
and GPU from the campaigns above, so this checks that the release path runs
end to end; it is not a row of that table:

| suite | successes | % |
|---|---|---|
| libero_spatial | 194/200 | 97.0 |
| libero_object | 199/200 | 99.5 |
| libero_goal | 194/200 | 97.0 |
| libero_10 | 172/200 | 86.0 |
| **all four** | **759/800** | **94.9** |
