# Evaluation protocol and results

Every number committed here is produced by the tools under
`models/<family>/foldquant_integration/`, which drive the **upstream**
evaluation harness of that family — the same rollout loop, wrappers,
episode caps and success definition the model's authors used to report
their own numbers. Nothing in this repository re-implements an evaluator.

That fidelity has a consequence worth stating before any table is read: each
family is reproduced under *its own* loop, and those loops differ (see the
settle-step note under Success rate). These are therefore **per-family
reproductions** — each answering "what does this arm do to this policy, run
the way its authors run it". A cross-family success-rate comparison is a
different measurement and needs one harness applied uniformly to all four,
which these drivers deliberately are not. Where a paper table compares
families in one column, it comes from such a uniform harness and says so.

## Arms

| arm | LLM | action expert | notes |
|---|---|---|---|
| bf16 | PyTorch | PyTorch | upstream checkpoint, no export |
| float TRT | TensorRT bf16 | TensorRT bf16 | upstream pipeline, unquantized |
| W8A8 | `w8a8_sr` | `w8a8_sh` | dynamic per-row INT8, folded |
| W4A4 | `w4a4_srg` | `w4a4_shg` | INT4 weights and activations, folded, GPTQ |
| W4A4 cascade | `w4a4_srg` | `w4a4_shg` (`--cascade`) | action expert calibrated under the quantized LLM |

For GR00T N1.7 the untouched modules of every TensorRT arm (vision tower, VL
self-attention, state / action encoders, action decoder) are the upstream float
engines, and the float TRT arm is upstream's full pipeline
(`n17_full_pipeline`) with nothing quantized — eight engines, text tower
included.

**N1.6's float arm has been two different things, and the records say which.**
Upstream's `scripts/deployment/export_onnx_n1d6.py` exports the DiT and nothing
else, so the first float arm built here was upstream's pipeline as upstream
ships it: DiT under TensorRT, text tower in eager PyTorch. That is the arm the
latency table below still uses, and its `components` field says `["dit"]`; the
timings show the scope directly, its backbone matching eager to within a fifth
of a millisecond while the quantized arms' backbone falls to 17–18 ms.

`--llm-scheme float` traces a float text tower of its own, so the drift table's
float row is now the full-scope arm — both modules under TensorRT, nothing
quantized — and reads `schemes {"llm": "float", "dit": "float"}` in its record.
The two arms answer different questions and are not interchangeable: check the
`components` or `schemes` field of the record before comparing a number to
another family's.

The consequence is a reading trap, and it is the reason the tables quote
speedup against **eager PyTorch** rather than against the float TRT arm. N1.6's
float baseline leaves the larger of the two modules unaccelerated, so a
float-relative ratio flatters it: measured on one RTX 4070 Ti SUPER, W4A4 is
1.42x its float arm on N1.6 against 1.27x on N1.7, while against eager the
order reverses to 1.94x and 2.08x. Only the second pair compares the two
families.

GR00T N1.5's upstream engines use a different DiT contract
(fp16, `sa_embs`/`vl_embs` inputs), and openpi ships no TensorRT path at
all, so for those two families only the two FoldQuant
engines run under TensorRT, the rest of the policy stays in PyTorch, there is
no float TRT arm, and drift is read against the bf16 PyTorch policy alone.

The second quantized module is the family's action generator, whatever its
shape: a DiT for GR00T, a Gemma-300M expert for pi.

## Calibration

128 observations, seeded (`--seed 0`), episode-balanced (round-robin over a
shuffled episode order, one uniform step per visit), drawn through the upstream
data path from the deployment distribution — for LIBERO, all four suites, not
one. The export manifest (`foldquant_export.json`) records every `(episode,
step)` used.

## Drift (`verify`)

Held-out observations from episodes the calibration never saw; per observation
the flow-matching noise is seeded identically for the PyTorch and the engine
pass. Reported: cosine (median, with mean and min kept in the record) of the LLM output the action head consumes
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

A count of "flips" is only readable against the family's action scale, and the
scales differ: a saturated channel is |Δ| ≈ 2 where actions run [-1, 1], and
the same absolute error means something else where they do not. So the column
that travels between families is the **median worst-|Δ|**, recorded per
observation as `action_worst`, rather than a count above a fixed threshold.

The worst channel's *name* travels less far than that. GR00T returns a named
action dict, so its records carry `gripper`, `x`, `y`, `z`; π₀.₅ returns an
unnamed vector, so `action_worst.label` is null there and only the index and
the magnitude are available. That is a property of what
those policies emit, not a gap in the records, and it is why the channel name
appears in this file only where a GR00T family is under discussion.

### The drift figure is the median

The tables report the **median** action cosine and nothing else. The mean and
the minimum stay in every `verify.json`, and they are not the number to read.

A VLA action space is clipped. A chunk is often railed on every channel at
once, and a cosine between two railed vectors compares their signs: it returns
1.000 whatever the arm did. A chunk with two channels railed and the rest near
zero is the opposite — a small absolute error on a near-zero channel rotates
the vector far, and the cosine falls to near nothing while the arm's behaviour
barely changes. The distribution is bimodal, and a mean averages across the two
modes, so it moves with how often the policy was railed rather than with how
faithful the engine was.

Measured, on a GR00T N1.6 Bridge W4A4 arm over 32 held-out observations: 20
observations have `|action| ≈ 2.830` (norm² = 8.01, `max|action|` exactly
1.0000 — eight channels at the rail) and every one of them scores above 0.998.
Six have `|action| ≈ 1.416` (norm² = 2.01 — two channels at the rail) and they
carry almost the whole deficit, down to 0.022. Mean 0.842, median 0.9985. The
same arm's SimplerEnv success rate, 200 episodes over seven tasks, is 0.612
against bf16's 0.622.

The median is not a predictor of success rate — on that checkpoint the arm with
the best median has the lowest success rate, and the ordering is close to
reversed. What it does is agree with the success rate on the question the drift
table is asking: is the typical action the same action? The mean, on the same
records, says an arm is broken that the closed-loop measurement shows is not.

Per-observation drift is in each arm's `verify.json`, so a tail can be examined
rather than inferred.

`--components` (`llm`, and the family's action module) installs one engine and
leaves the other in PyTorch, which is how a drift figure is attributed to a
seam rather than to the sum of both. GR00T N1.7 does the same through
upstream's own `--mode` (`vit_llm_only`, `dit_only`).

That attribution is by seam, not by observation. **Damage to the backbone
output does not predict which observation's chunk breaks.** Each `verify.json`
sample carries the worst per-position cosine of the representation the action
module consumes — `backbone_token_cos_min` for the three GR00T families,
`kv_stack_position_cos_min` for π₀.₅ — and over the 32 held-out
observations of each W4A4 arm its Pearson correlation with the action cosine
is +0.30 (N1.7), +0.16 (N1.6), +0.08 (N1.5), +0.31 (π₀.₅). Positive in every
family, weak in all four: in every one of them the single most-damaged prefix
decodes to an action cosine of 0.9987 or
better. The two depths measure the same arm; they are not two views of the
same observations, and a per-observation reading across them is not supported
by these files — see [`groot_n1_7/README.md`](groot_n1_7/README.md) for the
case that prompted the check. `scripts/results_tables.py --table correlation`
regenerates the four figures from the committed records alone.

## Success rate (`eval_libero`)

LIBERO `spatial`, `object`, `goal`, `10` — 10 tasks each, `n` episodes per
task, each family under its own upstream rollout loop, reported per suite and
pooled:

- **GR00T N1.7 / N1.6** — upstream `MultiStepWrapper` (8-step action chunks,
  504-step cap, terminate on success) and the upstream unseeded `reset()`. This
  harness takes **no settle steps**: the policy acts on the first frame after
  the reset. The other two families wait first (N1.5 and π₀.₅ ten steps)
  issuing LIBERO's own no-op,
  `[0, 0, 0, 0, 0, 0, -1]`, whose last channel holds the gripper open. That
  asymmetry is upstream's, not this repository's — each family runs the loop
  its authors published — but it is a real difference in starting conditions
  and success rates should not be read across families as if it were absent.
  TensorRT arms run one environment at a time (engines pin batch 1); PyTorch
  arms may batch — the protocol is otherwise identical.
- **GR00T N1.5** — upstream's `examples/Libero` client loop (`num_steps_wait`
  no-op steps, per-suite step budgets, 8-step chunks), run in-process around
  the served policy.
- **π₀.₅** — upstream's `examples/libero/main.py` unchanged, against
  `foldquant_integration.serve` (the upstream websocket server over the
  engines): `replan_steps 5`, per-suite step budgets (220 / 280 / 300 / 520),
  50 trials per task, `seed 7`.
## Latency (`benchmark`)

Upstream `benchmark_inference.py` for GR00T N1.7 / N1.6, and the
integration's own component timer where the release ships none (N1.5,
π₀.₅) — 5 warm-up, 20 timed chunks by default, end-to-end
from observation to action chunk, on

- an RTX 4070 Ti SUPER (sm89, 16 GB, TensorRT 10.15), and
- a Jetson AGX Orin 64 GB (sm87, JetPack TensorRT 10.3),

float TRT and FoldQuant arms with identical pipelines apart from the two
quantized engines. Where there is no float TRT arm the reference is the
upstream PyTorch serving configuration: bf16 eager for N1.5 — which serves
eagerly upstream — and for π₀.₅ both eager and
upstream's `torch.compile(max-autotune)` default (the FoldQuant seams need the
eager model, so the compiled arm is timed end to end only).

**Every latency figure quoted anywhere in this repository is a median.** The
records carry more than that on one line — upstream's N1.7 log prints
`median=`, `mean= ± sd`, `min` and `max` together, and the component rows
beneath it are medians already — so a cell copied from the wrong field of the
right line is an easy and invisible error. Medians for the same reason the
drift tables report them: 20 timed chunks have a tail, and a mean moves with
it. Where a family's records offer both, the tables take `median=`.

**A speedup against eager is not a quantization speedup**, and on these
records most of it is not. The float TRT arm is the control that separates the
two — the same graph, compiled, with nothing quantized — and where one exists
the split is consistent:

| module | eager | float engine | W4A4 | graph | precision |
|---|---|---|---|---|---|
| N1.7 text tower | 27.47 | 16.84 | 13.87 | 10.63 (78%) | 2.97 (22%) |
| N1.7 action head | 39.47 | 21.76 | 15.84 | 17.71 (75%) | 5.92 (25%) |
| N1.6 action head | 38.20 | 19.86 | 14.60 | 18.33 (78%) | 5.26 (22%) |

Each difference above is taken within one process wherever it can be: upstream's
N1.7 script re-times eager on every invocation, so the graph term pairs the
float arm with the eager measured in *its own* log, and only the float-to-W4A4
term crosses processes. Pairing an arm with a different log's eager moves the
split by a few points (73-78% across the four N1.7 logs, whose eager action
head spans 37.74-39.47 ms) without changing the reading.

Three module-level measurements across two families, landing at 75-78% under
the pairing above and 73-78% across every pairing:
**roughly three quarters of the latency a FoldQuant arm saves against eager is
TensorRT compiling the graph, and roughly one quarter is the precision.** That
is not an argument against the arms — it is what the deployment path is worth
end to end — but a number quoted against eager measures both, and only the
float arm tells them apart.

A float engine is not the only control that isolates the graph. Compiling the
model with nothing quantized does the same job by another route, and π₀.₅ has
such an arm already — upstream's `torch.compile(max-autotune)` default. It
recovers 69.07 ms of the 92.81 ms between eager and W4A4: **74.4%**, landing
inside the 73-78% the float engines give on a different family through a
different mechanism. Two unrelated ways of holding precision fixed agree on
roughly three quarters.

No family outside GR00T has a float engine for its action module, so none of
their tables makes the split that way; π₀.₅ makes it with a compile-only
arm instead, and N1.5 has neither. What every row
can bound is the last step alone, 8-bit to 4-bit, which no graph change
explains:

| | eager → W4A4 saved | of which 8→4 bit |
|---|---|---|
| N1.5 action head | 8.88 ms | 2.74 ms (31%) |
| π₀.₅ denoise loop | 48.70 ms | 3.46 ms (7%) |

The remainder of each row is graph and 8-bit quantization together. For N1.5
these records do not separate the two — which is not evidence that precision
did the work, only that nothing here isolates it.

## Results

Measured outputs are committed beside this file as `<family>/<arm>/`:
`verify.json`, `libero/summary.json`, `benchmark.log`, and the arm's
`foldquant_export.json`. Tables in this file are regenerated from those files,
so every cell traces to a committed artifact rather than to a transcript.

Where a family's records need a note of their own — a reading the files do
not support, a correction to something already pushed — it sits
in that family's folder: [`groot_n1_7/README.md`](groot_n1_7/README.md).

Each integration README carries a **smoke check** — a 16-observation
calibration, 8 held-out observations, one RTX 4070 Ti SUPER — that exercises
the whole export → build → verify → benchmark chain on that family. Those
numbers are sanity gates, not the paper's: the tables below are filled from
the 128-observation exports and the full LIBERO sweeps.

`scripts/check_records.py` checks the invariants those records have to satisfy —
a float arm no less faithful than the INT8 one built from the same graph, a stated
scope, a held-out split, no operator paths. Each rule is there because breaking it
produced a wrong number that looked plausible. Run it on a clone; it needs nothing
but `results/`.

Every table in this section is emitted by `scripts/results_tables.py`, which
reads only the committed records — no GPU, no engines, no upstream
environment — so a stale cell shows up as a diff rather than as a discrepancy
nobody notices.

### Drift — all families

Held-out action cosine per arm, 32 observations, seeded
(`<family>/<arm>/verify.json`). The PyTorch repeatability floor is 1.000000
under the same seeds in every family, so the deficits below are the arm's, not
the sampler's.

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

Read the median beside the mean: these distributions have tails, and on the
families where W4A4 breaks it is a minority of observations that carry the
deficit. Worst \|Δ\| is per family's action scale — 1.000 is a saturated 0/1
gripper on GR00T, ~2.0 is a saturated channel where actions run [-1, 1].

**Read the float row before reading the quantized ones.** It is the same graph
compiled with nothing quantized, so whatever it already costs is not
quantization. Every family has one now.

That row is only a control if it is built the way the quantized arms are. It
was not, at first: a traced float graph was compiled weakly typed, so TensorRT
chose a precision per layer and ran some of them in fp32 — *more* exact than
the bf16 reference the arm is scored against. π₀.₅'s float engine then read
0.98256 mean against an INT8 engine's 1.00000, which is not a thing a float
engine can honestly do. Built STRONGLY_TYPED, honouring the ONNX's own bf16
dtypes, the same engine reads 1.00000. The lesson generalises past this bug:
a control that differs from the arm in two ways measures neither.

### Latency — all families

End-to-end median ms per action chunk, one RTX 4070 Ti SUPER (sm89, TensorRT
10.15), 20 timed chunks after 5 warm-up. GR00T N1.7 re-times eager in every
arm's log; the column shows the eager from the W4A4 log, the pairing the split
above uses.

| family | eager | float TRT | W8A8 | W4A4 | W4A4 cascade | W4A4 vs eager |
|---|---|---|---|---|---|---|
| GR00T N1.7 | 67.80 | 41.50 | 36.50 | 32.60 | 33.20 | 2.08x |
| GR00T N1.6 | 69.26 | 50.71 | 40.93 | 35.79 | 35.79 | 1.94x |
| GR00T N1.5 | 54.54 | - | 37.22 | 32.76 | 33.53 | 1.66x |
| π₀.₅ | 169.36 | - | 89.85 | 76.55 | 76.54 | 2.21x |

The last column is the deployment-path result against upstream's own serving
configuration, **not** a quantization result — see the graph-versus-precision
split above, where roughly three quarters of it is the graph.

**These are this release's records, and they are not the paper's figure.** The
paper times three repeats of sixty iterations after ten warm-ups (five repeats
for the GR00T eight- and four-bit arms), and re-timed N1.6 and N1.5 in the
framework runtime with the configurations of its success campaign. N1.7 and
π₀.₅ agree with it to the digit; N1.6 and N1.5 do not, and the gap is not
rounding:

| family | eager | torch.compile | TRT bf16 | W8A8 | W4A4 | W4A4 + o/d INT8 | W4A4 vs eager |
|---|---:|---:|---:|---:|---:|---:|---:|
| GR00T N1.7 | 67.8 | 58.05 | 41.5 | 36.5 | 32.6 | 34.0 | 2.08× |
| GR00T N1.6 | 74.2 | 59.85 | 44.1 | 37.7 | 33.9 | 35.3 | 2.19× |
| GR00T N1.5 | 57.4 | 53.46 | 40.2 | 33.7 | 30.2 | 33.5 | 1.90× |
| π₀.₅ | 169.4 | 100.3 | 111.8 | 89.9 | 76.6 | 82.3 | 2.21× |

*The paper's latency figure, RTX 4070 Ti SUPER, batch 1.* The compiled control
supplies 74.7, 74.5, 63.2 and 74.5% of each eager-to-W4A4 reduction, and W4A4
removes 10.7, 10.2, 10.5 and 14.8% of the W8A8 latency. N1.7's torch.compile
bar comes from a runtime whose eager is 71.8 ms and is not used for attribution;
rebuilds of N1.6 and N1.5 through this release's export path give float
engines of 44.0 and 41.2 ms. The o/d INT8 bars are a re-timing in this
release's export path against the uniform W4A4 engine of the same run.

A reproduction of the N1.6 and N1.5 rows through `benchmark` will land on this
release's records above, not on the paper's, until those two families are
re-timed here under the paper's protocol.

#### Jetson AGX Orin

The paper's Orin figure: sm87, JetPack TensorRT 10.3, batch 1,
observation-to-action milliseconds.

| family | eager | torch.compile | TRT bf16 | W8A8 | W4A4 | W4A4 + o/d INT8 | ModelOpt W8A8 SQ | ModelOpt W4A16 AWQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GR00T N1.7 | 351 | 231 | 146 | 127 | 119 | 120 | 137 | 167 |
| GR00T N1.6 | 333 | 217 | 150 | 137 | 125 | 127 | 142 | 177 |
| GR00T N1.5 | 249 | 199 | 135 | 123 | 111 | 114 | 134 | 174 |
| π₀.₅ | 861 | 861 | 229 | 226 | 172 | 203 | 211 | 326 |

Uniform W4A4 reaches 1.23, 1.20, 1.22 and 1.33× the float engine, and the
weight-only four-bit baseline is 13–30% slower than that engine on every
checkpoint. Holding `o_proj`/`down_proj` at INT8 costs 1–3 ms on GR00T and
31 ms on π₀.₅. The per-family scripts have been run on an Orin as well —
`docs/deploy/jetson.md` lists which — but those runs checked that each path
completes and recorded no number.

### LIBERO success rate — all families

The paper's closed-loop campaigns: one H100 MIG 3g.40gb partition, four suites,
ten tasks, twenty initial states per task, 800 episodes per arm, seed 7;
serving-time flow noise unseeded. Success rate in percent; Wilson 95%
intervals describe each arm on its own and do not establish equivalence.

| family (K) | arm | spatial | object | goal | long | all | 95% CI |
|---|---|---:|---:|---:|---:|---:|---|
| GR00T N1.7 (8) | BF16 PyTorch | 97.5 | 100.0 | 97.5 | 90.0 | 96.25 | [94.7, 97.4] |
| | TRT bf16 | 96.5 | 99.5 | 97.5 | 88.5 | 95.50 | [93.8, 96.7] |
| | ModelOpt W8A8 SQ | 97.0 | 98.5 | 97.5 | 90.0 | 95.75 | [94.1, 96.9] |
| | ModelOpt W4A16 AWQ | 98.0 | 99.0 | 98.0 | 90.5 | 96.38 | [94.8, 97.5] |
| | FQ W8A8 | 97.0 | 99.5 | 98.0 | 87.0 | 95.38 | [93.7, 96.6] |
| | FQ W4A4 | 96.5 | 99.0 | 97.0 | 89.0 | 95.38 | [93.7, 96.6] |
| | FQ W4A4 + o/d INT8 | 93.5 | 99.5 | 98.0 | 89.0 | 95.00 | [93.3, 96.3] |
| GR00T N1.6 (8) | BF16 PyTorch | 97.0 | 100.0 | 97.0 | 91.5 | 96.38 | [94.8, 97.5] |
| | TRT bf16 | 98.0 | 100.0 | 96.5 | 96.5 | 97.75 | [96.5, 98.6] |
| | ModelOpt W8A8 SQ | 98.0 | 100.0 | 97.5 | 89.5 | 96.25 | [94.7, 97.4] |
| | ModelOpt W4A16 AWQ | 94.5 | 99.5 | 96.0 | 95.0 | 96.25 | [94.7, 97.4] |
| | FQ W8A8 | 95.5 | 100.0 | 94.0 | 93.5 | 95.75 | [94.1, 96.9] |
| | FQ W4A4 | 95.5 | 96.5 | 96.0 | 95.0 | 95.75 | [94.1, 96.9] |
| | FQ W4A4 + o/d INT8 | 95.0 | 100.0 | 99.5 | 92.0 | 96.62 | [95.1, 97.7] |
| GR00T N1.5 (1) | BF16 PyTorch | 92.0 | 95.5 | 89.5 | 68.5 | 86.38 | [83.8, 88.6] |
| | TRT bf16 | 93.0 | 95.5 | 87.0 | 68.5 | 86.00 | [83.4, 88.2] |
| | ModelOpt W8A8 SQ | 91.0 | 93.5 | 87.5 | 63.5 | 83.88 | [81.2, 86.3] |
| | ModelOpt W4A16 AWQ | 91.5 | 96.0 | 85.0 | 72.5 | 86.25 | [83.7, 88.5] |
| | FQ W8A8 | 90.5 | 98.0 | 88.5 | 71.5 | 87.12 | [84.6, 89.3] |
| | FQ W4A4 | 92.5 | 96.5 | 88.0 | 72.5 | 87.38 | [84.9, 89.5] |
| | FQ W4A4 + o/d INT8 | 91.5 | 97.5 | 90.5 | 68.5 | 87.00 | [84.5, 89.2] |
| π₀.₅ (5) | BF16 PyTorch | 99.5 | 98.0 | 98.0 | 90.5 | 96.50 | [95.0, 97.6] |
| | TRT bf16 | 100.0 | 100.0 | 99.0 | 93.0 | 98.00 | [96.8, 98.8] |
| | ModelOpt W8A8 SQ | 98.5 | 98.5 | 97.0 | 93.0 | 96.75 | [95.3, 97.8] |
| | ModelOpt W4A16 AWQ | 99.5 | 98.5 | 98.5 | 95.5 | 98.00 | [96.8, 98.8] |
| | FQ W8A8 | 99.5 | 99.5 | 96.5 | 94.0 | 97.38 | [96.0, 98.3] |
| | FQ W4A4 | 98.5 | 99.0 | 98.0 | 93.0 | 97.12 | [95.7, 98.1] |
| | FQ W4A4 + o/d INT8 | 98.5 | 99.5 | 98.5 | 94.0 | 97.62 | [96.3, 98.5] |

The FQ W4A4 expert uses GPTQ and a butterfly rotation with fold-before. The N1.6
and N1.7 o/d INT8 arms were built on the dense-rotation calibration preset, whose
matched uniform-W4A4 partners scored 95.38% and 94.62%; the N1.5 arm shares the
preset of its printed W4A4 row. H100 executes four-bit operands through an INT8
lowering, so these rows are closed-loop outcomes, not native INT4 latency.

Paired over the 40 tasks with a two-sided t test and Holm correction across the
41 arm-versus-reference comparisons, **no comparison survives**. That states no
loss is detected at this campaign size; it does not establish equivalence.

Per-episode outcomes behind these rows are not yet in this repository.

#### A release-harness check

GR00T N1.7 BF16 through this release's `eval_libero` (upstream's
`MultiStepWrapper` rollout), one RTX 4070 Ti SUPER, 1h54m — a different
harness and GPU from the campaigns above, so it is a check that the release
path runs end to end, not a row of that table:

| suite | successes | % |
|---|---|---|
| libero_spatial | 194/200 | 97.0 |
| libero_object | 199/200 | 99.5 |
| libero_goal | 194/200 | 97.0 |
| libero_10 | 172/200 | 86.0 |
| **all four** | **759/800** | **94.9** |
