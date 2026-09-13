# Reproducing the results

Three things can be checked, at very different cost. Start with the first: it
takes seconds, needs no GPU and no checkpoints, and it is the check that has
caught every defect found in this repository so far.

| | needs | time |
|---|---|---|
| 1. the committed records are internally consistent | a clone | seconds |
| 2. the chain runs on your machine | one family's environment, a GPU, one checkpoint | ~10 min per family |
| 3. a number is reproduced | the same checkpoint and dataset the record names | ~30 min per arm |

## 1. Check the records

```bash
git clone <repo> && cd FoldQuantVLA
python scripts/check_records.py
python scripts/results_tables.py --table drift
```

`check_records.py` reads `results/` and nothing else. It asserts what has to be
true of any honest set of arms: a float engine no less faithful to the bf16
reference than the INT8 engine built from the same graph; a stated scope; a
held-out split; no operator filesystem in the records. Each rule is there
because breaking it produced a plausible-looking wrong number — the history is
in the file.

`results_tables.py` regenerates the tables in
[`results/README.md`](../results/README.md) from the records, so a stale table
shows up as a diff rather than as a discrepancy nobody notices.

## 2. Run the chain

Each family runs in **its own virtualenv, from its own directory**. They pin
different Python and torch versions and share nothing but the `foldquant`
package, so there is no single environment that runs all six. Install one
family by following its integration README —
[N1.7](../models/groot_n1_7/foldquant_integration/README.md),
[N1.6](../models/groot_n1_6/foldquant_integration/README.md),
[N1.5](../models/groot_n1_5/foldquant_integration/README.md),
[π₀.₅](../models/pi05/foldquant_integration/README.md),
[SmolVLA](../models/smolvla/foldquant_integration/README.md),
[Evo-1](../models/evo_1/foldquant_integration/README.md) — then:

```bash
git submodule update --init third_party/cutlass
cd models/groot_n1_7 && uv sync && uv pip install -e ../..
python -m foldquant.kernels build     # compiles the plugin .so for this GPU / TensorRT
python -m foldquant.kernels status    # must print this device's sm-machine-trt slug
```

Then the smoke, which exports on 8 observations, builds the engines and scores
8 held-out ones:

```bash
export N17_MODEL=<checkpoint> GROOT_DATA=<LeRobot dataset>
scripts/smoke_family.sh groot_n1_7
```

With no arguments it tries every family and skips the ones whose paths are
unset, so holding two of six checkpoints still gives a useful report. The
environment variables it reads are listed in the script's header.

A smoke pass means the chain runs and emits a record. It does **not** mean the
paper's numbers reproduce — 8 observations is far below the 128 a GPTQ arm
needs, and the docstring in `export_foldquant` says what happens below that.

**Give it the GPU.** The script refuses to start when another process is on the
card, because contention fails in a way that looks like a defect: TensorRT
reports `execute_async_v3() failed` with no mention of memory, and engines
known to be good fail it exactly as a fresh build does. That was diagnosed here
by running a verified arm under the same conditions and watching it fail too.
`SMOKE_ALLOW_BUSY_GPU=1` overrides the check if you want it anyway.

All six pass on one RTX 4070 Ti SUPER, about 45 minutes for the set:

| family | action cos (median) | worst \|Δ\| |
|---|---|---|
| `groot_n1_7` | 0.99983 | 0.0366 |
| `groot_n1_6` | 0.99989 | 0.0293 |
| `groot_n1_5` | 0.99984 | 0.3616 |
| `pi05` | 0.99991 | 0.0328 |
| `smolvla` | 0.99869 | 2.0143 |
| `evo_1` | 0.99696 | 1.0474 |

Those are the default schemes on eight calibration observations, so they are
not the arms in `results/` and should not be compared with them. What they show
is that each family's export, engine build and verify run on this machine and
agree with its own **PyTorch** bf16 policy to the digits above — the reference
pass runs before any engine is installed, so a float engine is an arm in the
table like any other, not the thing the others are measured against.

The figure is the median, here and in `results/`; the mean and the minimum stay
in each `verify.json` and are not the number to read. A clipped action space
makes the per-observation distribution bimodal — a chunk railed on every
channel scores 1.000 by construction, one with a couple of channels railed lets
a small absolute error swing the cosine to near zero — so a mean moves with how
often the policy was railed rather than with how faithful the engine is.
`results/README.md` has the measurement behind that.

SmolVLA and Evo-1 sit lower because eight observations is far too few for their
four-bit action modules — the same effect the full protocol avoids with 128.

## 2b. The other two halves: serving, and the rollout

`smoke_family.sh` covers export → build → verify. Two things a reviewer will
ask about sit outside it, and each has its own script.

**The server.** `scripts/smoke_serve.sh` starts each family's `serve.py` on a
spare port, waits for the socket to bind, and kills it. That is the half a
robot depends on — the policy assembled, the engines installed if asked, the
port open — and it needs no simulator:

```bash
scripts/smoke_serve.sh                                  # bf16 policies
ENGINE_GROOT_N1_7=exports/w8a8/engines scripts/smoke_serve.sh groot_n1_7
```

All six bind on this desktop, N1.7 both as bf16 and over its W8A8 engines.
Readiness is decided by the socket, not by the log: each family announces
itself in its own words and matching those cost a false failure here.

**The rollout.** `scripts/smoke_eval.sh` runs one LIBERO suite at one episode
per task — ten episodes, two to nine minutes a family — through the same
`eval_libero` the sweeps use:

```bash
export N17_MODEL=<checkpoint>
scripts/smoke_eval.sh groot_n1_7
```

| family | rollout |
|---|---|
| `groot_n1_7` | 10/10 episodes, 10 tasks |
| `groot_n1_6` | 9/10 episodes, 10 tasks |
| `groot_n1_5` | 10/10 episodes, 10 tasks |
| `smolvla` | 80% over 10 episodes, 10 tasks |

**Ten episodes ranks nothing.** The question is whether the driver reaches the
family's upstream loop and writes a summary, not what the arm scores; the
published sweeps are 800 episodes. π₀.₅ and Evo-1 are skipped with that
reason: their rollout drives an upstream client from a second environment
against a running server, which is two processes and a different check.

The N1.5 release pins no LIBERO checkout — it is the operator's to supply — so
the script borrows a sibling's pinned copy and says so. `FOLDQUANT_LIBERO_DIR`
overrides.

This check earned its place twice on the day it was written. It found that
SmolVLA's `eval_libero` read `aggregated` / `per_task_infos` from upstream's
evaluator, which returns `overall` / `per_task`: every field came back `None`,
ten rollouts ran and their results were discarded, and the summary recorded
`nan%`. No published number depended on it — `results/smolvla/` carries drift
and latency, not success rate — but a reviewer running the obvious command
would have hit it first. It also caught the check's own first draft printing
`ok 0/0` for that empty rollout, which is why an empty rollout is now a
failure: in a reproducibility harness a false pass is worse than no check.

## 3. Reproduce a number

Every drift record names what it takes. Read it first:

```bash
python - <<'PY'
import json
d = json.load(open("results/groot_n1_7/w4a4/verify.json"))
print({k: d[k] for k in ("dataset_path", "episodes", "seed", "num_samples", "schemes")})
PY
```

Then run `verify` with those arguments against the arm's engines. The held-out
plan is seeded, but it is drawn from whatever episodes the dataset offers, so
the same seed over a different slice gives a different set — which is why the
record carries the dataset and the range and not only the seed.

Two things follow from that, both learned the hard way:

- **A dataset that has grown is a different dataset.** Two families' records
  were once measured over ~170 episodes and re-run over 1693; they shared one
  observation in thirty-two. If your episode count differs from the record's,
  the numbers will differ and neither is wrong.
- **An engine is compiled for one `(SM, TensorRT)` pair.** Rebuild on your own
  device; do not copy engines. The plugin `.so` is matched the same way.

## What is not here

- **LIBERO success rate** for most arms. Those sweeps run on an evaluation
  cluster; `results/README.md` carries the one arm measured on this desktop
  and marks the rest pending. That `eval_libero` runs at all is checked by
  `scripts/smoke_eval.sh` above, on ten episodes — enough to show the driver
  works, nowhere near enough to rank an arm.
- **Jetson AGX Orin latency.** See [`deploy/jetson.md`](deploy/jetson.md) for
  the procedure; the numbers are the paper's, not this repository's.
