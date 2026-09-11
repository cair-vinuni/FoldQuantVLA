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
  and marks the rest pending. `eval_libero` drives each family's own upstream
  rollout loop and runs, which `scripts/smoke_family.sh` does not check —
  a 40-episode spot check is about 7 minutes per arm.
- **Jetson AGX Orin latency.** See [`deploy/jetson.md`](deploy/jetson.md) for
  the procedure; the numbers are the paper's, not this repository's.
