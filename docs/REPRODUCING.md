# Reproducing the results

Choose a check based on the resources available:

| Check | Requirements | Approximate time |
|---|---|---|
| Committed record consistency | Repository clone | Seconds |
| Export, build, and verification | Family environment, GPU, checkpoint, dataset | 10 min per family |
| Published measurement | Checkpoint and dataset matching the record | 30 min per arm |

## Check the records

Run from the repository root:

```bash
python scripts/check_records.py
python scripts/results_tables.py --table drift
```

`check_records.py` checks declared scope, held-out status, sample counts, and
local paths in `results/`. It needs no GPU or model
dependencies. `results_tables.py` prints the drift and latency tables from
the records for comparison with the paper.

## Run export, build, and verification

Install a family's environment using its integration guide:
[N1.7](../models/groot_n1_7/foldquant_integration/README.md),
[N1.6](../models/groot_n1_6/foldquant_integration/README.md),
[N1.5](../models/groot_n1_5/foldquant_integration/README.md), or
[π₀.₅](../models/pi05/foldquant_integration/README.md).
Each family pins its own Python and torch versions.

For GR00T N1.7 on a workstation, start at the repository root:

```bash
git submodule update --init third_party/cutlass
cd models/groot_n1_7
uv sync
source .venv/bin/activate
uv pip install -e ../..
python -m foldquant.kernels build
python -m foldquant.kernels status
cd ../..

export N17_MODEL=/path/to/checkpoint
export GROOT_DATA=/path/to/lerobot_dataset
scripts/smoke_family.sh groot_n1_7
```

On Orin, follow the [Jetson environment setup](deploy/jetson_serve.md)
instead. Plugin status should report the current device's
`sm<CC>-<machine>-trt<major>.<minor>` target.

The smoke test uses eight calibration and eight held-out observations.
With no family argument, it tries all families and skips those with unset
checkpoint or dataset paths. The script header lists the environment variables.

Use an idle GPU: contention can cause TensorRT execution failures even with
valid engines. `SMOKE_ALLOW_BUSY_GPU=1` bypasses the occupancy check.

Recorded smoke results on RTX 4070 Ti SUPER:

| Family | Median action cosine | Worst \|Δ\| |
|---|---|---|
| `groot_n1_7` | 0.99983 | 0.0366 |
| `groot_n1_6` | 0.99989 | 0.0293 |
| `groot_n1_5` | 0.99984 | 0.3616 |
| `pi05` | 0.99991 | 0.0328 |

These checks confirm that the pipeline runs. Published arms use 128 calibration
observations, so the smoke results are not directly comparable to `results/`.
All arms are compared against the PyTorch BF16 policy before engine installation.
Tables report median action cosine; the records also retain mean and minimum
values. The paper describes the measurement protocol.

## Check serving and LIBERO

`smoke_serve.sh` starts a policy server, waits for its socket to bind, and
stops it. This checks startup only; it does not send an inference request.

```bash
scripts/smoke_serve.sh
ENGINE_GROOT_N1_7=exports/w8a8/engines scripts/smoke_serve.sh groot_n1_7
```

`smoke_eval.sh` runs one LIBERO suite with one episode per task through the
family's `eval_libero` entry point:

```bash
export N17_MODEL=/path/to/checkpoint
scripts/smoke_eval.sh groot_n1_7
```

| Family | Recorded rollout |
|---|---|
| `groot_n1_7` | 10/10 episodes, 10 tasks |
| `groot_n1_6` | 9/10 episodes, 10 tasks |
| `groot_n1_5` | 10/10 episodes, 10 tasks |

Ten episodes check rollout completion and summary writing; published sweeps use
800 episodes. Empty rollouts fail. π₀.₅ is skipped because its LIBERO client runs
in a separate environment against a server. N1.5 uses a sibling's pinned LIBERO
checkout unless `FOLDQUANT_LIBERO_DIR` is set.

## Reproduce the closed-loop campaign (protocol P3)

The paper's LIBERO table is protocol P3: four suites, ten tasks each, episode
`i` of every task from LIBERO's stored initial state `i` for `i = 0..19`, ten
no-op steps with the gripper open after the state is set, 520 environment
steps per episode, eight actions executed per policy call (five on π₀.₅),
terminate on success; 800 episodes per arm. `--protocol p3` selects it in every
family's `eval_libero`, and `summary.json` records the initial state and
outcome of every episode:

```bash
# GR00T N1.7 (N1.6: the same command in models/groot_n1_6)
MUJOCO_GL=egl python -m foldquant_integration.eval_libero --protocol p3 \
    --model-path <4-suite checkpoint> --engine-dir exports/<arm>/engines --output <out>
# GR00T N1.5
MUJOCO_GL=egl python -m foldquant_integration.eval_libero --protocol p3 \
    --model-path <ckpt> --embodiment-tag new_embodiment --denoising-steps 8 \
    --engine-dir exports/<arm>/engines --output <out>
# π₀.₅ (the client runs in examples/libero/.venv)
MUJOCO_GL=egl python -m foldquant_integration.eval_libero --protocol p3 \
    --checkpoint-dir <ckpt> --engine-dir exports/<arm>/engines \
    --client-python examples/libero/.venv/bin/python --output <out>
```

Omit `--engine-dir` for the BF16 PyTorch arm. `--max-episode-steps 720` with
`--protocol p3` on the NVIDIA per-suite N1.7 checkpoints is the paper's
Table I setting. Flow-matching noise is not seeded at serve time, so a rerun
reproduces the protocol, not the episode-by-episode outcomes; the paper's
paired tests at this size do not separate the arms, and neither will a rerun.

A `summary.json` carries a fingerprint of the checkpoint, the engine
directory and the protocol. An interrupted sweep resumes into the same
`--output` only when that fingerprint matches; a different checkpoint, engine
build, episode count or step cap is refused, and `--no-resume` starts over.

## Reproduce a measurement

Inspect the target record before running `verify`:

```bash
python - <<'PY'
import json
from pathlib import Path

record = json.loads(Path("results/groot_n1_7/w4a4/verify.json").read_text())
keys = ("dataset_path", "seed", "num_samples", "held_out", "schemes")
print({key: record[key] for key in keys})
print("held-out episodes:", sorted({s["episode"] for s in record["samples"]}))
PY
```

Match the checkpoint, dataset, episode set, seed, sample count, and schemes.
The sampling plan depends on the available episodes, so a changed dataset can
produce different observations with the same seed. Rebuild engines and plugin
libraries for the target device and TensorRT version.

The paper reports the Jetson AGX Orin latency. The
[Orin deployment checks](deploy/jetson.md) recorded completion only,
with no separate latency or accuracy measurements.
