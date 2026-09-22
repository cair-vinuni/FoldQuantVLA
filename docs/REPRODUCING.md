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

`check_records.py` checks float versus W8A8 drift, declared scope, held-out
status, sample counts, and local paths in `results/`. It needs no GPU or model
dependencies. `results_tables.py` prints tables derived from the records for
comparison with [`results/README.md`](../results/README.md).

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
values. See the [measurement protocol](../results/README.md) for interpretation.

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

## Reproduce a measurement

Inspect the target record before running `verify`:

```bash
python - <<'PY'
import json
from pathlib import Path

record = json.loads(Path("results/groot_n1_7/w4a4/verify.json").read_text())
keys = ("dataset_path", "episodes", "seed", "num_samples", "schemes")
print({key: record[key] for key in keys})
PY
```

Match the checkpoint, dataset, episode range, seed, sample count, and schemes.
The sampling plan depends on the available episodes, so a changed dataset can
produce different observations with the same seed. Rebuild engines and plugin
libraries for the target device and TensorRT version.

The [results guide](../results/README.md) includes the paper's Jetson AGX Orin
figure. The [Orin deployment checks](deploy/jetson.md) recorded completion only,
with no separate latency or accuracy measurements.
