# Installing and testing a FoldQuant arm on Jetson AGX Orin

An engine is compiled for exactly one `(GPU architecture, TensorRT version)`
pair, and so is the plugin `.so`. An engine built on an x86 workstation
(`sm89-x86_64-trt10.15`) **will not load** on an Orin (`sm87-aarch64-trt10.3`):
the locator matches `sm<CC>-<machine>-trt<MAJ>.<MIN>` exactly, tolerating only
an older minor within the same major.

So: **ONNX crosses, binaries do not.** Export on the workstation, build and
test on the board.

> The per-family scripts (`scripts/smoke_family.sh`, `smoke_serve.sh`,
> `smoke_eval.sh`, `bench_all.sh`) have been run on a Jetson AGX Orin
> (JetPack 6.2, CUDA 12.6, TensorRT 10.3, Python 3.10) for GR00T N1.7, N1.6,
> N1.5 and π₀.₅ (see [Status on Orin](#status-on-orin)). No latency or accuracy figure from those runs
> is recorded: they check that each path completes, not what it measures.

## 1. Export, on the workstation

The family's ordinary export, stopped before the build. Calibrating on the
Orin would work (64 GB unified memory is enough) but GPTQ is CPU-heavy, so this
is about wall-clock, not capability.

```bash
cd models/groot_n1_7 && source .venv/bin/activate
CK=<checkpoint directory>      # a local path: a hub id would record its org
DS=<LeRobot dataset>

# the upstream float graphs: every arm shares them
python scripts/deployment/build_trt_pipeline.py \
    --model-path "$CK" --dataset-path "$DS" --embodiment-tag <TAG> \
    --output-dir exports/float --steps export

# the FoldQuant graphs
python -m foldquant_integration.export_foldquant \
    --model-path "$CK" --dataset-path "$DS" --embodiment-tag <TAG> \
    --num-calib 128 --seed 0 \
    --llm-scheme w8a8_sr --dit-scheme w8a8_sh \
    --output-dir exports/w8a8
```

`--steps export` omits the build deliberately. Keep `--num-calib` at 128 or
more for a GPTQ (`_g`) arm.

## 2. Copy to the board

```bash
rsync -avP exports/float/onnx  orin:<repo>/models/groot_n1_7/exports/float/
rsync -avP exports/w8a8/onnx   orin:<repo>/models/groot_n1_7/exports/w8a8/
```

Copy whole directories: a large graph keeps its weights in a sibling
`.onnx.data` file and the `.onnx` alone is useless without it. Budget ~15-20 GB
free for one arm (W8A8 engines are 4.5 GB, their ONNX 2.0 GB).

**The checkpoint and the dataset have to be on the board too**: `serve` builds
the upstream policy around the engines, and step 5 opens the dataset to score
against it.

## 3. Build the plugin library, on the Orin

```bash
git submodule update --init third_party/cutlass
cd models/groot_n1_7
# environment: the Orin recipe, not a plain `uv sync` -- see jetson_serve.md, section 1
uv sync --project scripts/deployment/orin --no-install-project
export PYTHONPATH=$PWD/../..:$PWD
TENSORRT_ROOT=/usr python -m foldquant.kernels build
python -m foldquant.kernels status
```

A plain `uv sync` resolves the family's top-level `pyproject.toml`, whose torch
comes from PyPI and does not run on the Orin's GPU. The full no-sudo setup is in
[`jetson_serve.md`](jetson_serve.md), which also covers building everything on
the board and serving it in one script.

`status` must print an Orin slug (`sm87-aarch64-trt10.3`, or whatever
TensorRT JetPack installed. An x86 slug means you are on the wrong machine.
Headers are picked from the **installed** TensorRT major
(`third_party/tensorrt-headers/include-trt10/` for TensorRT 10); compiling
against the wrong one fails at plugin load, not at build.

## 4. Build the engines, on the Orin

```bash
python -m foldquant_integration.build_engines \
    --onnx-dir exports/w8a8/onnx --engine-dir exports/w8a8/engines \
    --float-onnx-dir exports/float/onnx
```

`--float-onnx-dir`, not `--float-engine-dir`: the board has no float engines
yet, so the untouched components are **built** from ONNX rather than copied.

## 5. Test before serving

```bash
python -m foldquant_integration.verify \
    --model-path "$CK" --dataset-path "$DS" --embodiment-tag <TAG> \
    --engine-dir exports/w8a8/engines --num-samples 32 --seed 42
```

Not optional on a new board. An engine built against the wrong headers, or from
a graph whose `.onnx.data` did not transfer, loads without complaint and
returns wrong actions, with no error to catch. `verify` scores the engines
against the bf16 policy on the same board, so a bad build shows up as a number
before any hardware moves.

## 6. Serve

```bash
python -m foldquant_integration.serve \
    --model-path "$CK" --embodiment-tag <TAG> \
    --engine-dir exports/w8a8/engines \
    --mode n17_full_pipeline --port 5555
```

Drop `--engine-dir` for the bf16 reference arm. The wire protocol is upstream's
(a robot client points at the Orin by changing a host and a port); each
family's own repository documents its client.

`FOLDQUANT_TRT_CUDA_GRAPH=1` enables CUDA-graph replay, **but not on this
family.** The flag is read by `foldquant.runtime.engine.TensorRTEngine`, and
GR00T N1.7 is the one integration that does not use it: the release ships its
own seven-component pipeline swap, so `serve`, `verify` and `benchmark` all go
through upstream's `trt_model_forward` / `trt_torch.Engine`, which enqueues
directly. Setting the variable here is inert: no graph is captured and the
latency is unchanged (measured on a served SO101 W8A8 arm: 43.5 ms median round
trip either way).

The other three families do route through the runtime, and there the flag pays
most for engines called many times per chunk (π₀.₅'s expert runs 10×). Launch overhead is relatively larger on an Orin than on x86, so it
is worth measuring there even where it did not pay on a workstation
(`results/FLOAT_ARMS.md` has the x86 numbers).

Absolute latency will be higher than a workstation's; the ordering between
arms should hold, the magnitudes will not. Measure, do not extrapolate.

## Failure modes

| symptom | cause |
|---|---|
| `Failed to deserialize the cuda engine` | engine built on another device, or not enough free VRAM at load |
| plugin library not found / fails to load | `foldquant.kernels build` never run on this board, or `FOLDQUANT_CACHE_DIR` points elsewhere |
| engines load, actions are wrong | a graph's `.onnx.data` did not transfer, or the engine directory does not match the checkpoint being served |

The third is the dangerous one: nothing reports it. Step 5 is how you find it.

## Status on Orin

Checked on a Jetson AGX Orin 64 GB, JetPack 6.2 (L4T R36.4.3), CUDA 12.6,
TensorRT 10.3.0, with each family in its own virtualenv built from its
`scripts/deployment/orin` recipe where one exists. "Runs" means the script
completed on that board; these were checks that the path works, so no number
from them belongs in `results/`.

| family | export → build → verify | serve (bf16 / engines) | LIBERO rollout | benchmark |
|---|---|---|---|---|
| GR00T N1.7 | runs | runs / runs | runs | runs |
| GR00T N1.6 | runs | runs / runs | runs | runs |
| GR00T N1.5 | runs | runs / runs | runs | runs |
| π₀.₅ | runs | runs / runs | client not covered | runs |

Platform notes that are not bugs in this repository:

- GPTQ's Cholesky falls back to the CPU on every family: JetPack 6.2's
  `libcusolver.so.11` is older than the one torch expects. It is logged and
  only costs calibration time.
- The LIBERO rollouts need the simulator stack from each family's integration
  README (`robosuite==1.4.0`, `mujoco==2.3.7`, ...); those wheels install on
  aarch64 without changing anything else in the environment.
- torchcodec does not load in the GR00T N1.5 venv; set
  `N15_VIDEO_BACKEND=decord` for `smoke_family.sh` and `bench_all.sh`.
- π₀.₅ rolls out LIBERO from a separate client environment against
  a running server, which `smoke_eval.sh` does not cover; only its server
  side (the serve column) was checked.
