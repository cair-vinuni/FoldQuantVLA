# Installing and testing a FoldQuant arm on Jetson AGX Orin

An engine is compiled for exactly one `(GPU architecture, TensorRT version)`
pair, and so is the plugin `.so`. An engine built on an x86 workstation
(`sm89-x86_64-trt10.15`) **will not load** on an Orin (`sm87-aarch64-trt10.3`):
the locator matches `sm<CC>-<machine>-trt<MAJ>.<MIN>` exactly, tolerating only
an older minor within the same major.

So: **ONNX crosses, binaries do not.** Export on the workstation, build and
test on the board.

> The steps are derived from the code paths they invoke and every flag is
> checked against its CLI, but they have not been run end to end on an Orin —
> `results/README.md` still carries the Jetson latency row as pending.

## 1. Export, on the workstation

The family's ordinary export, stopped before the build. Calibrating on the
Orin would work (64 GB unified memory is enough) but GPTQ is CPU-heavy, so this
is about wall-clock, not capability.

```bash
cd models/groot_n1_7 && source .venv/bin/activate
CK=<checkpoint directory>      # a local path: a hub id would record its org
DS=<LeRobot dataset>

# the upstream float graphs — every arm shares them
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
`.onnx.data` file and the `.onnx` alone is useless without it. Budget ~15–20 GB
free for one arm (W8A8 engines are 4.5 GB, their ONNX 2.0 GB).

**The checkpoint and the dataset have to be on the board too** — `serve` builds
the upstream policy around the engines, and step 5 opens the dataset to score
against it.

## 3. Build the plugin library, on the Orin

```bash
git submodule update --init third_party/cutlass
cd models/groot_n1_7
uv sync && uv pip install -e ../..
python -m foldquant.kernels build
python -m foldquant.kernels status
```

`status` must print an Orin slug — `sm87-aarch64-trt10.3`, or whatever
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
returns wrong actions — there is no error to catch. `verify` scores the engines
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
— a robot client points at the Orin by changing a host and a port; each
family's own repository documents its client.

`FOLDQUANT_TRT_CUDA_GRAPH=1` enables CUDA-graph replay. It pays most for
engines called many times per chunk (Evo-1's head runs 50×, π₀.₅'s expert 10×),
and launch overhead is relatively larger on an Orin than on x86, so it is worth
measuring there even where it did not pay on a workstation.

Absolute latency will be higher than a workstation's — the ordering between
arms should hold, the magnitudes will not. Measure, do not extrapolate.

## Failure modes

| symptom | cause |
|---|---|
| `Failed to deserialize the cuda engine` | engine built on another device, or not enough free VRAM at load |
| plugin library not found / fails to load | `foldquant.kernels build` never run on this board, or `FOLDQUANT_CACHE_DIR` points elsewhere |
| engines load, actions are wrong | a graph's `.onnx.data` did not transfer, or the engine directory does not match the checkpoint being served |

The third is the dangerous one: nothing reports it. Step 5 is how you find it.
