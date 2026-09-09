# Deploying a FoldQuant arm on Jetson AGX Orin

The engine an arm runs is compiled for exactly one `(GPU architecture,
TensorRT version)` pair. An engine built on an x86 workstation
(`sm89-x86_64-trt10.15`, say) **will not load** on an Orin
(`sm87-aarch64-trt10.3`), and neither will the plugin `.so`: the locator
matches `sm<CC>-<machine>-trt<MAJ>.<MIN>` exactly, tolerating only an older
minor within the same major.

That single fact shapes the whole procedure. ONNX is portable; engines and
plugin binaries are not.

```
        workstation (GPU, RAM)                  Jetson AGX Orin
 ┌──────────────────────────────────┐    ┌──────────────────────────────┐
 │ calibrate + export → ONNX        │    │ build the plugin .so         │
 │ (the expensive half: checkpoint, │───▶│ build engines from the ONNX  │
 │  dataset, GPTQ, VRAM)            │ONNX│ verify, then serve           │
 └──────────────────────────────────┘    └──────────────────────────────┘
```

Calibrating on the Orin itself is possible — 64 GB of unified memory is
enough — but GPTQ is CPU-heavy (measured at ~25 minutes per arm on eight x86
cores at 770% CPU), so the split above is about wall-clock, not capability.

> **Status.** The steps below are derived from the code paths they invoke —
> the kernel locator, the build's header selection, the engine builder — and
> every flag has been checked against the CLI it belongs to. They have **not**
> been executed end to end on an Orin: `results/README.md` still carries the
> Jetson latency row as pending. Where a number appears it is from an x86
> workstation and says so.

## 1. Export, on the workstation

Nothing here is Jetson-specific; it is the family's ordinary export, stopped
before the build step.

```bash
cd models/groot_n1_7 && source .venv/bin/activate
CK=<checkpoint directory>      # a local path: a hub id would record its org
DS=<LeRobot dataset>

# 1a. the upstream float graphs — every arm shares them
python scripts/deployment/build_trt_pipeline.py \
    --model-path "$CK" --dataset-path "$DS" --embodiment-tag <TAG> \
    --output-dir exports/float --steps export

# 1b. the FoldQuant graphs
python -m foldquant_integration.export_foldquant \
    --model-path "$CK" --dataset-path "$DS" --embodiment-tag <TAG> \
    --num-calib 128 --seed 0 \
    --llm-scheme w8a8_sr --dit-scheme w8a8_sh \
    --output-dir exports/w8a8
```

`--steps export` omits the build deliberately: an engine compiled here cannot
run on the Orin. Add `build` only if you also want to test on the workstation.

Keep `--num-calib` at 128 or more for a GPTQ (`_g`) arm — the DiT's widest
site is K = 6144, and 32 observations leave the Hessian rank deficient at
every block.

## 2. Move the ONNX

Engines stay behind. Sizes measured on a GR00T N1.7 export:

| directory | contents | size |
|---|---|---|
| `exports/float/onnx/` | 7 graphs: ViT, VL self-attention, state / action encoders, decoder, LLM, DiT | 6.1 GB |
| `exports/w8a8/onnx/` | the 2 FoldQuant graphs plus their manifests | 2.0 GB |
| `exports/w4a4/onnx/` | same two, INT4 | 958 MB |

```bash
rsync -avP exports/float/onnx  orin:<repo>/models/groot_n1_7/exports/float/
rsync -avP exports/w8a8/onnx   orin:<repo>/models/groot_n1_7/exports/w8a8/
```

Copy whole directories. A large graph carries its weights in a sibling
`.onnx.data` file, and the `.onnx` alone is useless without it.

The checkpoint is needed on the Orin too, even though the engines hold the
quantized weights: `serve` builds the upstream policy for everything FoldQuant
does not replace, and for the processor.

## 3. Build the plugin library, on the Orin

```bash
git submodule update --init third_party/cutlass
cd models/groot_n1_7
uv sync && uv pip install -e ../..
python -m foldquant.kernels build
python -m foldquant.kernels status
```

`status` prints the target slug it resolved. On an Orin it must read
`sm87-aarch64-trt10.3` (or whatever TensorRT JetPack installed) — if it prints
an x86 slug you are on the wrong machine.

The build picks its headers from the **installed** TensorRT major:
`third_party/tensorrt-headers/include-trt10/` for TensorRT 10 (Orin),
`include/` for 11.x. Compiling one against the other is an ABI mismatch that
does not fail at build time — it fails when the plugin loads.

Binaries land in `~/.cache/foldquant/plugins/<slug>/`, or under
`FOLDQUANT_CACHE_DIR` if set. Nothing is committed to the repository.

## 4. Build the engines, on the Orin

```bash
python -m foldquant_integration.build_engines \
    --onnx-dir exports/w8a8/onnx --engine-dir exports/w8a8/engines \
    --float-onnx-dir exports/float/onnx
```

Pass `--float-onnx-dir`, not `--float-engine-dir`: the Orin has no float
engines yet, so the untouched components have to be **built** from their ONNX
rather than copied.

Budget roughly 15–20 GB of free space for one complete arm — the engine
directory alone is 4.5 GB for W8A8, 3.5 GB for W4A4, and the ONNX stays beside
it.

## 5. Verify before serving

```bash
python -m foldquant_integration.verify \
    --model-path "$CK" --dataset-path "$DS" --embodiment-tag <TAG> \
    --engine-dir exports/w8a8/engines --num-samples 32 --seed 42
```

This is not optional on a new device. An engine built against the wrong
headers, or from a graph whose `.onnx.data` did not transfer, loads without
complaint and returns wrong actions — there is no error to catch. `verify`
scores the engine against the bf16 policy on the same board, so a build
problem shows up as a number, before any hardware moves.

## 6. Serve

```bash
python -m foldquant_integration.serve \
    --model-path "$CK" --embodiment-tag <TAG> \
    --engine-dir exports/w8a8/engines \
    --mode n17_full_pipeline --port 5555
```

Drop `--engine-dir` for the bf16 reference arm. The wire protocol is
upstream's, so a robot client points at the Orin by changing a host and a port
— see [`../real_robot/README.md`](../real_robot/README.md).

`FOLDQUANT_TRT_CUDA_GRAPH=1` enables CUDA-graph replay, which removes
per-call launch overhead. It matters most for engines called many times per
action chunk (Evo-1's action head runs 50×, π₀.₅'s expert 10×) and least for
GR00T N1.7's four denoising steps. Launch overhead is relatively larger on an
Orin than on a workstation, so it is worth measuring there even where it did
not pay on x86.

## Reading the latency you get

Two cautions, both from measurements in `results/README.md`.

**Absolute numbers will be higher than a workstation's.** On an RTX 4070 Ti
SUPER a GR00T N1.7 policy served over ZMQ answers in about 75 ms eager and
43 ms under W8A8. The Orin has fewer SMs and less bandwidth; the ordering
between arms should hold, the magnitudes will not. Measure, do not
extrapolate.

**Most of the speedup is TensorRT, not quantization.** Across three modules in
two families the float-engine control puts 75–78% of the saving on compiling
the graph and the rest on precision, and an unrelated mechanism
(`torch.compile` on π₀.₅) agrees at 74%. If you want that split on your board,
build the `float` arm as well and compare against it — comparing a quantized
arm against eager PyTorch measures both effects at once.

## Failure modes

| symptom | cause |
|---|---|
| `Failed to deserialize the cuda engine` | engine built on another device, or not enough free VRAM at load |
| plugin library not found / fails to load | `foldquant.kernels build` was never run on this board, or `FOLDQUANT_CACHE_DIR` points elsewhere |
| engines load, actions are wrong | a graph's `.onnx.data` did not transfer, or the engine directory does not match the checkpoint being served |

The third is the dangerous one: nothing reports it. Step 5 is how you find it.
