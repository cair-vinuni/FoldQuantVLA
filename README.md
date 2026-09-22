<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/foldquantvla-wordmark-dark.svg">
    <img src="docs/assets/foldquantvla-wordmark.svg" alt="FoldQuantVLA" height="56">
  </picture>
</p>

<p align="center">
  <img alt="Native INT4 / INT8" src="https://img.shields.io/badge/precision-W4A4%20%7C%20W8A8%20native-B9141A">
  <img alt="Runtime" src="https://img.shields.io/badge/runtime-TensorRT%2010%20%2F%2011-17201C">
  <img alt="Targets" src="https://img.shields.io/badge/native%20INT4-sm__87%20Orin%20%7C%20sm__89%20Ada-627067">
  <img alt="Families" src="https://img.shields.io/badge/VLA%20families-GR00T%20N1.5%2FN1.6%2FN1.7%20%7C%20%CF%80%E2%82%80.%E2%82%85-78877E">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11-DAE3DC">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-F2F5F2"></a>
</p>

**FoldQuantVLA: Native Low-Bit Quantization of Vision-Language-Action Models
via Consistent Folding**

W8A8 and W4A4 quantization of vision-language-action (VLA) models, executed
natively on the device's INT8 / INT4 tensor cores (not simulated), via
**consistent offline folding**.

## Highlights

- **Native low-bit kernels.** Custom TensorRT plugins on CUTLASS integer GEMMs run W8A8 and W4A4 on the integer tensor cores. One fused plugin per linear site applies the site's transform, quantizes with a dynamic per-token scale and runs the GEMM, with no separate rotation operator or extra graph nodes. Both the language backbone and the action expert run this way. Four-bit is native INT4 on Ada (sm_89) and Orin (sm_87); H100 lowers the four-bit operands to its INT8 datapath.
- **o/d INT8: W8A8 behaviour at W4A4 latency.** Holding only `o_proj` and `down_proj` at INT8 in a W4A4 language tower raises the worst held-out action cosine from 0.46 to 0.85 on GR00T N1.6 and from 0.85 to 0.998 on π₀.₅, keeps every projection on the integer GEMM path, and costs 0.4-1.8 ms on GR00T. In closed loop it matches the W8A8 engine; on four real-robot tasks it recovers the episodes uniform W4A4 loses (74 against 64 of 80). This is the recommended configuration.
- **Fold offline, keep online work minimal.** SmoothQuant scale and block rotation compose into one transform `T_v = D^o R D^i` per activation site. Its inverse is folded into every consuming weight, GPTQ rounds in those coordinates, and scales absorb into a preceding learned normalization gain where there is one. Only applying `T_v` to each new activation and its dynamic per-token quantization stay online, as one fused prologue shared by the site's projections.
- **Action-referenced calibration.** Presets are screened on decoded-action cosine against the BF16 policy in a statistics-matched emulation, then rebuilt and checked on the assembled engine.
- **Four VLA releases, one build path.** GR00T N1.5 / N1.6 / N1.7 and π₀.₅ use their upstream code, evaluation harness and policy server unchanged; float, W8A8 and W4A4 engines come off the same `export → build → install` path and differ only in projection precision.
- **Deployable.** Engines install into the upstream policy server; the same arm serves LIBERO, a Jetson AGX Orin and a real robot through the family's own upstream client (see [`docs/`](docs)).

## How it works

Only the projection GEMMs change precision. The vision encoder and the decoder
stay floating point; the language backbone and the action expert run INT8 or
INT4, and the expert's denoising loop reuses one engine for every step.

<p align="center">
  <picture>
    <source media="(max-width: 700px)" srcset="docs/assets/foldquantvla-method-mobile.svg">
    <img src="docs/assets/foldquantvla-method.svg" alt="FoldQuantVLA method: only the projections of the language backbone and action expert change precision; one shared transform per activation site is fixed offline and applied as a fused prologue online" width="100%">
  </picture>
</p>

The GR00T DiT and π₀.₅ expert use shared plugin kernels through separate
emitters. LLM backbones use the INT8 per-row path (`w8a8_sr`) or the INT4
path (`w4a4_srg`) with the same weight contract.

The repository contains the shared [`foldquant`](foldquant) package and
pinned upstream releases under `models/<family>/`. Each release adds a
`foldquant_integration/` adapter for export and runtime installation.

| family | upstream | integration | support |
|---|---|---|---|
| GR00T N1.7 | NVIDIA Isaac GR00T, `n1.7-release` (`23ace64f`) | [`models/groot_n1_7`](models/groot_n1_7/foldquant_integration/README.md) | ✓ |
| GR00T N1.6 | NVIDIA Isaac GR00T, `n1.6.1-release` (`5dc80c4a`) | [`models/groot_n1_6`](models/groot_n1_6/foldquant_integration/README.md) | ✓ |
| GR00T N1.5 | NVIDIA Isaac GR00T, `n1.5-release` (`4af2b622`) | [`models/groot_n1_5`](models/groot_n1_5/foldquant_integration/README.md) | ✓ |
| π₀.₅ | openpi, `main` (`215abfb2`) | [`models/pi05`](models/pi05/foldquant_integration/README.md) | ✓ |

Support covers export, engine build, held-out drift, latency, LIBERO, and
policy serving. GR00T runs LIBERO in process; π₀.₅ uses a separate upstream
client environment and a running policy server.

[`results/`](results) holds the held-out drift and desktop latency records
for the W8A8 and W4A4 arms; every table, the protocol and the LIBERO and
Jetson figures are in the paper.

## Schemes

A scheme key is `w{W}a{A}` followed by its folds, one letter per pass in a
fixed order: `s` SmoothQuant scale, then a rotation, then `g` GPTQ rounding. On
the action expert `r` is a learned dense block rotation (the matrix ships with
the engine) and `h` the fixed Sylvester butterfly, applied as an FWHT. On the
LLM backbone `r` is the fixed block Hadamard (block 64), also applied as an
FWHT inside the plugin. `w8a8` alone is the dynamic per-row baseline and folds
nothing.

`float` exports an unquantized engine with the same runtime bindings as the
quantized graph; `none` keeps the module in PyTorch. GR00T N1.7 uses upstream
float graphs through `build_engines --float-onnx-dir`.

Float and quantized engines use strong typing to preserve the graph's declared
dtypes. This keeps the float baseline comparable to the BF16 reference.

| target | schemes | plugin library |
|---|---|---|
| action expert (DiT / expert) | `w4a4_shg` · `w4a4_sh` · `w4a4_sr` · `w8a8_sh` · `w8a8` | `foldquant_int4_per_row` / `foldquant_int8_per_row` |
| LLM backbone | `w8a8_sr` · `w8a8_s` · `w4a4_srg` · `w4a4_sg` · `w4a8_srg` | same, by activation width |
| W4A16 baseline | ModelOpt AWQ groupwise | `foldquant_int4_groupwise` |

[`foldquant/schemes.py`](foldquant/schemes.py) defines supported keys and
modules. GPTQ (`…g`) changes offline weight rounding while retaining the same
runtime kernel, node attributes, and byte layout.

### Selective INT8 inside a W4A4 tower

A W4A4 language tower can hold chosen projection sites at INT8 with
`site_bits`, passed through `--llm-params`:

```bash
--llm-scheme w4a4_srg --llm-params '{"site_bits": {"o": 8, "down": 8}}'
```

Valid sites are `qkv`, `o`, `gateup`, and `down`. Keeping `o_proj` and
`down_proj` at INT8 improves action cosine over uniform W4A4 while retaining
integer GEMMs at every projection. These sites have no preceding learned gain
for scale folding. The paper measures this arm on every checkpoint.

## Layout

```
foldquant/            algorithm (foldq.py), calibration, emitters, export API
  kernels/            TensorRT plugin sources (CUDA + CUTLASS), locator, build CLI
  runtime/            plugin loading, engine build helper, engine wrapper
models/groot_n1_7/    upstream GR00T N1.7 + foldquant_integration/
models/groot_n1_6/    upstream GR00T N1.6.1 + foldquant_integration/
models/groot_n1_5/    upstream GR00T N1.5 + foldquant_integration/
models/pi05/          upstream openpi (π₀ / π₀.₅ PyTorch path) + foldquant_integration/
third_party/          CUTLASS (submodule), vendored TensorRT public headers
tests/                unit tests for the algorithm, emitters, kernel locator / build
results/              evaluation protocol and measured results
docs/deploy/          export on a workstation, build and test on Jetson AGX Orin
```

## Install

Each model directory pins its own environment (Python, torch, TensorRT); the
`foldquant` package installs into it:

```bash
git submodule update --init third_party/cutlass
cd models/groot_n1_7
uv sync
source .venv/bin/activate
uv pip install -e ../..
python -m foldquant.kernels build      # compiles the plugin .so for this GPU / TensorRT
python -m foldquant.kernels status
```

Each integration README lists the family's environment (GR00T N1.6 / N1.5 pin
older torch and flash-attn wheels; openpi pins Python 3.11 and copies its
patched `transformers` files over the installed package) and its LIBERO
submodule (`models/groot_n1_{7,6}/external_dependencies/LIBERO`,
`models/pi05/third_party/libero`; N1.5 pins none, so its README points at
upstream's install steps).

Kernels build with `nvcc`, the CUTLASS submodule and the vendored TensorRT
headers. Binaries are cached per `(SM, machine, TensorRT major.minor)` under
`~/.cache/foldquant` with device and version checks at load. From the repository
root, install the development dependencies and run the unit tests:

```bash
pip install -e ".[dev]"
pytest tests -q
```

Then follow the family's integration README, e.g.
[GR00T N1.7](models/groot_n1_7/foldquant_integration/README.md): float
export/engines with the upstream pipeline → `export_foldquant` → `build_engines`
→ `verify` / `eval_libero` / `benchmark`.

## Results

[`results/`](results) holds the measured records behind the paper's drift
and desktop latency tables: `results/<family>/<arm>/verify.json` (32 held-out
observations, per-observation cosines) for the `w8a8` and `w4a4` arms, and
`results/<family>/benchmark.json` (or `results/groot_n1_7/<arm>/benchmark.log`)
for latency on an RTX 4070 Ti SUPER. `scripts/results_tables.py` prints the
tables from those files and `scripts/check_records.py` checks their
invariants; the protocol and the LIBERO and Jetson AGX Orin figures are in the
paper.

[`docs/REPRODUCING.md`](docs/REPRODUCING.md) lists what can be checked, and at
what cost, from a clone upwards:

- `python scripts/check_records.py`: no GPU, no checkpoint; asserts the
  invariants every record has to satisfy.
- `scripts/smoke_family.sh`: one family's export → build → verify chain on
  eight observations.
- `scripts/smoke_serve.sh`: each family's policy server starts and binds,
  over bf16 or over a built arm.
- `scripts/smoke_eval.sh`: one LIBERO suite at one episode per task, through
  the same `eval_libero` the sweeps use.

A smoke pass means the chain runs, not that a published number reproduces.
That needs the checkpoint and dataset named in the number's record.

## Deploying

Engines install into the **upstream release's own policy server**, so an
unmodified upstream client drives a quantized policy by changing a host and a
port. Each family's repository documents its client and real-robot examples.

Quantization changes the arithmetic, not the policy: an engine built from a
LIBERO checkpoint emits LIBERO actions on any robot, so a real deployment
starts from a checkpoint fine-tuned for that embodiment.

[`docs/deploy/jetson.md`](docs/deploy/jetson.md) covers the edge target the
paper measures: the plugin library and the engines are built on the board
(neither is portable across `(SM, TensorRT)`), while ONNX crosses from a
workstation.
[`docs/deploy/jetson_serve.md`](docs/deploy/jetson_serve.md) builds and serves
GR00T N1.7 entirely on an Orin, step by step or through
[`scripts/deploy_groot_n17_jetson.sh`](scripts/deploy_groot_n17_jetson.sh).

## License

FoldQuant code is released under the
[Apache License, Version 2.0](LICENSE); see [NOTICE](NOTICE). Upstream
model code under `models/*/` and third-party sources keep their own licenses
(Apache-2.0 / BSD-3-Clause), retained alongside them. See
[CITATION.cff](CITATION.cff) to cite.
