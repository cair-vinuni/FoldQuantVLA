# FoldQuant

W4A4 / W8A8 quantization for vision-language-action (VLA) inference on edge
GPUs by **offline folding**: the SmoothQuant scale, a per-block rotation and
the weight rounding are folded into each linear site's weights before export,
so at inference a single fused TensorRT plugin quantizes the activation row,
runs the INT4 / INT8 GEMM and dequantizes — no online rotation, no per-token
scale search, no extra graph nodes between the plugin and its neighbours.

This repository is the paper's artifact. It has two parts:

- **[`foldquant/`](foldquant)** — the algorithm, the ONNX emitters and the
  TensorRT plugin kernels. Model-agnostic; installed as a Python package into
  each model's own environment.
- **`models/<family>/`** — one directory per VLA release, a trimmed copy of
  the upstream repository at a pinned commit plus a `foldquant_integration/`
  folder. Upstream code, data path, evaluation harness and deployment tools
  are used **unchanged**; the integration only emits the FoldQuant graphs for
  the modules it quantizes and slots them into the upstream TensorRT pipeline.

| family | upstream | integration | status |
|---|---|---|---|
| GR00T N1.7 | NVIDIA Isaac GR00T, `n1.7-release` (`23ace64f`) | [`models/groot_n1_7`](models/groot_n1_7/foldquant_integration/README.md) | complete |
| GR00T N1.6 | NVIDIA Isaac GR00T, `n1.6.1-release` (`5dc80c4a`) | [`models/groot_n1_6`](models/groot_n1_6/foldquant_integration/README.md) | complete |
| GR00T N1.5 | NVIDIA Isaac GR00T, `n1.5-release` (`4af2b622`) | [`models/groot_n1_5`](models/groot_n1_5/foldquant_integration/README.md) | complete |
| π₀.₅ | openpi, `main` (`215abfb2`) | [`models/pi05`](models/pi05/foldquant_integration/README.md) | complete |
| SmolVLA | LeRobot | — | planned |
| Evo-1 | Evo-1 | — | planned |

## Schemes

A scheme key is `w{W}a{A}` followed by the fold it applies, one letter per pass
in a fixed order: `s` SmoothQuant scale, then `r` (learned dense block
rotation) or `h` (fixed Sylvester butterfly, applied as an FWHT), then `g`
GPTQ rounding. `w8a8` alone is the dynamic per-row baseline and folds nothing.

| target | schemes | plugin library |
|---|---|---|
| action expert (DiT / action head / expert) | `w4a4_shg` · `w4a4_sh` · `w4a4_sr` · `w8a8_sh` · `w8a8` | `foldquant_int4_per_row` / `foldquant_int8_per_row` |
| LLM backbone | `w8a8_sr` · `w8a8_s` · `w4a4_srg` · `w4a4_sg` · `w4a8_srg` | same, by activation width |
| W4A16 baseline | ModelOpt AWQ groupwise | `foldquant_int4_groupwise` |

`foldquant/schemes.py` is the single source of these names and of which
modules each is allowed on. GPTQ (`…g`) changes nothing at runtime — same
kernel, node attributes and byte layout — it only spends the same 16 levels
better, so every `_shg` engine runs at the `_sh` engine's latency.

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
paper/                evaluation protocol and measured results
```

## Install

Each model directory pins its own environment (Python, torch, TensorRT); the
`foldquant` package installs into it:

```bash
git submodule update --init third_party/cutlass
cd models/groot_n1_7 && uv sync && uv pip install -e ../..
python -m foldquant.kernels build      # compiles the plugin .so for this GPU / TensorRT
python -m foldquant.kernels status
```

Each integration README lists the family's own environment (GR00T N1.6 / N1.5
pin older torch and flash-attn wheels; openpi pins Python 3.11 and copies its
patched `transformers` files over the installed package) and which submodule
it needs — the LIBERO harness is pinned per family
(`models/groot_n1_{7,6}/external_dependencies/LIBERO`,
`models/pi05/third_party/libero`; the N1.5 release pins none, so its README
points at upstream's install steps).

Kernels build with `nvcc`, the CUTLASS submodule and the vendored TensorRT
headers; binaries are cached per `(SM, machine, TensorRT major.minor)` under
`~/.cache/foldquant` and matched exactly at load, never approximately. Unit
tests need only the package:

```bash
pip install -e ".[dev]" && pytest tests -q
```

Then follow the family's integration README, e.g.
[GR00T N1.7](models/groot_n1_7/foldquant_integration/README.md): float
export/engines with the upstream pipeline → `export_foldquant` → `build_engines`
→ `verify` / `eval_libero` / `benchmark`.

## Results

[`paper/README.md`](paper/README.md) describes the protocol (upstream
harnesses, held-out drift, LIBERO success rate, latency on RTX 4070 Ti SUPER
and Jetson AGX Orin) and holds the measured numbers per family and arm.

## License

FoldQuant code is released under the
[PolyForm Noncommercial License 1.0.0](LICENSE) for research use. Upstream
model code under `models/*/` and third-party sources keep their own licenses
(Apache-2.0 / BSD-3-Clause), retained alongside them. See
[CITATION.cff](CITATION.cff) to cite.
