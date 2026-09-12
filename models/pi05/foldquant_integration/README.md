# FoldQuant on Pi0 / Pi0.5 (openpi)

This folder is the whole of what FoldQuant adds to the upstream openpi
release. The `openpi` package, its PyTorch model (`PI0Pytorch`), its policy
transforms, its websocket server and its LIBERO client
(`examples/libero/main.py`) are used unchanged; the algorithm, the ONNX
emitters and the TensorRT plugins are the top-level
[`foldquant`](../../../foldquant) package.

Two modules of the policy are replaced by FoldQuant plugin graphs:

| module | attribute | graph | schemes |
|---|---|---|---|
| LLM (PaliGemma Gemma-2B decoder, prefix pass) | `policy._model.paligemma_with_expert.paligemma.language_model` | `llm_bf16.onnx` | `w8a8_sr` (default), `w8a8_s`, `w8a8`, `w4a4_srg`, `w4a4_sg`, `w4a8_srg` |
| action expert (Gemma-300M, one denoise step) | `policy._model.paligemma_with_expert.gemma_expert` + the projections around it | `expert_bf16.onnx` | `w4a4_shg` (default), `w4a4_sh`, `w4a4_sr`, `w8a8_sh`, `w8a8` |

Pi's inference splits into a prefix pass and a suffix loop, and the graphs
follow that split rather than the module tree:

- **Prefix.** SigLIP on the three cameras and the prompt token embedding
  (`embed_prefix`, PyTorch) produce `prefix_embs`; the LLM graph takes
  those with the 4-D additive attention mask and `position_ids`, and
  returns the stacked post-RoPE KV cache `[layers, 2, 1, kv_heads, S, 256]`
  in bf16. The prefix length `S` is a constant of the checkpoint — the
  processor pads the prompt to `max_token_len` and the camera set is fixed
  (3 × 256 image tokens + 200 text tokens = 968 for `pi05_libero`) — so
  the export pins it from a captured call and the engine is fully static.
- **Suffix.** The expert graph is one `denoise_step`: `x_t`, `timestep`,
  `prefix_pad_masks` and the KV stack in, `velocity` out (Pi0 adds
  `state`). Everything a step does is inside it — the action / time
  projections, the 18 Gemma-300M layers cross-attending the cached prefix
  with the adaRMS modulation (Pi0.5) or the state token (Pi0), the output
  projection — and the 10-step Euler loop stays in PyTorch.

The KV stack is the contract between the two seams, so each engine can also
be installed alone: the runtime stacks the PyTorch `DynamicCache` when the
LLM stays float, and rebuilds one from the engine's stack when the expert
stays float.

Upstream openpi ships no TensorRT path, so there is no float-engine floor
arm here; `verify` compares against the bf16 PyTorch policy.

Engines are served by [`runtime.py`](runtime.py), which rebinds
`paligemma_with_expert.forward` (prefix branch only) and
`PI0Pytorch.denoise_step` on the loaded instance; nothing else in the policy
changes, so the same `Policy` runs float, partly or fully quantized, behind
the same `WebsocketPolicyServer`.

The rebinding needs the eager model. Upstream builds `PI0Pytorch` with
`pytorch_compile_mode = "max-autotune"` — `sample_actions` is traced and
replayed as CUDA graphs — and a traced call keeps running the prefix pass
it captured, so a seam bound on the instance afterwards is skipped
(`denoise_step` then receives a `DynamicCache` instead of a KV stack), and
tensors a calibration hook keeps across replays are overwritten.
`calibration.load_policy(..., compile=False)` clears that mode on the train
config; `install_engines` and the calibration capture refuse a compiled
model. The reference serving arm keeps upstream's compiled configuration,
and `benchmark` times it as its own arm.

## Environment

The upstream pins apply (Python 3.11, torch 2.7.1, transformers 4.53.2 with
openpi's patched Gemma copied over it). From this directory:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
source .venv/bin/activate
cp -r --remove-destination src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/
uv pip install "tensorrt-cu12==10.15.1.29"
uv pip install -e ../.. "onnx<1.18"    # the foldquant package into it
python -m foldquant.kernels build      # compile the plugin libraries for this GPU / TensorRT
```

Two deviations from a plain `uv pip install`: `--remove-destination` because uv
hardlinks site-packages into its wheel cache, so copying the patched files *over*
them would also rewrite the cached `transformers` wheel for every other
environment on the machine; and `onnx<1.18` because upstream pins
`ml-dtypes==0.4.1` (an override in `pyproject.toml`) and `onnx` 1.18+ imports
`ml_dtypes.float4_e2m1fn`, which that version does not have.

`foldquant.kernels build` needs `nvcc` and the TensorRT headers; the result
is cached per `(SM, arch, TensorRT major.minor)` and looked up exactly, so a
different GPU or TensorRT build compiles its own copy rather than loading an
ABI-mismatched library. The checkpoint must be the **PyTorch** form
(`model.safetensors`); convert a JAX checkpoint with upstream's
`examples/convert_jax_model_to_pytorch.py` first. The LIBERO client runs in
its own environment (`examples/libero/README.md`, "Without Docker").

## Workflow

Every step takes `--config` (the upstream training config; default
`pi05_libero`) and `--checkpoint-dir`. The calibration dataset is any
LeRobot-format LIBERO dataset (`observation.images.image`,
`observation.images.wrist_image`, `observation.state`, tasks) — the
integration builds the client's observation dictionaries from it, so the
policy sees exactly what the websocket server hands it.

1. **Calibrate and emit the FoldQuant graphs.** Observations are drawn from
   a seeded, episode-balanced plan and run through `Policy.infer` with
   explicit, seeded flow-matching noise, so every capture sees the
   inference-time tensors and the same inputs emit byte-identical graphs.

   ```bash
   python -m foldquant_integration.export_foldquant \
       --checkpoint-dir <pi05_libero PyTorch checkpoint> \
       --dataset-path <LeRobot LIBERO dataset> --num-calib 128 \
       --llm-scheme w8a8_sr --expert-scheme w4a4_shg \
       --output-dir exports/pi05_w8a8_w4a4
   ```

   `--cascade` calibrates the expert while the LLM runs under FoldQuant's
   fake-quant emulation of its own fold. `--llm-scheme none` /
   `--expert-scheme none` leaves a module in PyTorch. Keep `--num-calib`
   at 128 or more for a GPTQ (`_g`) arm.

2. **Build the engine directory.**

   ```bash
   python -m foldquant_integration.build_engines \
       --onnx-dir exports/pi05_w8a8_w4a4/onnx --engine-dir exports/pi05_w8a8_w4a4/engines
   ```

   Loads the plugin libraries, then compiles each FoldQuant graph strongly
   typed. Both graphs are static at the captured prefix length;
   `foldquant_engines.json` records what was built from where, and
   `foldquant_export.json` travels with the engines so the runtime tools
   know which plugin libraries to load.

3. **Verify, serve or evaluate, benchmark.**

   ```bash
   python -m foldquant_integration.verify --checkpoint-dir ... --dataset-path ... \
       --engine-dir exports/pi05_w8a8_w4a4/engines

   # upstream's client/server protocol, server side quantized:
   python -m foldquant_integration.serve --checkpoint-dir ... --engine-dir exports/pi05_w8a8_w4a4/engines
   MUJOCO_GL=egl <libero venv>/bin/python examples/libero/main.py --args.task-suite-name libero_spatial

   # the same pairing over every suite, resumable:
   MUJOCO_GL=egl python -m foldquant_integration.eval_libero --checkpoint-dir ... \
       --engine-dir exports/pi05_w8a8_w4a4/engines \
       --client-python examples/libero/.venv/bin/python --output results/pi05_w8a8_w4a4

   python -m foldquant_integration.benchmark --checkpoint-dir ... --dataset-path ... \
       --arms w4a4=exports/pi05_w8a8_w4a4/engines
   ```

   `verify` scores the prefix KV stack and the action chunk the client
   receives under seeded flow-matching noise on held-out observations from
   episodes the calibration never saw, and records every observation's
   drift in `verify.json`; `--split-from <quantized engine dir>` scores
   another directory on that arm's held-out set. `serve` is upstream's
   `WebsocketPolicyServer` around a policy with the engines installed, so
   upstream's client runs unchanged. `eval_libero` starts that server and
   runs the unmodified upstream client per suite in its own environment,
   reading the final success rate off its log into a resume-safe
   `summary.json` (the client's replay videos land under
   `<output>/videos/`). `benchmark` times the input transforms, prefix
   embedding, prefix LLM pass, the denoise loop and the whole
   `Policy.infer` for the eager PyTorch arm and each `--arms` engine
   directory in one process, plus upstream's `torch.compile(max-autotune)`
   serving configuration end to end only (the component stopwatches would
   break its graphs).

Plugin graphs are emitted at batch 1; the upstream client sends one
observation per request.

## A checkpoint the release has never heard of

Every tool takes `--config`, and upstream resolves that name from a list of its
own `TrainConfig` entries. A checkpoint fine-tuned elsewhere carries a config
that is not in that list, and editing upstream's file to add one is not an
option here — the tree under `src/openpi` is used unchanged.

`FOLDQUANT_PI05_PLUGIN` names a Python file imported before any config is
resolved. Upstream keeps its configs in a module-level dict, so the plugin
registers the entry itself:

```python
# my_plugin.py  -- its directory goes on sys.path first, so siblings import
from openpi.training import config as _config
from openpi.training.config import DataConfig, TrainConfig
import my_policy                      # the checkpoint's own transforms

_config._CONFIGS_DICT.setdefault("pi05_mine", TrainConfig(name="pi05_mine", ...))
```

```bash
export FOLDQUANT_PI05_PLUGIN=/path/to/my_plugin.py
python -m foldquant_integration.export_foldquant \
    --checkpoint-dir <ckpt> --config pi05_mine --dataset-path <lerobot dataset> \
    --llm-scheme w8a8_sr --expert-scheme w8a8_sh --output-dir exports/mine
```

An import error in the plugin is raised, not swallowed: a plugin that fails to
load would otherwise surface as "config not found", which points at the wrong
thing.

### Dataset columns

Calibration reads two camera streams and a state vector out of the LeRobot
dataset. The defaults are LIBERO's column names
(`observation.images.image`, `observation.images.wrist_image`,
`observation.state`); a dataset that has them is read with them, and only a
column the dataset does **not** carry is looked up in the train config's repack
transform.

That order matters. Deriving the names from the config alone is wrong:
`pi05_libero` repacks from `image` / `wrist_image` / `state`, the columns of the
`physical-intelligence/libero` release, while the LeRobot conversion calibrated
on here stores `observation.images.image`. The dataset is the authority on its
own columns; the config is the fallback for a dataset that names them
differently. If neither answers, the error lists the columns that are present.

### Video timestamps

`load_dataset` takes a `video_backend`. LeRobot fetches frames **by timestamp**
and checks the result to 1e-4 s, so a dataset whose `.mp4` files carry a
non-zero container `start_time` either raises or — worse — returns a
neighbouring frame. An index-based loader would not notice, which is how such a
dataset can look fine elsewhere. The fix belongs in the dataset, losslessly:

```bash
ffmpeg -nostdin -fflags +genpts -i in.mp4 -c copy -reset_timestamps 1 out.mp4
```

Raising the tolerance instead would accept the wrong frame silently, which is
the one outcome calibration cannot afford.

## Smoke check

`w8a8_sr` LLM + `w4a4_sh` expert, 16 calibration observations, 8 held-out
observations from episodes the calibration never saw, float64 cosines
against the bf16 eager PyTorch policy under the same seeded flow-matching
noise (`verify`), `pi05_libero` checkpoint at its 10 denoising steps, one
RTX 4070 Ti SUPER (sm89), TensorRT 10.15:

| seam | cos mean | cos min | note |
|---|---|---|---|
| `kv_stack` (18 layers × K,V × 968 × 256) | 0.99922 | 0.99910 | per-position min 0.988 |
| action chunk (10 × 7, normalised) | 0.99987 | 0.99984 | max abs 0.026 |

`benchmark` on the same engines (10 iterations, median): PyTorch eager
165.3 ms (prefix LLM 58.4, denoise loop 79.1); upstream's serving default,
`torch.compile(max-autotune)`, 100.1 ms end to end; FoldQuant 85.7 ms
(prefix LLM 26.8, denoise loop 30.8) — 1.93× over eager, 1.17× over the
compiled baseline, with SigLIP, the prefix embedding and the Euler loop
still in eager PyTorch in the FoldQuant arm. Paper numbers use 128
calibration observations and the server's harness; this is the
installation check.

## Files

| file | role |
|---|---|
| `calibration.py` | upstream policy / dataset loading, seeded sample plan, client-format observations, forward loop |
| `export_foldquant.py` | scheme validation, shape capture, `export_llm` / `export_expert`, manifests |
| `build_engines.py` | plugin load + `foldquant.runtime.builder.build_engine` per component |
| `runtime.py` | engine installer (`install_engines`), the two rebinds, KV-stack helpers, `PrefixCapture` |
| `verify.py` | held-out PyTorch-vs-engine drift report |
| `serve.py` | upstream `WebsocketPolicyServer` with engines installed |
| `eval_libero.py` | LIBERO sweep through upstream's client, per-suite `summary.json` |
| `benchmark.py` | component and end-to-end timing over the PyTorch arm and engine directories |
| `_upstream.py` | paths, component table |
