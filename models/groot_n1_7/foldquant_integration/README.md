# FoldQuant on GR00T N1.7

This folder is the whole of what FoldQuant adds to the upstream GR00T N1.7
release. The `gr00t` package, its data path, its LIBERO rollout loop and its
`scripts/deployment` TensorRT tools are used unchanged; the algorithm, the
ONNX emitters and the TensorRT plugins are the top-level
[`foldquant`](../../../foldquant) package.

Two modules of the policy are replaced by FoldQuant plugin graphs:

| module | attribute | graph | schemes |
|---|---|---|---|
| LLM (Qwen3-VL decoder, 12 kept layers) | `policy.model.backbone.model.model.language_model` | `llm_bf16.onnx` | `w8a8_sr` (default), `w8a8_s`, `w4a4_srg`, `w4a4_sg`, `w4a8_srg` |
| DiT (flow-matching action expert) | `policy.model.action_head.model` | `dit_bf16.onnx` | `w4a4_shg` (default), `w4a4_sh`, `w4a4_sr`, `w8a8_sh`, `w8a8` |

Both graphs keep the upstream drop-in I/O contract (same input names,
dtypes, dynamic axes and output names as `export_onnx_n1d7.py`), so the
upstream engine builder compiles them and `trt_model_forward.py` serves them
without knowing they are quantized. The action head reads the LLM's
**pre-final-norm** residual stream (`hidden_states[-1]`), as upstream does;
the FoldQuant LLM graph is emitted with `final_norm=False` for that reason.

## Environment

The upstream pins apply (Python 3.10, torch 2.7.1, transformers 4.57.3,
TensorRT 10.15). From this directory:

```bash
uv sync                         # upstream environment
uv pip install -e ../..         # the foldquant package into it
python -m foldquant.kernels build   # compile the plugin libraries for this GPU / TensorRT
```

`foldquant.kernels build` needs `nvcc` and the TensorRT headers; the result
is cached per `(SM, arch, TensorRT major.minor)` and looked up exactly, so a
different GPU or TensorRT build compiles its own copy rather than loading an
ABI-mismatched library.

## Workflow

1. **Float export and engines (upstream).** Everything FoldQuant does not
   replace — ViT, VL self-attention, state/action encoders, action decoder —
   comes from the upstream pipeline:

   ```bash
   python scripts/deployment/build_trt_pipeline.py \
       --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
       --dataset-path demo_data/libero_demo --embodiment-tag LIBERO_PANDA \
       --output-dir exports/n17_float --steps export,build
   ```

2. **Calibrate and emit the FoldQuant graphs.** Observations are drawn
   through the upstream data path (`LeRobotEpisodeLoader` →
   `extract_step_data` → `parse_observation_gr00t`) from a seeded, episode-
   balanced plan; the forward loop replays `policy.get_action` so every
   capture sees exactly the inference-time tensors.

   ```bash
   python -m foldquant_integration.export_foldquant \
       --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
       --dataset-path <calibration dataset> --num-calib 128 \
       --llm-scheme w8a8_sr --dit-scheme w4a4_shg \
       --output-dir exports/n17_w8a8_w4a4
   ```

   `--cascade` calibrates the DiT while the LLM runs under FoldQuant's
   fake-quant emulation of its own fold, so the DiT's SmoothQuant scales and
   GPTQ Hessians see the activations it will receive at deployment. Pass
   `--llm-scheme none` (or `--dit-scheme none`) to leave a module at the
   float export.

3. **Build the engine directory.**

   ```bash
   python -m foldquant_integration.build_engines \
       --onnx-dir exports/n17_w8a8_w4a4/onnx --engine-dir exports/n17_w8a8_w4a4/engines \
       --float-onnx-dir exports/n17_float/onnx --float-engine-dir exports/n17_float/engines
   ```

   Loads the plugin library, compiles each FoldQuant graph with upstream's
   `build_engine` (strongly typed, same shape profiles), and fills the
   remaining components from the float directory — by building their ONNX
   when given, or copying their `.engine`. The result is a complete
   `n17_full_pipeline` directory; `foldquant_export.json` travels with it so
   the runtime tools know which plugin library to load.

4. **Verify, evaluate, benchmark.**

   ```bash
   python -m foldquant_integration.verify --model-path ... --dataset-path ... \
       --engine-dir exports/n17_w8a8_w4a4/engines
   MUJOCO_GL=egl python -m foldquant_integration.eval_libero --model-path ... \
       --engine-dir exports/n17_w8a8_w4a4/engines --n-envs 1 --output exports/n17_w8a8_w4a4/libero
   python -m foldquant_integration.benchmark --model-path ... \
       --trt-engine-path exports/n17_w8a8_w4a4/engines --trt-mode n17_full_pipeline
   ```

   `verify` follows upstream's `verify_n1d7_trt.py` (backbone features and
   the decoded action chunk, seeded flow-matching noise) on held-out
   observations from episodes the calibration never saw. `eval_libero` runs
   the upstream `MultiStepWrapper` rollout over every task of the requested
   suites with a resume-safe `summary.json`. `benchmark` and `rollout` are
   upstream's own scripts with the plugin library preloaded — the arguments
   are theirs.

Plugin graphs are emitted at the batch the calibration captured (1), so
TensorRT arms run `--n-envs 1`; the PyTorch arm may batch.

## Files

| file | role |
|---|---|
| `calibration.py` | upstream policy / dataset loading, seeded sample plan, observation building, forward loop |
| `export_foldquant.py` | scheme validation, shape-metadata capture, `export_llm` / `export_dit`, manifests |
| `build_engines.py` | plugin load + upstream `build_engine` per component; float completion |
| `verify.py` | held-out PyTorch-vs-engine drift report |
| `eval_libero.py` | LIBERO sweep over suites × tasks, per-task `summary.json` |
| `rollout.py`, `benchmark.py` | upstream tools with plugins preloaded |
| `_upstream.py`, `_runpy.py` | paths, component table, `runpy` hand-off |
