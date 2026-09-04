# FoldQuant on SmolVLA (LeRobot)

This folder is the whole of what FoldQuant adds to the upstream LeRobot
release. The `lerobot` package, its `SmolVLAPolicy`, its processor pipelines,
its LIBERO environment and its evaluator (`lerobot.scripts.lerobot_eval`) are
used unchanged; the algorithm, the ONNX emitters and the TensorRT plugins are
the top-level [`foldquant`](../../../foldquant) package.

Two modules of the policy are replaced by FoldQuant plugin graphs:

| module | attribute | graph | schemes |
|---|---|---|---|
| LLM (SmolVLM2-500M text decoder, prefix pass) | `policy.model.vlm_with_expert.get_vlm_model().text_model` | `llm_bf16.onnx` | `w8a8_sr` (default), `w8a8_s`, `w8a8`, `w4a4_srg`, `w4a4_sg`, `w4a8_srg` |
| action expert (one layer per VLM layer, one denoise step) | `policy.model.vlm_with_expert.lm_expert` + the projections around it | `expert_bf16.onnx` | `w4a4_shg` (default), `w4a4_sh`, `w4a4_sr`, `w8a8_sh`, `w8a8` |

SmolVLA's inference splits into a prefix pass and a suffix loop, and the graphs
follow that split rather than the module tree:

- **Prefix.** SigLIP on the cameras, the prompt token embedding and the state
  token (`embed_prefix`, PyTorch) produce `prefix_embs`; the LLM graph takes
  those with the BOOL block attention mask and `position_ids`, and returns the
  stacked post-RoPE KV cache `[layers, 2, 1, prefix_len, kv_heads, head_dim]`
  in bf16. The prefix length is **not** constant: the images and the state
  token contribute a fixed count, but the processor tokenizes the prompt with
  `pad_language_to = "longest"`, which at batch 1 means the task string's own
  length, truncated to `tokenizer_max_length` (48). The export records the
  fixed part from a captured call and the config's cap, and the engines are
  profiled over that whole range — a prefix pinned to one captured length
  refuses the next task string.
- **Suffix.** The expert graph is one `denoise_step`: `x_t`, `timestep`,
  `prefix_pad_masks` and the KV stack in, `velocity` out. Everything a step
  does is inside it — the action and time projections, the expert layers alternating
  self-attention with cross-attention over the cached prefix
  (`self_attn_every_n_layers = 2`), the final RMSNorm and the float32
  `action_out_proj` — and the 10-step Euler loop stays in PyTorch. The expert
  carries one layer per VLM layer: 32 for `smolvla_libero`, whose
  `num_vlm_layers = 0` keeps all of SmolVLM2-500M's text stack, against the 16
  of the config default. The emitter reads the count, the widths and the GQA
  ratio off the loaded module, so a checkpoint with a different trim needs no
  change here.

The KV stack is the contract between the two seams, so each engine can also be
installed alone: the runtime stacks the PyTorch `DynamicCache` when the LLM
stays float, and rebuilds one from the engine's stack when the expert stays
float. Note that the stack keeps upstream's own `[batch, seq, heads, dim]`
order — `DynamicCache` stores the transpose of it, and the runtime undoes that
transpose rather than the module's.

Upstream LeRobot ships no TensorRT path, so there is no float-engine floor arm
here; `verify` compares against the bf16 PyTorch policy.

Engines are served by [`runtime.py`](runtime.py), which rebinds
`vlm_with_expert.forward` (prefix branch only) and `denoise_step` on the loaded
instance; nothing else in the policy changes, so the same policy runs float,
partly or fully quantized under the same evaluator.

The rebinding needs the eager model. Upstream's default already is eager
(`compile_model` ships `False`), but a checkpoint that turns it on would have
`sample_actions` traced and replayed as CUDA graphs, and a traced call keeps
running the prefix pass it captured — so a seam bound on the instance
afterwards is skipped, and tensors a calibration hook keeps across replays are
overwritten. `calibration.load_policy(..., compile=False)` clears that flag;
`install_engines` and `PrefixCapture` refuse a compiled model.

## Environment

The upstream pins apply (Python 3.12, the release's `uv.lock`). From this
directory:

```bash
uv sync --extra smolvla --extra libero
source .venv/bin/activate
uv pip install "tensorrt-cu12==10.15.1.29" tyro
uv pip install -e ../..                # the foldquant package into it
python -m foldquant.kernels build      # compile the plugin libraries for this GPU / TensorRT
```

`foldquant.kernels build` needs `nvcc` and the TensorRT headers; the result is
cached per `(SM, arch, TensorRT major.minor)` and looked up exactly, so a
different GPU or TensorRT build compiles its own copy rather than loading an
ABI-mismatched library. The `libero` extra pulls `hf-libero`, so the LIBERO
suites need no separate checkout; rendering wants `MUJOCO_GL=egl` on a
headless machine.

## Workflow

Every step takes `--checkpoint` (a directory or hub id; SmolVLA keeps its
config, normalisation statistics and processor pipeline inside it) and a
`--dataset-path`. The calibration dataset is any LeRobot-format LIBERO
conversion carrying the checkpoint's camera keys and its 8-d state; a dataset
that names its cameras differently is mapped onto the policy's visual features
in declaration order, the same correspondence upstream's `--rename_map`
states by hand.

1. **Calibrate and emit the FoldQuant graphs.** Observations are drawn from a
   seeded, episode-balanced plan and pushed through the checkpoint's own
   preprocessor and `predict_action_chunk` with explicit, seeded flow-matching
   noise, so every capture sees the inference-time tensors and the same inputs
   emit byte-identical graphs.

   ```bash
   python -m foldquant_integration.export_foldquant \
       --checkpoint HuggingFaceVLA/smolvla_libero \
       --dataset-path HuggingFaceVLA/libero --num-calib 128 \
       --llm-scheme w8a8_sr --expert-scheme w4a4_shg \
       --output-dir exports/smolvla_w8a8_w4a4
   ```

   `--cascade` calibrates the expert while the LLM runs under FoldQuant's
   fake-quant emulation of its own fold. `--llm-scheme none` /
   `--expert-scheme none` leaves a module in PyTorch. `--episodes` restricts
   the dataset load, so a hub dataset fetches only the files those episodes
   live in. Keep `--num-calib` at 128 or more for a GPTQ (`_g`) arm.

2. **Build the engine directory.**

   ```bash
   python -m foldquant_integration.build_engines \
       --onnx-dir exports/smolvla_w8a8_w4a4/onnx --engine-dir exports/smolvla_w8a8_w4a4/engines
   ```

   Loads the plugin libraries, then compiles each FoldQuant graph strongly
   typed. Both graphs are profiled at the captured prefix length;
   `foldquant_engines.json` records what was built from where, and
   `foldquant_export.json` travels with the engines so the runtime tools know
   which plugin libraries to load.

3. **Verify, evaluate, benchmark.**

   ```bash
   python -m foldquant_integration.verify --checkpoint ... --dataset-path ... \
       --engine-dir exports/smolvla_w8a8_w4a4/engines

   MUJOCO_GL=egl python -m foldquant_integration.eval_libero \
       --engine-dir exports/smolvla_w8a8_w4a4/engines \
       --output results/smolvla_w8a8_w4a4 --n-episodes 50

   python -m foldquant_integration.benchmark --checkpoint ... --dataset-path ... \
       --arms w4a4=exports/smolvla_w8a8_w4a4/engines
   ```

   `verify` scores the prefix KV stack and the action chunk the postprocessor
   returns under seeded flow-matching noise on held-out observations from
   episodes the calibration never saw, and records every observation's drift in
   `verify.json`; `--split-from <quantized engine dir>` scores another
   directory on that arm's held-out set. `eval_libero` installs the engines and
   runs upstream's own `eval_policy_all` over each suite in turn, writing a
   resume-safe `summary.json` (omit `--engine-dir` to score the bf16 PyTorch
   policy through the same driver). `benchmark` times the processor pipeline,
   the prefix embedding, the prefix LLM pass, the denoise loop and a whole
   inference for the eager PyTorch arm and each `--arms` engine directory in
   one process.

Plugin graphs are emitted at batch 1; the engines therefore evaluate one
environment at a time.

## Smoke check

`w8a8_sr` LLM + `w8a8_sh` expert, 16 calibration observations, 8 held-out
observations from episodes the calibration never saw, float64 cosines against
the bf16 eager PyTorch policy under the same seeded flow-matching noise
(`verify`), the `smolvla_libero` checkpoint at its 10 denoising steps, one RTX
4070 Ti SUPER (sm89), TensorRT 10.15:

| seam | cos mean | cos min | note |
|---|---|---|---|
| `kv_stack` (32 layers × K,V × ~145 × 5 × 64) | 0.99987 | 0.99983 | per-position min 0.996 |
| action chunk (50 × 7, denormalised) | 0.99985 | 0.99935 | max abs 0.264 |

`benchmark` on the same engines (20 iterations, median): PyTorch eager 210.7 ms
(prefix LLM 19.2, denoise loop 176.7); FoldQuant 38.1 ms (prefix LLM 3.0,
denoise loop 21.4) — **5.53×**, with SigLIP and the prompt embedding still in
eager PyTorch in the FoldQuant arm. Upstream serves SmolVLA eagerly, so that is
the deployed reference rather than a compiled one. The speedup is larger than
the other families' because the eager denoise loop is launch-bound: 32 narrow
expert layers, ten times per chunk. Paper numbers use 128 calibration
observations and the upstream evaluator; this is the installation check.

### The expert needs the fold

Which scheme the expert runs is not a free choice on this family, and the same
16/8 check makes that plain (LLM left in PyTorch, so these score the expert
seam alone):

| expert scheme | action cos mean | cos min |
|---|---|---|
| `w8a8` (dynamic per-row, folds nothing) | 0.665 | −0.009 |
| `w4a4_sh` | 0.961 | 0.864 |
| `w8a8_sh` | 0.99956 | 0.99656 |

Raising the budget to the paper's 128 observations and adding GPTQ
(`w4a4_shg`) pulls the 4-bit arm up to 0.992 mean / 0.966 min — better than
`w4a4_sh` at 16, still an order of magnitude behind the folded INT8 arm. That
one number is a fit rather than a held-out reading: 128 observations reach
every episode of the 24 this smoke dataset loads, so its verification episodes
were also calibration episodes. Which arm the paper reports is a success-rate
question, not a drift question; the drift ordering is what this table fixes.

SmolLM2's activations carry the outliers SmoothQuant exists to move, so the
unfolded baseline that is harmless on Pi's Gemma expert is worthless here —
`foldquant/schemes.py` says as much where `w8a8_sh` is defined. The graph
itself is not at fault: exporting with `FOLDQUANT_SMOLVLA_FLOAT_LINEAR=1`,
which keeps the whole graph and floats only the GEMMs, tracks PyTorch at cosine
0.99998 per denoise step, so the scaffolding — KV layout, masks, RoPE, the
baked time embedding — is exact and every point above is the quantizer.

## Files

| file | role |
|---|---|
| `calibration.py` | upstream policy / processor / dataset loading, seeded sample plan, batch building, forward loop |
| `export_foldquant.py` | scheme validation, shape capture, `export_llm` / `export_expert`, manifests |
| `build_engines.py` | plugin load + `foldquant.runtime.builder.build_engine` per component |
| `runtime.py` | engine installer (`install_engines`), the two rebinds, KV-stack helpers, `PrefixCapture` |
| `verify.py` | held-out PyTorch-vs-engine drift report |
| `eval_libero.py` | LIBERO sweep through upstream's evaluator, per-suite `summary.json` |
| `benchmark.py` | component and end-to-end timing over the PyTorch arm and engine directories |
| `_upstream.py` | paths, checkpoint default, suite list, component table |
