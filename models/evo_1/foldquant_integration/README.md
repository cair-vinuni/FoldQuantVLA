# FoldQuant on Evo-1

This folder is the whole of what FoldQuant adds to the upstream Evo-1 release.
The `Evo_1` package, its `EVO1` model, its InternVL3 embedder, its normalizer,
its websocket server and its LIBERO client are used unchanged; the algorithm,
the ONNX emitters and the TensorRT plugins are the top-level
[`foldquant`](../../../foldquant) package.

Two modules of the model are replaced by FoldQuant plugin graphs:

| module | attribute | graph | schemes |
|---|---|---|---|
| LLM (InternVL3 Qwen2 tower) | `model.embedder.model.language_model` | `llm_bf16.onnx` | `w8a8_sr` (default), `w8a8_s`, `w8a8`, `w4a4_srg`, `w4a4_sg`, `w4a8_srg` |
| action head (one denoise step) | `model.action_head` | `action_head_bf16.onnx` | `w4a4_shg` (default), `w4a4_sh`, `w4a4_sr`, `w8a8_sh`, `w8a8` |

Evo-1's inference splits into a fused-token pass and a denoise loop, and the
graphs follow that split:

- **Fused tokens.** The vision tower tiles the three images, the prompt is
  embedded and the two are fused (PyTorch); the LLM graph takes
  `inputs_embeds` with the padding `attention_mask` and returns
  `hidden_states`. Upstream sets `lm_head = Identity()`, so what it reads off
  the call as `logits` *is* the last hidden state. The prompt is padded to a
  fixed length and the image set is fixed at three, so the sequence is a
  constant of the checkpoint and the engine is static; `padded_query_mask`
  reproduces `flash_attention_2`'s treatment of the padded keys, which is what
  the checkpoint was evaluated under.
- **Denoise step.** The head graph is one Euler step: `action_seq`,
  `context_tokens` (the fused tokens with the state token appended) and
  `time_emb` in, `velocity` out. The action encoder, the cross-attention
  blocks, `norm_out`, `seq_pool_proj` and `mlp_head` all live in the graph;
  upstream's 50-step loop stays in PyTorch.

Unlike the KV-stack families the two seams share no cached tensor — they meet
on the fused tokens the tower already returns — so either engine can be
installed alone with no rebuilding on the boundary.

Upstream ships no TensorRT path, so there is no float-engine floor arm here;
`verify` compares against the bf16 PyTorch model.

## One place this mirrors upstream instead of calling it

Every other family here has a method for one denoise step, so the seam is a
rebind. Evo-1 has none: `FlowmatchingActionHead.get_action` inlines the step
inside its Euler loop. [`runtime.py`](runtime.py) therefore rebinds
`get_action` and reimplements the loop around the engine — the state encoder,
the sinusoidal time table, the action mask, the start sample and the
`action + dt * pred` update are all still upstream's own calls, and only the
block stack is replaced. `_check_loop_contract` asserts the pieces it reuses
are still present, and a release that reshapes that loop must be re-read
against `get_action` before its numbers are quoted.

## Environment

The upstream pins apply (Python 3.10, torch 2.5.1, transformers 4.39.0). From
this directory:

```bash
uv venv --python 3.10
uv pip install -r Evo_1/requirements.txt
source .venv/bin/activate
MAX_JOBS=64 uv pip install -v flash-attn --no-build-isolation   # upstream calls this critical
uv pip install "tensorrt-cu12==10.15.1.29" tyro
uv pip install -e ../..                 # the foldquant package into it
python -m foldquant.kernels build       # compile the plugin libraries for this GPU / TensorRT
```

The `lerobot` package is deliberately **not** installed: every release that
still supports Python 3.10 upgrades torch past upstream's pin, and upstream is
explicit that its numbers were measured on the pinned attention path. The
calibration reads LeRobot v3 datasets itself
([`_dataset.py`](_dataset.py)) — a parquet index and stored frames, no
transforms — while every transform an observation passes through stays
upstream's.

`foldquant.kernels build` needs `nvcc` and the TensorRT headers; the result is
cached per `(SM, arch, TensorRT major.minor)`. The LIBERO client runs in its
own environment (upstream's README builds a Python 3.8 one).

## Workflow

Every step takes `--checkpoint-dir` (upstream's directory: `config.json`,
`norm_stats.json`, `mp_rank_00_model_states.pt`) and a `--dataset-path`. The
`--arm-key` / `--dataset-key` pair indexes `norm_stats.json` and defaults to
what upstream's server starts with for LIBERO.

1. **Calibrate and emit the FoldQuant graphs.** Observations are drawn from a
   seeded, episode-balanced plan, built into the JSON dictionaries upstream's
   LIBERO client sends (two cameras, a zero third slot, `image_mask [1,1,0]`,
   the 8-d state, `action_mask` with seven live channels) and pushed through
   upstream's own `infer_from_json_dict`.

   ```bash
   python -m foldquant_integration.export_foldquant \
       --checkpoint-dir <Evo1_LIBERO checkpoint> \
       --dataset-path <LeRobot LIBERO dataset> --num-calib 128 \
       --llm-scheme w8a8_sr --head-scheme w4a4_shg \
       --output-dir exports/evo1_w8a8_w4a4
   ```

   `--cascade` calibrates the head while the tower runs under FoldQuant's
   fake-quant emulation of its own fold. `--llm-scheme none` / `--head-scheme
   none` leaves a module in PyTorch. Keep `--num-calib` at 128 or more for a
   GPTQ (`_g`) arm.

2. **Build the engine directory.**

   ```bash
   python -m foldquant_integration.build_engines \
       --onnx-dir exports/evo1_w8a8_w4a4/onnx --engine-dir exports/evo1_w8a8_w4a4/engines
   ```

3. **Verify, serve or evaluate, benchmark.**

   ```bash
   python -m foldquant_integration.verify --checkpoint-dir ... --dataset-path ... \
       --engine-dir exports/evo1_w8a8_w4a4/engines

   # upstream's client/server protocol, server side quantized:
   python -m foldquant_integration.serve --checkpoint-dir ... --engine-dir exports/evo1_w8a8_w4a4/engines
   MUJOCO_GL=egl <libero venv>/bin/python LIBERO_evaluation/libero_client_4tasks.py

   # the same pairing, driven from one script:
   MUJOCO_GL=egl python -m foldquant_integration.eval_libero --checkpoint-dir ... \
       --engine-dir exports/evo1_w8a8_w4a4/engines \
       --client-python <libero venv>/bin/python --output results/evo1_w8a8_w4a4

   python -m foldquant_integration.benchmark --checkpoint-dir ... --dataset-path ... \
       --arms w4a4=exports/evo1_w8a8_w4a4/engines
   ```

   `verify` scores the fused tokens and the denormalised action chunk on
   held-out observations from episodes the calibration never saw, and records
   every observation's drift in `verify.json`; `--split-from <engine dir>`
   scores another directory on that arm's held-out set, and `--components llm`
   / `--components action_head` install one engine and leave the other module
   in PyTorch, which is how a drift figure is attributed to a seam rather than
   to their sum. `eval_libero` starts
   the served policy and runs upstream's client against it. That client takes
   no arguments — its suite list, episode count, horizon and step budgets are
   class attributes — so it walks all four suites in one process and the
   summary is parsed per suite from its own log; a finished run is not
   repeated, but an interrupted one restarts rather than resuming mid-suite.
   `benchmark` times the embedder (tower included), the tower alone, the
   50-step denoise loop and a whole request for the eager PyTorch arm and each
   `--arms` engine directory.

Plugin graphs are emitted at batch 1; upstream's server answers one request at
a time.

## Smoke check

`w8a8_sr` tower + `w8a8_sh` head, 8 calibration observations, 8 held-out
observations from episodes the calibration never saw, float64 cosines against
the bf16 eager PyTorch model under the same seeded start (`verify`), the
released `Evo1_LIBERO` checkpoint at its 50 denoising steps, one RTX 4070 Ti
SUPER (sm89), TensorRT 10.15, flash-attn installed:

| seam | cos mean | cos min | note |
|---|---|---|---|
| fused tokens (1025 × 896) | 0.99984 | 0.99973 | per-token min 0.996 |
| action chunk (50 × 24, denormalised) | 0.99877 | 0.99094 | max abs 1.006 |

That action-chunk min is tower-side, and `--components` says so by scoring one
seam at a time on the same held-out observations:

| engines installed | actions cos mean | cos min | max abs |
|---|---|---|---|
| `--components action_head` (tower float) | 0.99998 | 0.99994 | 0.011 |
| `--components llm` (head float) | 0.99880 | 0.99098 | 1.006 |

So the head graph and the reimplemented Euler loop around it are very nearly
lossless, and the whole floor is the W8A8 tower on one observation — whose own
fused-token cosine, 0.99983, is unremarkable. Fifty Euler steps are what turn
it into a 1.0 action swing: a small context error is re-read by every step, so
this family amplifies tower drift far more than the ten-step families do. It is
the loosest figure in this repository and it is not a seam defect.

`benchmark` on the same engines (20 iterations, median):

| arm | embed (incl. LLM) | LLM | denoise loop | E2E | speedup |
|---|---:|---:|---:|---:|---:|
| PyTorch eager | 48.7 | 13.3 | 97.9 | 173.0 ms | 1.00× |
| FoldQuant | 39.8 | 4.5 | 54.7 | 119.1 ms | **1.45×** |

The tower is 3.0× faster and the 50-step loop 1.8×, but the end-to-end figure
is held back by what stays in PyTorch: the InternVL3 vision tower and the tile
fusion are ~35 ms of the embed bucket and neither is quantized here, so they
are most of the remaining time. This is the installation check — paper numbers
use 128 calibration observations and upstream's client.

Install flash-attn before reading these numbers. Without it upstream falls back
to standard attention, while the tower graph is emitted with
`padded_query_mask`, which reproduces what `flash_attention_2` does with a
padded prompt; the reference and the engine would then be answering different
questions at the padded positions.

## Files

| file | role |
|---|---|
| `calibration.py` | upstream model / normalizer loading, seeded sample plan, client-format requests, forward loop |
| `_dataset.py` | LeRobot v3 frame reader (parquet index; upstream's decoder for video features) |
| `export_foldquant.py` | scheme validation, shape capture, `export_llm` / `export_action_head`, manifests |
| `build_engines.py` | plugin load + `foldquant.runtime.builder.build_engine` per component |
| `runtime.py` | engine installer (`install_engines`), the tower rebind, the `get_action` loop, `ContextCapture` |
| `verify.py` | held-out PyTorch-vs-engine drift report |
| `serve.py` | upstream's websocket server with engines installed |
| `eval_libero.py` | LIBERO sweep through upstream's client, per-suite `summary.json` |
| `benchmark.py` | component and end-to-end timing over the PyTorch arm and engine directories |
| `_upstream.py` | paths, checkpoint and normalizer defaults, suite list, component table |
