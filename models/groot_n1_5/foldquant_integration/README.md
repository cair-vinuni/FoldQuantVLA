# FoldQuant on GR00T N1.5

This folder is the whole of what FoldQuant adds to the upstream GR00T N1.5
release. The `gr00t` package, its data path, its ZMQ inference service and
its LIBERO client (`examples/Libero/eval/run_libero_eval.py`) are used
unchanged; the algorithm, the ONNX emitters and the TensorRT plugins are the
top-level [`foldquant`](../../../foldquant) package.

Two modules of the policy are replaced by FoldQuant plugin graphs:

| module | attribute | graph | schemes |
|---|---|---|---|
| LLM (Eagle Qwen3 decoder, 12 kept layers) | `policy.model.backbone.eagle_model.language_model` | `llm_bf16.onnx` | `w8a8_sr` (default), `w8a8_s`, `w4a4_srg`, `w4a4_sg`, `w4a8_srg` |
| DiT (plain `DiT`, 16 layers) | `policy.model.action_head.model` | `dit_bf16.onnx` | `w4a4_shg` (default), `w4a4_sh`, `w4a4_sr`, `w8a8_sh`, `w8a8` |

N1.5 differs from N1.6/N1.7 in three places the integration has to know
about, all read off the loaded model rather than assumed:

- The Eagle wrapper pops the decoder down to `select_layer` layers at
  construction and reads `hidden_states[select_layer]` — the last entry,
  which under the pinned `transformers` 4.51.3 is the residual stream *after*
  the decoder's final norm. `export_foldquant` measures this
  (`llm_final_norm` in `export_metadata.json`) and emits the graph
  accordingly; the runtime returns the engine's `hidden_states` at exactly
  that index.
- The action head is a plain `DiT`: its forward takes no
  `image_mask`/`backbone_attention_mask` and every cross-attention block
  attends the whole encoder sequence. The FoldQuant DiT graph keeps the
  five-input contract shared with N1.6/N1.7; for a plain DiT the runtime
  feeds all-True masks and the emitter bakes an attend-all mask, which is
  what "no mask" means. The DiT's self-attention sequence is
  `1 + num_target_vision_tokens + action_horizon` (state token, learned
  future-vision token bank, action tokens) and is captured at export.
- `Gr00tPolicy` runs the model under `torch.autocast(bf16)`, so the DiT
  receives fp32 encoder states out of the action head's `vlln` LayerNorm.
  Calibration replays hand the module its inputs in its own dtype — the
  engine's input dtype — where autocast used to reconcile the two.

Upstream N1.5 ships its own TensorRT path (`deployment_scripts/`, fp16/fp8
per-module engines behind `scripts/inference_service.py --use-tensorrt`).
Its DiT graph has a different contract (`sa_embs`, `vl_embs`,
`timesteps_tensor`) and its LLM graph is fp16, so the two engine sets are not
interchangeable and there is no float-engine floor arm here; `verify`
compares against the bf16 PyTorch policy.

Engines are served by [`runtime.py`](runtime.py), which rebinds the two
modules' `forward`; nothing else in the policy changes, so the same
`Gr00tPolicy` runs float, partly or fully quantized, behind the same
`RobotInferenceServer`.

## Environment

The upstream pins apply (Python 3.10, torch 2.5.1, transformers 4.51.3,
flash-attn 2.7.1.post4). From this directory:

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install --upgrade setuptools wheel
uv pip install -e ".[base]"
uv pip install flash-attn==2.7.1.post4 --no-build-isolation   # or the matching release wheel
uv pip install "tensorrt-cu12==10.13.0.35"                     # upstream's `deploy` extra
uv pip install -e ../..                                        # the foldquant package into it
python -m foldquant.kernels build   # compile the plugin libraries for this GPU / TensorRT
```

`foldquant.kernels build` needs `nvcc` and the TensorRT headers; the result
is cached per `(SM, arch, TensorRT major.minor)` and looked up exactly, so a
different GPU or TensorRT build compiles its own copy rather than loading an
ABI-mismatched library. LIBERO (`libero` + `robosuite==1.4.0`) is only
needed by `eval_libero` and upstream's client; install it as upstream's
`examples/Libero/README.md` describes.

### LIBERO, for `eval_libero`

The rollout runs in this same environment, so the simulator has to live beside
the model. This release ships `examples/Libero` — the client loop — but pins no
LIBERO benchmark of its own, so the checkout is yours to supply and its
location is named by an environment variable rather than hard-coded:

```bash
uv pip install "robosuite==1.4.0" "mujoco==2.3.7" bddl easydict hydra-core einops termcolor thop gym
FOLDQUANT_LIBERO_DIR=/path/to/LIBERO python -m foldquant_integration.eval_libero ...
```

Both pins are required. `robosuite==1.4.0` is LIBERO's own: 1.5 moved
`robosuite.environments.manipulation.single_arm_env`, which LIBERO imports.
`mujoco==2.3.7` is required by that robosuite, which declares only
`mujoco>=2.3.0` and so resolves to 3.x — where the rollout dies inside
`robosuite.utils.binding_utils.get_joint_qpos_addr` on an assertion about
joint types. That one fails at the first `env.reset()`, not at import, so it
survives any check short of actually rolling an episode. Do
**not** install LIBERO's `requirements.txt` — it pins `numpy==1.22.4`,
`transformers==4.21.1` and `robomimic==0.2.0` and would tear out the stack the
policy runs on. The list above was resolved against this environment and
changes nothing else.

An already-installed `libero` satisfies this on its own and the variable is
then unnecessary. Note that `pip install -e` on a LIBERO checkout does not
count: it reports success while leaving nothing importable, because LIBERO
declares no dependencies and its `libero/` directory has no `__init__.py`.

## Workflow

Every step takes `--embodiment-tag` (the LIBERO post-trained checkpoints use
`new_embodiment`), `--data-config` (defaults to upstream's
`examples.Libero.custom_data_config:LiberoDataConfig`; the LIBERO-Goal
checkpoint needs `LiberoDataConfigMeanStd`) and `--denoising-steps`
(upstream serves LIBERO with 8; omitted, the checkpoint's own value is used).
Calibration, verification and serving must agree on these — they change the
tensors the model sees.

1. **Calibrate and emit the FoldQuant graphs.** Observations are drawn
   through the upstream data path (`LeRobotSingleDataset` with the
   checkpoint's own modality config and transforms) from a seeded,
   episode-balanced plan; the forward loop replays `policy.get_action` so
   every capture sees exactly the inference-time tensors.

   ```bash
   python -m foldquant_integration.export_foldquant \
       --model-path <checkpoint> --embodiment-tag new_embodiment --denoising-steps 8 \
       --dataset-path <calibration dataset> --num-calib 128 \
       --llm-scheme w8a8_sr --dit-scheme w4a4_shg \
       --output-dir exports/n15_w8a8_w4a4
   ```

   `--cascade` calibrates the DiT while the LLM runs under FoldQuant's
   fake-quant emulation of its own fold. `--llm-scheme none` / `--dit-scheme
   none` leaves a module in PyTorch. The export is seeded end to end (sample
   plan and flow-matching noise), so the same inputs emit byte-identical
   graphs; keep `--num-calib` at 128 or more for a GPTQ (`_g`) arm — see the
   N1.7 README for why 32 observations leave the DiT Hessians rank deficient.

   `--llm-params` / `--dit-params` take the fold's own knobs as JSON —
   `sq_alpha`, `act_clip_ratio`, `site_bits`, `rot_block_size`,
   `learned_calib`. The tuned arms built from them (`arc`, `res8`, `w4a8`) and
   the two scripts that select the values are in
   [`scripts/README.md`](../../../scripts/README.md). One trap worth knowing
   before reading a preset: `sq_fold_order` resolves to `before` whenever the
   scheme carries an FWHT (`_h`, `_sh`, `_shg`) and to `after` otherwise
   (`foldquant/export.py:100`), so a preset's `sq_fold_order: before` is a
   no-op on the action module's `_h` schemes and only changes an `_r` head.

2. **Build the engine directory.**

   ```bash
   python -m foldquant_integration.build_engines \
       --onnx-dir exports/n15_w8a8_w4a4/onnx --engine-dir exports/n15_w8a8_w4a4/engines
   ```

   Loads the plugin libraries, then compiles each FoldQuant graph strongly
   typed with the shape profiles derived from the graph and the captured
   shapes: batch 1, `sa_seq_len` static at `1 + num_target_vision_tokens +
   action_horizon` (49 for the LIBERO checkpoints), the LLM sequence and
   the DiT's `vl_seq_len` ranged `(1, captured, max(2 × captured, captured
   + 64))` (`--llm-max-seq-len`, `--vl-max-seq-len`).
   `foldquant_engines.json` records what was built from where;
   `foldquant_export.json` travels with the engines so the runtime tools
   know which plugin libraries to load.

3. **Verify, serve or evaluate, benchmark.**

   ```bash
   python -m foldquant_integration.verify --model-path ... --embodiment-tag new_embodiment \
       --denoising-steps 8 --dataset-path ... --engine-dir exports/n15_w8a8_w4a4/engines

   # upstream's client/server protocol, server side quantized:
   python -m foldquant_integration.serve --model-path ... --embodiment-tag new_embodiment \
       --denoising-steps 8 --engine-dir exports/n15_w8a8_w4a4/engines --port 5555
   MUJOCO_GL=egl python examples/Libero/eval/run_libero_eval.py \
       --task_suite_name libero_spatial --num_trials_per_task 20 --headless

   # the same rollout loop in process, every suite, resumable:
   MUJOCO_GL=egl python -m foldquant_integration.eval_libero --model-path ... \
       --embodiment-tag new_embodiment --denoising-steps 8 \
       --engine-dir exports/n15_w8a8_w4a4/engines --output exports/n15_w8a8_w4a4/libero

   python -m foldquant_integration.benchmark --model-path ... --embodiment-tag new_embodiment \
       --dataset-path ... --arms w4a4=exports/n15_w8a8_w4a4/engines
   ```

   `verify` scores backbone features and the decoded action chunk under
   seeded flow-matching noise on held-out observations from episodes the
   calibration never saw, and records every observation's drift in
   `verify.json`; `--split-from <quantized engine dir>` scores another
   directory on that arm's held-out set. `serve` is upstream's
   `RobotInferenceServer` around a policy with the engines installed, so
   upstream's client runs unchanged. `eval_libero` subclasses the client's
   own `GR00TPolicy` (observation and action conversion byte-identical to
   the served path) around the in-process policy, and runs upstream's
   episode loop — `num_steps_wait` no-op steps, per-suite step budgets —
   over every task of the requested suites with a resume-safe
   `summary.json`. `benchmark` times data processing, backbone, action head
   and the whole `get_action` for the PyTorch arm and each `--arms` engine
   directory in one process (upstream N1.5 has no timing script of its own).

Plugin graphs are emitted at the batch the calibration captured (1); the
upstream client sends one observation per request.

## Files

| file | role |
|---|---|
| `calibration.py` | upstream policy / dataset loading, seeded sample plan, observation building, forward loop |
| `export_foldquant.py` | scheme validation, shape + final-norm capture, `export_llm` / `export_dit`, manifests |
| `build_engines.py` | plugin load + `foldquant.runtime.builder.build_engine` per component |
| `runtime.py` | engine installer (`install_engines`), the two forward rebinds |
| `verify.py` | held-out PyTorch-vs-engine drift report |
| `serve.py` | upstream `RobotInferenceServer` with engines installed |
| `eval_libero.py` | LIBERO sweep over suites × tasks, per-task `summary.json` |
| `benchmark.py` | component and end-to-end timing over the PyTorch arm and engine directories |
| `_upstream.py` | paths, component table |

## Smoke check

`eval_libero` installation check (not a suite result): on the `w8a8` engines,
`--suites libero_spatial --n-episodes 2` completes **18/20**, ten tasks, zero
task failures, ~16 s per task on one RTX 4070 Ti SUPER. Two episodes per task
is an installation check — it says the rollout, the engines and a supplied
LIBERO checkout work together end to end, and nothing about success rate.
Paper numbers come from the full sweep.

`w8a8_sr` LLM + `w4a4_sh` DiT, 16 calibration observations, 8 held-out
observations, float64 cosines against the bf16 PyTorch policy (`verify`),
LIBERO four-suite checkpoint at its own 4 denoising steps, one RTX 4070 Ti
SUPER (sm89), TensorRT 10.13:

| seam | cos mean | cos min | note |
|---|---|---|---|
| `backbone_features` | 0.99972 | 0.99968 | per-token min 0.986 |
| action chunk (16 × 7) | 0.99862 | 0.99293 | max abs 0.318 |

`benchmark` on the same engines (10 iterations, median): PyTorch eager
53.8 ms (backbone 29.3, action head 20.5) → FoldQuant 34.8 ms (backbone 17.8,
action head 12.1), 1.55× end to end; the vision tower stays in PyTorch in
both arms. Paper numbers use 128 calibration observations and the server's
harness; this is the installation check.
