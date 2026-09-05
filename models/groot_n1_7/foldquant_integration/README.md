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

### LIBERO, for `eval_libero`

The rollout runs in this same environment (there is no separate client
process), so the simulator has to live beside the model. `pip install -e` on
the pinned submodule is **not** enough and misleadingly reports success:
LIBERO declares `install_requires=[]`, and its `libero/` directory carries no
`__init__.py`, so nothing becomes importable. The integration puts the
submodule on `sys.path` itself; what has to be installed is the simulator
stack, and one version pin matters:

```bash
git submodule update --init external_dependencies/LIBERO
uv pip install "robosuite==1.4.0" "mujoco==2.3.7" bddl easydict hydra-core einops termcolor thop gym
```

Both pins are required. `robosuite==1.4.0` is LIBERO's own: 1.5 moved
`robosuite.environments.manipulation.single_arm_env`, which LIBERO imports.
`mujoco==2.3.7` is required by that robosuite, which declares only
`mujoco>=2.3.0` and so resolves to 3.x — where the rollout dies inside
`robosuite.utils.binding_utils.get_joint_qpos_addr` on an assertion about
joint types. That one fails at the first `env.reset()`, not at import, so it
survives any check short of actually rolling an episode.
Do **not** install LIBERO's `requirements.txt` — it pins `numpy==1.22.4`,
`transformers==4.21.1` and `robomimic==0.2.0`, which would tear out the stack
the policy runs on. The list above was resolved against this environment and
changes nothing else: torch, transformers and numpy are untouched.

## Workflow

Every step takes `--embodiment-tag`. It may be omitted only when the
checkpoint's `processor_config.json` names a single embodiment; a fine-tune
that names several (the LIBERO 4-suite checkpoint lists nine) is refused
without it, by upstream's loader as much as by these tools.

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

   The export is reproducible: the sample plan and the flow-matching noise
   `get_action` starts from are both driven by `--seed`, so the same
   checkpoint, dataset and arguments emit byte-identical plugin graphs (two
   `w4a4_shg` exports compared attribute by attribute). Seeding also keeps
   the GPTQ pass on the very activations the SmoothQuant pass fitted.

   Keep `--num-calib` at 128 or more for a GPTQ arm. The DiT's widest site
   (FFN `proj2`, K = 6144) sees 41 action-stream tokens × 4 denoising steps
   per observation, so 32 observations give a 5248-row Hessian — rank
   deficient at every block — and the fold then tracks whichever noise it
   happened to see: two unseeded 32-observation `w4a4_shg` exports differed
   in 17–37 % of their INT4 weight bytes and by 0.003 in mean action cosine
   on the same held-out set. 128 observations give 21k rows.

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
   observations from episodes the calibration never saw, and records every
   observation's drift in `verify.json`; `--split-from <quantized engine
   dir>` scores a float directory on that arm's held-out set. `eval_libero` runs
   the upstream `MultiStepWrapper` rollout over every task of the requested
   suites with a resume-safe `summary.json`. `benchmark` and `rollout` are
   upstream's own scripts with the plugin library preloaded — the arguments
   are theirs.

Plugin graphs are emitted at the batch the calibration captured (1), so
TensorRT arms run `--n-envs 1`; the PyTorch arm may batch.

## Smoke check

`eval_libero` installation check (not a suite result): on the `w8a8` engines,
`--suites libero_spatial --n-episodes 2` completes **20/20**, ten tasks, zero
task failures, ~12 s per task on one RTX 4070 Ti SUPER. Two episodes per task
is an installation check — it says the rollout, the engines and the LIBERO
environment work together end to end, and nothing about success rate. Paper
numbers come from the full sweep.

The chain above, run end to end on the LIBERO 4-suite fine-tune
(`--embodiment-tag libero_sim`, RTX 4070 Ti SUPER / sm89, TensorRT 10.15):
LLM `w8a8_sr`, `--cascade`, 128 calibration observations, scored by
`verify` on 32 held-out observations from episodes the calibration never
saw. The float engines are scored on the same observations
(`--split-from`) — they are the floor of the drift metric, not zero:

| engines | backbone cos mean / min | actions cos mean / min | actions max abs |
|---|---|---|---|
| float (upstream bf16) | 0.99997 / 0.99991 | 0.99977 / 0.99415 | 0.436 |
| LLM `w8a8_sr` + DiT `w4a4_sh` | 0.99996 / 0.99990 | 0.99950 / 0.99653 | 0.330 |
| LLM `w8a8_sr` + DiT `w4a4_shg` | 0.99996 / 0.99990 | 0.99951 / 0.99184 | 0.504 |

Two readings. Quantization costs 0.00026 of mean action cosine over the
float engines; four Euler steps in bf16 already move single actions by 0.4
in the float engines themselves, so `max_abs` on a handful of observations
does not separate the arms. And on this DiT the GPTQ pass (`_shg`) does not
improve on round-to-nearest (`_sh`) at 128 observations — the means tie and
its worst observation is worse. Success rate under the upstream LIBERO
harness (`eval_libero`) is what decides between them.

Cost on that GPU: kernels build 30 s; upstream float export + engines
3.5 min; `export_foldquant` LLM `w8a8_sr` 17 s, DiT `w4a4_sh` 20 s, DiT
`w4a4_shg` 20 min (the 129 Hessians are accumulated on the host in float64 —
they do not fit next to the model on 16 GB); `build_engines` 47 s; `verify`
40 s for 32 observations.

## Serving

For a real robot the arm under test is a server: upstream's `PolicyServer`
answers over ZMQ, and [`serve.py`](serve.py) is that server with the FoldQuant
engines installed into the policy first. Nothing on the wire changes, so a
robot loop moves between the bf16 reference and a quantized arm by pointing at
a different port.

```bash
# terminal 1 — the arm under test
python -m foldquant_integration.serve --model-path <ckpt> \
    --embodiment-tag libero_panda --engine-dir exports/w4a4/engines

# terminal 2 — upstream's own client, unchanged
python -c "
from gr00t.policy.server_client import PolicyClient
client = PolicyClient(host='127.0.0.1', port=5555)
print(client.get_action(observation))
"
```

The released LIBERO checkpoint carries nine embodiments in its
`processor_config.json`, so `--embodiment-tag` is required for it; the tag
`libero_panda` resolves to the checkpoint's `libero_sim` entry. A
single-embodiment checkpoint needs no tag.

Omit `--engine-dir` to serve the bf16 PyTorch policy — the reference arm, same
process and same protocol, which is what makes the two comparable on hardware.
The release ships no standalone client script; `PolicyClient` is the class its
real-robot evaluators construct (`gr00t/eval/real_robot/SO100`).

Engines are installed by upstream's own `trt_model_forward.setup_tensorrt_engines`,
the same call `verify` and `benchmark` use — this family has no `runtime.py`,
because the release ships the whole pipeline swap itself and a second install
path could drift from it silently.


## Files

| file | role |
|---|---|
| `calibration.py` | upstream policy / dataset loading, seeded sample plan, observation building, forward loop |
| `export_foldquant.py` | scheme validation, shape-metadata capture, `export_llm` / `export_dit`, manifests |
| `build_engines.py` | plugin load + upstream `build_engine` per component; float completion |
| `verify.py` | held-out PyTorch-vs-engine drift report |
| `serve.py` | upstream's ZMQ `PolicyServer` with the engines installed |
| `eval_libero.py` | LIBERO sweep over suites × tasks, per-task `summary.json` |
| `rollout.py`, `benchmark.py` | upstream tools with plugins preloaded |
| `_upstream.py`, `_runpy.py` | paths, component table, `runpy` hand-off |
