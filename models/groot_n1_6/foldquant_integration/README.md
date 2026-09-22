# FoldQuant on GR00T N1.6

This adapter integrates FoldQuant with GR00T N1.6.1, using upstream
data loading, LIBERO evaluation, and benchmarking. The shared
[`foldquant`](../../../foldquant) package provides quantization, ONNX export,
and TensorRT plugins.

Two modules of the policy are replaced by FoldQuant plugin graphs:

| module | attribute | graph | schemes |
|---|---|---|---|
| LLM (Eagle3 Qwen3 decoder, 16 kept layers) | `policy.model.backbone.model.language_model` | `llm_bf16.onnx` | `w8a8_sr` (default), `w8a8_s`, `w4a4_srg`, `w4a4_sg`, `w4a8_srg` |
| DiT (`AlternateVLDiT`, 32 layers) | `policy.model.action_head.model` | `dit_bf16.onnx` | `w4a4_shg` (default), `w4a4_sh`, `w4a4_sr`, `w8a8_sh`, `w8a8` |

The DiT graph keeps the I/O contract of upstream's `export_onnx_n1d6.py`
(`sa_embs`, `vl_embs`, `timestep`, `image_mask`, `backbone_attention_mask` →
`output`), so upstream's own DiT engine and the FoldQuant one are
interchangeable. Upstream N1.6 exports no LLM graph; the FoldQuant LLM graph
takes the Eagle wrapper's `inputs_embeds` and `attention_mask` and returns
`hidden_states`, the tensor the wrapper reads as `hidden_states[-1]`. Whether
that tensor is the residual stream before or after the decoder's final norm
depends on the installed `transformers`; `export_foldquant` measures it on
the loaded model (`llm_final_norm` in `export_metadata.json`) instead of
assuming, and emits the graph accordingly.

Engines are served by [`runtime.py`](runtime.py), which rebinds the two
modules' `forward` the way upstream's `standalone_inference_script.py` does
for its DiT engine; nothing else in the policy changes, so the same
`Gr00tPolicy` runs float, partly or fully quantized.

## Environment

The upstream pins apply (Python 3.10, torch 2.7.1, transformers 4.51.3,
TensorRT ≥ 10.14). From this directory:

```bash
uv sync --python 3.10           # upstream environment
source .venv/bin/activate
uv pip install -e ../..         # the foldquant package into it
python -m foldquant.kernels build   # compile the plugin libraries for this GPU / TensorRT
```

`foldquant.kernels build` needs `nvcc` and the TensorRT headers. Binaries are
cached per `(SM, arch, TensorRT major.minor)` and matched exactly, so another
GPU or TensorRT version compiles its own copy.

### LIBERO, for `eval_libero`

The rollout runs in this environment (no separate client process), so the
simulator lives beside the model. `pip install -e` on the pinned submodule
reports success but makes nothing importable (LIBERO declares
`install_requires=[]` and `libero/` has no `__init__.py`), so the integration
puts the submodule on `sys.path` itself. Install the simulator stack with these
pins:

```bash
git submodule update --init external_dependencies/LIBERO
uv pip install "robosuite==1.4.0" "mujoco==2.3.7" bddl easydict hydra-core einops termcolor thop gym
```

Both pins are required. `robosuite==1.4.0` is LIBERO's own (1.5 moved
`robosuite.environments.manipulation.single_arm_env`, which LIBERO imports).
That robosuite declares only `mujoco>=2.3.0`, which resolves to 3.x and fails
at the first `env.reset()` (an assertion on joint types in
`robosuite.utils.binding_utils.get_joint_qpos_addr`), not at import.
Do **not** install LIBERO's `requirements.txt`: it pins `numpy==1.22.4`,
`transformers==4.21.1` and `robomimic==0.2.0` and would replace the policy's
stack. The list above leaves torch, transformers and numpy untouched.

## Workflow

Every step takes `--embodiment-tag`. It may be omitted only when the
checkpoint's `processor_config.json` names a single embodiment; the LIBERO
fine-tune names four (`behavior_r1_pro`, `gr1`, `robocasa_panda_omron`,
`libero_panda`) and is refused without `--embodiment-tag libero_panda`, by
upstream's loader as much as by these tools.

1. **Calibrate and emit the FoldQuant graphs.** Observations are drawn
   through the upstream data path (`LeRobotEpisodeLoader` →
   `extract_step_data` → `parse_observation_gr00t`) from a seeded, episode-
   balanced plan; the forward loop replays `policy.get_action` so every
   capture sees exactly the inference-time tensors.

   ```bash
   python -m foldquant_integration.export_foldquant \
       --model-path checkpoints/GR00T-N1.6-LIBERO --embodiment-tag libero_panda \
       --dataset-path <calibration dataset> --num-calib 128 \
       --llm-scheme w8a8_sr --dit-scheme w4a4_shg \
       --output-dir exports/n16_w8a8_w4a4
   ```

   `--cascade` calibrates the DiT while the LLM runs under FoldQuant's
   fake-quant emulation of its own fold. `--llm-scheme none` / `--dit-scheme
   none` leaves a module in PyTorch. The export is seeded end to end (sample
   plan and flow-matching noise), so the same inputs emit byte-identical
   graphs. Keep `--num-calib` at 128 or more for a GPTQ (`_g`) arm; the N1.7
   README explains why 32 observations leave the DiT Hessians rank deficient.

   `--llm-params` / `--dit-params` take the fold's knobs as JSON (`sq_alpha`,
   `act_clip_ratio`, `site_bits`, `rot_block_size`, `learned_calib`). The
   tuned arms built from them (`arc`, `res8`, `w4a8`) and the two scripts that
   select the values are in [`scripts/README.md`](../../../scripts/README.md).
   Note that `sq_fold_order` resolves to `before` whenever the scheme carries an
   FWHT (`_h`, `_sh`, `_shg`) and to `after` otherwise
   (`foldquant/export.py:100`), so a preset's `sq_fold_order: before` is a
   no-op on the action module's `_h` schemes and only changes an `_r` head.

2. **Build the engine directory.**

   ```bash
   python -m foldquant_integration.build_engines \
       --onnx-dir exports/n16_w8a8_w4a4/onnx --engine-dir exports/n16_w8a8_w4a4/engines
   ```

   Loads the plugin libraries, then compiles each FoldQuant graph strongly
   typed with the shape profiles derived from the graph and the captured
   shapes: batch 1, `sa_seq_len` static at `1 + action_horizon` (51), the LLM
   sequence and the DiT's `vl_seq_len` ranged `(1, captured, max(2 ×
   captured, captured + 64))` (`--llm-max-seq-len`, `--vl-max-seq-len`). A
   DiT left float can be built from upstream's export instead:

   ```bash
   GR00T_ONNX_EXPORTER_MODE=legacy \
   python scripts/deployment/export_onnx_n1d6.py --model_path ... --dataset_path ... \
       --embodiment_tag libero_panda --output_dir exports/n16_float/onnx
   python -m foldquant_integration.build_engines --onnx-dir exports/n16_w8a8_none/onnx \
       --float-onnx-dir exports/n16_float/onnx --engine-dir exports/n16_w8a8_none/engines
   ```

   `GR00T_ONNX_EXPORTER_MODE=legacy` is upstream's own knob, set on purpose.
   This release passes `dynamo=use_dynamo_exporter` to the DiT export, which
   defaults to true off Spark, while N1.7 hard-codes `dynamo=False` for its DiT
   because *"DiT specializes vl_seq_len under dynamo; legacy exporter needed"*.
   N1.6's DiT is the same module family, so the float arm (the reference for
   every drift number here) is exported the N1.7 way.

   The dynamo path also needs `onnxscript`, which neither release declares.
   Installing it would make the export run and could return a
   sequence-length-specialised reference: worse than a missing package,
   because it succeeds.

   `--float-onnx-dir` alone (with `--metadata` pointing at any export's
   `export_metadata.json` for the shapes) gives the float-DiT-engine arm, the
   floor of the drift metric. `foldquant_engines.json` records what was built
   from where; `foldquant_export.json` travels with the engines so the runtime
   tools know which plugin libraries to load.

3. **Verify, evaluate, benchmark.**

   ```bash
   python -m foldquant_integration.verify --model-path ... --embodiment-tag libero_panda \
       --dataset-path ... --engine-dir exports/n16_w8a8_w4a4/engines
   MUJOCO_GL=egl python -m foldquant_integration.eval_libero --model-path ... \
       --engine-dir exports/n16_w8a8_w4a4/engines --n-envs 1 --output exports/n16_w8a8_w4a4/libero
   python -m foldquant_integration.benchmark --model-path ... --embodiment-tag libero_panda \
       --dataset-path ... --arms w4a4=exports/n16_w8a8_w4a4/engines
   ```

   `verify` scores backbone features and the decoded action chunk under
   seeded flow-matching noise on held-out observations from episodes the
   calibration never saw, and records every observation's drift in
   `verify.json`; `--split-from <quantized engine dir>` scores a float
   directory on that arm's held-out set. `eval_libero` runs upstream's
   `MultiStepWrapper` rollout (8-step chunks, 504-step cap, terminate on
   success) over every task of the requested suites with a resume-safe
   `summary.json`; upstream's factory has no engine hook, so the tool builds
   `Gr00tPolicy`, installs the engines and wraps it in `Gr00tSimPolicyWrapper`
   itself. `benchmark` times upstream's component loop for the PyTorch arm and
   each `--arms` engine directory in one process, and prints upstream's
   markdown table.

Plugin graphs are emitted at the batch the calibration captured (1), so
TensorRT arms run `--n-envs 1`; the PyTorch arm may batch.

## Serving

For a real robot the arm under test is a server: upstream's `PolicyServer`
answers over ZMQ, and [`serve.py`](serve.py) is that server with the FoldQuant
engines installed into the policy first. Nothing on the wire changes, so a
robot loop moves between the bf16 reference and a quantized arm by pointing at
a different port.

```bash
# terminal 1: the arm under test
python -m foldquant_integration.serve --model-path <ckpt> --embodiment-tag libero_panda \
    --engine-dir exports/w4a4/engines

# terminal 2: upstream's own client, unchanged
python -c "
from gr00t.policy.server_client import PolicyClient
client = PolicyClient(host='127.0.0.1', port=5555)
print(client.get_action(observation))
"
```

Omit `--engine-dir` to serve the bf16 PyTorch policy: the reference arm, in the
same process and protocol, so the two are comparable on hardware.

The same server drives upstream's simulation clients. For SimplerEnv (set up once
with `gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh`, see `examples/SimplerEnv/`),
add `--use-sim-policy-wrapper` and point upstream's client at the port:

```bash
# terminal 1: a FoldQuant arm of the Bridge checkpoint
python -m foldquant_integration.serve --model-path nvidia/GR00T-N1.6-bridge \
    --embodiment-tag OXE_WIDOWX --engine-dir exports/bridge_w4a4/engines \
    --use-sim-policy-wrapper --port 5555

# terminal 2: upstream's SimplerEnv client, unchanged
gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --policy_client_host 127.0.0.1 --policy_client_port 5555 \
    --env_name simpler_env_widowx/widowx_spoon_on_towel \
    --n_episodes 200 --n_envs 5 --n_action_steps 4 --max_episode_steps 300
```

[`eval_simpler.sh`](eval_simpler.sh) wraps the two: one invocation per arm starts
the server, sweeps the seven Bridge tasks with upstream's client and writes one
log per task plus `summary.tsv` under `exports/simpler/<ARM>/`:

```bash
ARM=bf16 bash foldquant_integration/eval_simpler.sh                       # reference
ARM=w4a4 ENGINE_DIR=exports/bridge_w4a4/engines bash foldquant_integration/eval_simpler.sh
# N_EPISODES, N_ENVS, TASKS, MODEL_PATH, EMBODIMENT, PORT override the defaults.
```

**Build the engines for the batch the client sends.** `rollout_policy.py`
vectorises the environment (`--n_envs 5` above) and sends that many
observations at once, while `build_engines` pins the batch profile to 1 by
default (what a robot client and the drift protocol need). A batch outside the
compiled profile is refused at inference:

```
TensorRTEngine: input 'inputs_embeds' axis 0 = 5 is outside the compiled
profile bounds [1, 1]. Shape profiles are fixed at compile time
```

`--max-batch` widens it for the **LLM**, whose graph carries a symbolic batch
axis, at the cost of a rebuild and no re-export:

```bash
python -m foldquant_integration.build_engines \
    --onnx-dir exports/bridge_w4a4/onnx --engine-dir exports/bridge_w4a4/engines_b5 \
    --max-batch 5
```

The **DiT** graph pins batch to a literal 1 on purpose: `dit_common.py` makes
only `sa_seq_len` and `vl_seq_len` symbolic, so upstream's
`export_metadata.json` shape hints apply to a FoldQuant DiT graph unchanged. No
rebuild can widen a static dimension, so a vectorised client hits the same
refusal one engine later, on `sa_embs`. Until the DiT is exported with a
dynamic batch axis, sweep a quantized arm with `N_ENVS=1`:

```bash
ARM=w4a4 ENGINE_DIR=exports/bridge_w4a4/engines N_ENVS=1 \
    bash foldquant_integration/eval_simpler.sh
```

Compare arms at the same `N_ENVS`; the bf16 policy accepts any batch, so hold
it to the engine's terms.

Setup notes: `setup_SimplerEnv.sh` expects the SimplerEnv checkout at
`external_dependencies/SimplerEnv` (upstream pins it as a submodule; this release
does not, so clone it there first) and pins `setuptools<81`, which SAPIEN needs
for `pkg_resources`. SAPIEN renders through Vulkan, so the host needs a GPU
visible to Vulkan: MIG instances expose compute only and cannot run it.

The release ships no standalone client script; `PolicyClient` is the class its
real-robot evaluators construct (`gr00t/eval/real_robot/SO100`).


## Files

| file | role |
|---|---|
| `calibration.py` | upstream policy / dataset loading, seeded sample plan, observation building, forward loop |
| `export_foldquant.py` | scheme validation, shape + final-norm capture, `export_llm` / `export_dit`, manifests |
| `build_engines.py` | plugin load + `foldquant.runtime.builder.build_engine` per component; float DiT from upstream's export |
| `runtime.py` | engine installer (`install_engines`), the two forward rebinds |
| `verify.py` | held-out PyTorch-vs-engine drift report |
| `serve.py` | upstream's ZMQ `PolicyServer` with the engines installed |
| `eval_libero.py` | LIBERO sweep over suites × tasks, per-task `summary.json` |
| `eval_simpler.sh` | SimplerEnv Bridge sweep for one arm: server + upstream client over seven tasks, `summary.tsv` |
| `benchmark.py` | upstream timing loop over the PyTorch arm and engine directories |
| `_upstream.py` | paths, component table |

## Smoke check

`eval_libero` installation check (not a suite result): on the `w8a8` engines,
`--suites libero_spatial --n-episodes 2` completes **20/20** (ten tasks, zero
task failures, ~12 s per task on one RTX 4070 Ti SUPER). This shows the
rollout, the engines and the LIBERO environment work together end to end, and says
nothing about success rate. Paper
numbers come from the full sweep.

`w8a8_sr` LLM + `w4a4_sh` DiT, 16 calibration observations, 8 held-out
observations, float64 cosines against the bf16 PyTorch policy (`verify`),
one RTX 4070 Ti SUPER (sm89), TensorRT 10.16:

| seam | cos mean | cos min | note |
|---|---|---|---|
| `backbone_features` | 0.99965 | 0.99913 | per-token min 0.931 |
| action chunk (50 × 7) | 0.99957 | 0.99870 | max abs 0.066 |

`benchmark` on the same engines (4 denoising steps, 10 iterations): PyTorch
eager 69 ms (backbone 27, action head 38) → FoldQuant 37 ms (backbone 18,
action head 14), 1.88× end to end. Paper numbers use 128 calibration
observations and the server's harness; this is the installation check.

## Device memory per arm

`memory.py` measures one arm in one fresh process and reports two numbers to
read together:

* **as served** (`--keep-replaced-weights`): what `serve` holds, with the
  replaced PyTorch weights still resident on the GPU;
* **floor** (default for an engine arm): the engines plus the PyTorch
  components the runtime still executes (the vision tower, the LLM embedding table and the action encoders/decoder).

The value is the median `cudaMemGetInfo` plateau (total minus free) over 60
timed calls after 10 warm-ups; `memory --help` gives the details. Run the eager
arm first so the engine arms can report their decoded-action cosine against it:

```bash
python -m foldquant_integration.memory --model-path <ckpt> --embodiment-tag libero_panda --dataset-path <LIBERO calib> --reference-actions ref.npz
python -m foldquant_integration.memory --model-path <ckpt> --embodiment-tag libero_panda --dataset-path <LIBERO calib> --engine-dir exports/<arm>/engines --reference-actions ref.npz
python -m foldquant_integration.memory --model-path <ckpt> --embodiment-tag libero_panda --dataset-path <LIBERO calib> --engine-dir exports/<arm>/engines --keep-replaced-weights
```
