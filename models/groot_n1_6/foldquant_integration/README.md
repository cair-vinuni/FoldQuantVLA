# FoldQuant on GR00T N1.6

This folder is the whole of what FoldQuant adds to the upstream GR00T N1.6.1
release. The `gr00t` package, its data path, its LIBERO rollout loop and the
timing loop of `scripts/deployment/benchmark_inference.py` are used unchanged;
the algorithm, the ONNX emitters and the TensorRT plugins are the top-level
[`foldquant`](../../../foldquant) package.

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

   `GR00T_ONNX_EXPORTER_MODE=legacy` is upstream's own knob and is set on
   purpose. This release passes `dynamo=use_dynamo_exporter` to the DiT export
   and that defaults to true off Spark, while the N1.7 release hard-codes
   `dynamo=False` for its DiT with the reason in the source: *"DiT specializes
   vl_seq_len under dynamo; legacy exporter needed"*. N1.6's DiT is the same
   module family, so the float arm — which is the reference every drift number
   here is measured against — is exported the way N1.7 exports its DiT rather
   than the way this script defaults.

   The dynamo path also wants `onnxscript`, which neither release declares and
   no environment here installs; that absence is what surfaces the difference,
   but it is not the reason for the choice. Installing it would make the export
   run and could hand back a sequence-length-specialised reference, which is a
   worse failure than a missing package because it succeeds.

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
# terminal 1 — the arm under test
python -m foldquant_integration.serve --model-path <ckpt> --embodiment-tag libero_panda \
    --engine-dir exports/w4a4/engines

# terminal 2 — upstream's own client, unchanged
python -c "
from gr00t.policy.server_client import PolicyClient
client = PolicyClient(host='127.0.0.1', port=5555)
print(client.get_action(observation))
"
```

Omit `--engine-dir` to serve the bf16 PyTorch policy — the reference arm, same
process and same protocol, which is what makes the two comparable on hardware.

The same server drives upstream's simulation clients. For SimplerEnv (set up once
with `gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh`, see `examples/SimplerEnv/`),
add `--use-sim-policy-wrapper` and point upstream's client at the port:

```bash
# terminal 1 — a FoldQuant arm of the Bridge checkpoint
python -m foldquant_integration.serve --model-path nvidia/GR00T-N1.6-bridge \
    --embodiment-tag OXE_WIDOWX --engine-dir exports/bridge_w4a4/engines \
    --use-sim-policy-wrapper --port 5555

# terminal 2 — upstream's SimplerEnv client, unchanged
gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --policy_client_host 127.0.0.1 --policy_client_port 5555 \
    --env_name simpler_env_widowx/widowx_spoon_on_towel \
    --n_episodes 200 --n_envs 5 --n_action_steps 4 --max_episode_steps 300
```

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
| `benchmark.py` | upstream timing loop over the PyTorch arm and engine directories |
| `_upstream.py` | paths, component table |

## Smoke check

`eval_libero` installation check (not a suite result): on the `w8a8` engines,
`--suites libero_spatial --n-episodes 2` completes **20/20**, ten tasks, zero
task failures, ~12 s per task on one RTX 4070 Ti SUPER. Two episodes per task
is an installation check — it says the rollout, the engines and the LIBERO
environment work together end to end, and nothing about success rate. Paper
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
