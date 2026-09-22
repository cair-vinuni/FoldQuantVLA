# FoldQuant on GR00T N1.7

This adapter integrates FoldQuant with GR00T N1.7, using upstream
data loading, LIBERO evaluation, and TensorRT deployment. The shared
[`foldquant`](../../../foldquant) package provides quantization, ONNX export,
and TensorRT plugins.

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

For comparison, either module can instead take an NVIDIA ModelOpt INT8
SmoothQuant Q/DQ graph (`modelopt_w8a8_smoothquant`) under the same file name
and I/O contract; see [ModelOpt INT8 SmoothQuant baseline](#modelopt-int8-smoothquant-baseline).

## Environment

The upstream pins apply (Python 3.10, torch 2.7.1, transformers 4.57.3,
TensorRT 10.15). From this directory:

```bash
uv sync                         # upstream environment
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
checkpoint's `processor_config.json` names a single embodiment; a fine-tune
that names several (the LIBERO 4-suite checkpoint lists nine) is refused
without it, by upstream's loader as much as by these tools.

1. **Float export and engines (upstream).** Everything FoldQuant does not
   replace (ViT, VL self-attention, state/action encoders, action decoder)
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

   `--llm-params` / `--dit-params` take the fold's knobs as JSON (`sq_alpha`,
   `act_clip_ratio`, `site_bits`, `rot_block_size`, `learned_calib`). The
   tuned arms built from them (`arc`, `res8`, `w4a8`) and the two scripts that
   select the values are in [`scripts/README.md`](../../../scripts/README.md).
   Note that `sq_fold_order` resolves to `before` whenever the scheme carries an
   FWHT (`_h`, `_sh`, `_shg`) and to `after` otherwise
   (`foldquant/export.py:100`), so a preset's `sq_fold_order: before` is a
   no-op on the action module's `_h` schemes and only changes an `_r` head.

   The export is reproducible: `--seed` drives both the sample plan and the
   flow-matching noise `get_action` starts from, so the same checkpoint,
   dataset and arguments emit byte-identical plugin graphs (two `w4a4_shg`
   exports compared attribute by attribute), and the GPTQ pass sees the same
   activations the SmoothQuant pass fitted.

   Keep `--num-calib` at 128 or more for a GPTQ arm. The DiT's widest site
   (FFN `proj2`, K = 6144) sees 41 action-stream tokens × 4 denoising steps
   per observation, so 32 observations give a 5248-row Hessian, rank deficient
   at every block, and the fold tracks whatever noise it saw: two unseeded
   32-observation `w4a4_shg` exports differed in 17-37 % of their INT4 weight
   bytes and by 0.003 in mean action cosine on the same held-out set. 128
   observations give 21k rows.

3. **Build the engine directory.**

   ```bash
   python -m foldquant_integration.build_engines \
       --onnx-dir exports/n17_w8a8_w4a4/onnx --engine-dir exports/n17_w8a8_w4a4/engines \
       --float-onnx-dir exports/n17_float/onnx --float-engine-dir exports/n17_float/engines
   ```

   Loads the plugin library, compiles each FoldQuant graph with upstream's
   `build_engine` (strongly typed, same shape profiles), and fills the
   remaining components from the float directory, building their ONNX when
   given or copying their `.engine`. The result is a complete
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
   upstream's own scripts (and arguments) with the plugin library preloaded.

Plugin graphs are emitted at the batch the calibration captured (1), so
TensorRT arms run `--n-envs 1`; the PyTorch arm may batch.

## Smoke check

`eval_libero` installation check (not a suite result): on the `w8a8` engines,
`--suites libero_spatial --n-episodes 2` completes **20/20** (ten tasks, zero
task failures, ~12 s per task on one RTX 4070 Ti SUPER). This shows the
rollout, the engines and the LIBERO environment work together end to end, and says
nothing about success rate. Paper
numbers come from the full sweep.

The chain above, run end to end on the LIBERO 4-suite fine-tune
(`--embodiment-tag libero_sim`, RTX 4070 Ti SUPER / sm89, TensorRT 10.15):
LLM `w8a8_sr`, `--cascade`, 128 calibration observations, scored by
`verify` on 32 held-out observations from episodes the calibration never
saw. The float engines are scored on the same observations
(`--split-from`); they are the floor of the drift metric, not zero:

| engines | backbone cos mean / min | actions cos mean / min | actions max abs |
|---|---|---|---|
| float (upstream bf16) | 0.99997 / 0.99991 | 0.99977 / 0.99415 | 0.436 |
| LLM `w8a8_sr` + DiT `w4a4_sh` | 0.99996 / 0.99990 | 0.99950 / 0.99653 | 0.330 |
| LLM `w8a8_sr` + DiT `w4a4_shg` | 0.99996 / 0.99990 | 0.99951 / 0.99184 | 0.504 |

Quantization costs 0.00026 of mean action cosine over the float engines.
Four bf16 Euler steps already move single actions by 0.4 in the float engines,
so `max_abs` on a few observations does not separate the arms. On this DiT
the GPTQ pass (`_shg`) does not beat round-to-nearest (`_sh`) at 128
observations: the means tie and its worst observation is worse. Success rate
under the upstream LIBERO harness (`eval_libero`) decides between them.

Cost on that GPU: kernels build 30 s; upstream float export + engines
3.5 min; `export_foldquant` LLM `w8a8_sr` 17 s, DiT `w4a4_sh` 20 s, DiT
`w4a4_shg` 20 min (the 129 Hessians are accumulated on the host in float64;
they do not fit next to the model on 16 GB); `build_engines` 47 s; `verify`
40 s for 32 observations.

## Serving

On a Jetson AGX Orin, [`docs/deploy/jetson_serve.md`](../../../docs/deploy/jetson_serve.md)
walks through building and serving an arm on the board, and
[`scripts/deploy_groot_n17_jetson.sh`](../../../scripts/deploy_groot_n17_jetson.sh)
does it in one command.

For a real robot the arm under test is a server: upstream's `PolicyServer`
answers over ZMQ, and [`serve.py`](serve.py) is that server with the FoldQuant
engines installed into the policy first. Nothing on the wire changes, so a
robot loop moves between the bf16 reference and a quantized arm by pointing at
a different port.

```bash
# terminal 1: the arm under test
python -m foldquant_integration.serve --model-path <ckpt> \
    --embodiment-tag libero_panda --engine-dir exports/w4a4/engines

# terminal 2: upstream's own client, unchanged
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

Omit `--engine-dir` to serve the bf16 PyTorch policy: the reference arm, in the
same process and protocol, so the two are comparable on hardware.
The release ships no standalone client script; `PolicyClient` is the class its
real-robot evaluators construct (`gr00t/eval/real_robot/SO100`).

Engines are installed by upstream's own `trt_model_forward.setup_tensorrt_engines`,
the same call `verify` and `benchmark` use. This family has no `runtime.py`:
the release ships the whole pipeline swap itself, and a second install path
could drift from it.


## ModelOpt INT8 SmoothQuant baseline

`modelopt_w8a8_smoothquant` is not a FoldQuant fold. It reproduces the authors' framework
preset `groot_n1_7/tensorrt/modelopt_w8a8_smoothquant`, so a FoldQuant arm and
the ModelOpt baseline can be built, verified and served by the same tools and
compared on a robot. It needs two extra packages in the family environment,
which add to it without changing any pinned version:

```bash
uv pip install --python .venv/bin/python nvidia-modelopt==0.45.0 onnx-graphsurgeon==0.6.1
```

```bash
python -m foldquant_integration.export_foldquant --model-path ... --dataset-path ... \
    --embodiment-tag ... --num-calib 64 --seed 0 \
    --llm-scheme modelopt_w8a8_smoothquant --dit-scheme modelopt_w8a8_smoothquant \
    --output-dir exports/modelopt_w8a8_sq
python -m foldquant_integration.build_engines \
    --onnx-dir exports/modelopt_w8a8_sq/onnx --engine-dir exports/modelopt_w8a8_sq/engines \
    --float-onnx-dir exports/float/onnx --float-engine-dir exports/float/engines
```

What the arm does, step for step with the preset (`foldquant/modelopt_int8.py`,
`modelopt_export.py`):

| step | this arm |
|---|---|
| calibration data | `--num-calib` observations (the preset uses 64), `torch.manual_seed(seed + i)` before each; one bf16 policy replay captures every call into the LLM and every DiT denoising step |
| config | `mtq.INT8_SMOOTHQUANT_CFG`: per-channel INT8 weights, per-tensor static INT8 activations, SmoothQuant pre-quant scales |
| excluded leaves | Linear / Conv whose name matches `*norm*`, `*layernorm*`, `*final_action*`, `*action_proj*` |
| quantization | `mtq.quantize` on the live module, calibrated by replaying its captured calls; the DiT sees float-LLM activations (no cascade) |
| export | legacy TorchScript exporter, opset 20 (`--modelopt-opset`), dtype repairs for TensorRT's parser, export refused when no Q/DQ node survived |
| graph outputs | ModelOpt dequantizes to float32, so `embeddings` / `output` would be float32; a final `Cast` to bf16 keeps upstream's contract (the framework's engines output float32 and its runtime casts at the next engine's bf16 input, which is the same arithmetic) |
| engine | strongly-typed network (the provider records `builder_flags: {strongly_typed: true}`), built by upstream's `build_engine` like every other graph; the other five components stay upstream bf16 |

Differences that remain: the graphs use upstream's I/O names (the framework's own
engines do not load into `trt_model_forward`), the LLM wrapper is upstream's
`LLMForExport` over the live layers, and the engine directory is served by
upstream's pipeline swap rather than the framework's runtime.

Measured on a GR00T N1.7 SO101 checkpoint (Jetson AGX Orin, 64 calibration
samples, the 32 held-out samples of the `w8a8` arm via `verify --split-from`):
backbone cosine 0.99974, action cosine mean 0.9986 (min 0.9915). The same
recipe is close to lossless on this checkpoint. The framework's own `dit.onnx` for
this preset, renamed to upstream's I/O and built and verified here, scores
0.9986 as well, and its SmoothQuant vectors match this arm's (cosine >= 0.97
per layer), so the two graphs agree.

When comparing against an engine served from a framework artifact, check that
artifact's `embodiments/<tag>/action_schema.json`: a `clip_range` of
`[-3.14159, 3.14159]` with `units: rad` is applied to every action channel. On
a checkpoint whose actions are in degrees (SO101) that clips the joints to
+-3.14, and the artifact scores about 0.84 action cosine against the PyTorch
policy with or without quantization (its float TensorRT artifact measured
0.839, its ModelOpt INT8 SmoothQuant artifact 0.840). Differences observed on
the robot between the two stacks can come from that clip rather than from
quantization.

`--cascade` and `--llm-params` / `--dit-params` are refused for this scheme.
The export compiles ModelOpt's CUDA fake-quant extension on first use (about
95 s on an Orin, cached in `~/.cache/torch_extensions`); without it the ONNX
trace segfaults, so the export fails up front when it cannot be built.
`foldquant_export.json` records, per module, the config, the excluded leaves,
the number of enabled quantizers, the Q/DQ node counts and the repairs made.

## W4A4 baselines for comparison (emulated): HoloQ-style and DuQuant-style

Two published W4A4 recipes are shipped as **emulated** arms so the closed-loop
comparison against the FoldQuant engines can be rerun from this release alone.
They are post-training fake quantization of the same 112 LLM + 192 DiT
projection Linears (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj` in the
16 language layers; `attn1.{to_q,to_k,to_v,to_out.0}`, `ff.net.{0.proj,2}` in the
32 DiT blocks): INT4 codes for both operands are dequantised before a BF16
`F.linear`. No INT4 kernel runs, so these arms carry **no latency claim**; they
measure what each recipe's rounding does to the policy. AdaLN modulation, the
`vl_self_attention` blocks and the cross-attention encoder KV stay BF16 (the
FoldQuant engines quantise those too).

| | `--method holoq` (HoloQ-VLA style) | `--method duquant` (DuQuant style) |
|---|---|---|
| reference | HoloQ-VLA, arXiv 2605.28803 | DuQuant, as the baseline of HoloQ-VLA Table 2 |
| input permutation | zigzag over input-channel weight energy, blocks of 64 | same |
| block rotation (64×64) | `U · H_s`: left singular vectors of the weight block ᵀ times a sign-randomised normalised Hadamard | `U` alone: the eigenvectors of `WᵀW` per block (HoloQ-VLA `rot_mode=svd`) |
| LLM weights | GPTQ, block 128, damping 0.01, per-output-channel INT4 | same |
| DiT weights | RTN, per-output-channel INT4 | same |
| LLM activations | dynamic per-token INT4 (`max|x|` of the token) | **static per-channel** INT4: q99.9 of each channel over the calibration tokens, running max across observations, frozen |
| DiT activations | static per-denoising-step per-channel INT4 (q99.9) | static per-channel INT4 (one table, no step dependence) |
| calibration | 128 seeded dataset observations, seed 0 (same sampler as `export_foldquant`) | same |

The two arms differ **only** in the rotation and in the activation-scale rule;
solvers, scope, permutation, block sizes and calibration data are identical, so
the DuQuant row isolates those two choices. The static per-channel rule is what
HoloQ-VLA's DuQuant layers do (`PercentileCalibrator`: per-channel quantile over
tokens, running max, then `scale = q / 7`); applying a per-token dynamic scale
after the SVD-only rotation instead collapses the policy to 0 % success on every
LIBERO suite, because `U` concentrates a token's energy into one channel per block
and the remaining channels round to zero. DuQuant's output-row rotation
(`ROW_ROT=restore`) is not ported; it is mathematically weight-only.

Code in `foldquant_integration/baselines/`: `packing.py` (permutation, rotations,
INT4 packing, GPTQ), `scope.py` (the 304-Linear scope), `calibration.py`
(collector), `builder.py` (pack), `runtime.py` (emulated layers), `context.py`
(denoising-step context attached from outside the vendored model). Ported from the
authors' Isaac-GR00T fork (where the HoloQ-style port was first written),
with the static per-channel activation path added; the vendored upstream tree is
untouched.

```bash
# 1. calibrate + build + 8-observation action-cosine check (one call), per suite checkpoint
python -m foldquant_integration.baseline_w4a4 --command all --method duquant \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 --embodiment-tag libero_sim \
    --dataset-path data/libero_10_no_noops_1.0.0_lerobot --num-calib 128 --seed 0 \
    --output-dir exports/baseline_duquant_libero_10
#    -> exports/baseline_duquant_libero_10/{calibration.pt, pack.pt, pack.pt.sha256, check.json}
#    --method holoq for the HoloQ-style arm.

# 2a. closed loop, in process (the paper's baseline-comparison protocol)
MUJOCO_GL=egl python -m foldquant_integration.eval_libero \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --baseline-pack exports/baseline_duquant_libero_10/pack.pt \
    --suites libero_10 --n-episodes 20 --n-envs 1 --n-action-steps 8 --max-episode-steps 720 \
    --output exports/baseline_duquant_libero_10/libero

# 2b. or serve it to upstream's client
python -m foldquant_integration.serve --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --embodiment-tag libero_sim --baseline-pack exports/baseline_duquant_libero_10/pack.pt
```

`--baseline-pack` and `--engine-dir` are mutually exclusive. The pack is bound to
the checkpoint it was calibrated on (`config_sha256` guard) and to its denoising
step count. `check.json` reports the action cosine of the emulated arm against the
bf16 policy on held-out observations; a pack whose median is far below 0.99 is not
worth 800 episodes.

Known results (NVIDIA per-suite `nvidia/GR00T-N1.7-LIBERO` checkpoints, 10 tasks × 20
episodes per suite, `n_action_steps` 8, cap 720; successes of 200):

| arm | spatial | object | goal | long | total /800 |
|---|---:|---:|---:|---:|---:|
| BF16 PyTorch | 197 | 197 | 185 | 187 | 766 |
| HoloQ-style W4A4 (emulated, this release) | 188 | 197 | 179 | 180 | 744 |
| DuQuant-style W4A4 (emulated, this release) | 194 | 195 | 190 | 167 | 746 |
| FoldQuant W4A4 (INT4 engines) | 197 | 194 | 191 | 177 | 759 |
| FoldQuant W4A4 + o/d INT8 | 193 | 197 | 188 | 187 | 765 |

## Files

| file | role |
|---|---|
| `calibration.py` | upstream policy / dataset loading, seeded sample plan, observation building, forward loop |
| `export_foldquant.py` | scheme validation, shape-metadata capture, `export_llm` / `export_dit`, manifests |
| `modelopt_export.py` | ModelOpt INT8 SmoothQuant baseline: live-module quantization, upstream-contract Q/DQ export |
| `build_engines.py` | plugin load + upstream `build_engine` per component; float completion |
| `verify.py` | held-out PyTorch-vs-engine drift report |
| `serve.py` | upstream's ZMQ `PolicyServer` with the engines installed |
| `eval_libero.py` | LIBERO sweep over suites × tasks, per-task `summary.json` |
| `rollout.py`, `benchmark.py` | upstream tools with plugins preloaded |
| `_upstream.py`, `_runpy.py` | paths, component table, `runpy` hand-off |


Batch profile: `build_engines` declares the DiT batch axis with an upper bound of 8 (upstream's default, for vectorised sim clients); a batch-1 deployment should pass `--max-batch 1`, which drops the DiT engine's reserved activation memory from ~3 GB to a few MB without changing its outputs.

## Device memory per arm

`memory.py` measures one arm in one fresh process and reports two numbers to
read together:

* **as served** (`--keep-replaced-weights`): what `serve` holds, with the
  replaced PyTorch weights still resident on the GPU;
* **floor** (default for an engine arm): the engines plus the PyTorch
  components the runtime still executes (what upstream's pipeline swap leaves in PyTorch: embedding table, encoders' glue).

The value is the median `cudaMemGetInfo` plateau (total minus free) over 60
timed calls after 10 warm-ups; `memory --help` gives the details. Run the eager
arm first so the engine arms can report their decoded-action cosine against it:

```bash
python -m foldquant_integration.memory --model-path <ckpt> --embodiment-tag libero_panda --dataset-path <LIBERO calib> --reference-actions ref.npz
python -m foldquant_integration.memory --model-path <ckpt> --embodiment-tag libero_panda --dataset-path <LIBERO calib> --engine-dir exports/<arm>/engines --reference-actions ref.npz
python -m foldquant_integration.memory --model-path <ckpt> --embodiment-tag libero_panda --dataset-path <LIBERO calib> --engine-dir exports/<arm>/engines --keep-replaced-weights
```
