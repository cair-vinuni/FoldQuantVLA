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
   python scripts/deployment/export_onnx_n1d6.py --model_path ... --dataset_path ... \
       --embodiment_tag libero_panda --output_dir exports/n16_float/onnx
   python -m foldquant_integration.build_engines --onnx-dir exports/n16_w8a8_none/onnx \
       --float-onnx-dir exports/n16_float/onnx --engine-dir exports/n16_w8a8_none/engines
   ```

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
