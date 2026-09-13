# GR00T N1.7 on Jetson AGX Orin: from checkpoint to a running server

This guide builds FoldQuant TensorRT engines for a GR00T N1.7 checkpoint
**on the Orin itself** and serves them to a robot client. Everything runs on
the board; nothing needs root once the base JetPack install is in place.

- One command: [`scripts/deploy_groot_n17_jetson.sh`](../../scripts/deploy_groot_n17_jetson.sh)
  runs every step below, skips the ones whose outputs already exist, and ends
  by starting the server.
- Step by step: sections 1 to 7.

For building on an x86 workstation and copying graphs to the board, and for
the status of the other model families on Orin, see [`jetson.md`](jetson.md).

Tested on Jetson AGX Orin 64 GB, JetPack 6.2 (L4T R36.4.3), CUDA 12.6,
TensorRT 10.3.0, Python 3.10, with a GR00T N1.7 SO101 checkpoint.

## 0. Quick start

```bash
cd <repo>
export CKPT=/path/to/GR00T-N1.7-checkpoint        # local directory
export DS=/path/to/lerobot_dataset                # calibration + verify data
export TAG=new_embodiment                         # embodiment tag of the checkpoint

scripts/deploy_groot_n17_jetson.sh                # w8a8 arm, serves on :5555
```

Useful variations:

```bash
# another arm, reusing the float pipeline already built
ARM=w4a4 LLM_SCHEME=w4a4_srg DIT_SCHEME=w4a4_shg scripts/deploy_groot_n17_jetson.sh

# mixed precision: keep two LLM projection sites at 8 bit
ARM=w4a4_res8 LLM_SCHEME=w4a4_srg DIT_SCHEME=w4a4_shg \
LLM_PARAMS='{"site_bits": {"o": 8, "down": 8}}' scripts/deploy_groot_n17_jetson.sh

# build and verify only, no server
STEPS=check,kernels,float,export,build,verify scripts/deploy_groot_n17_jetson.sh

# serve an arm that is already built, on another port
STEPS=serve PORT=5556 scripts/deploy_groot_n17_jetson.sh
```

Outputs land in `models/groot_n1_7/exports/` (override with `OUT=`):

```
exports/
  float/{onnx,engines}          upstream bf16 pipeline, shared by every arm
  <arm>/onnx                    FoldQuant graphs + foldquant_export.json
  <arm>/engines                 the seven engines the server loads
  <arm>/verify.json             engine-vs-PyTorch report
  <arm>/logs/                   one log per step
```

The header of the script lists every variable it reads.

## 1. Environment (once per board)

Use the Orin recipe, not the family's top-level `pyproject.toml`: that one
resolves torch from PyPI, whose aarch64 wheels are not built for the Orin's
GPU (sm_87).

```bash
cd <repo>/models/groot_n1_7
export UV_PROJECT_ENVIRONMENT=$PWD/.venv          # or any path outside the repo

uv sync --project scripts/deployment/orin --no-install-project
PY=$UV_PROJECT_ENVIRONMENT/bin/python

# torch 2.10 needs libcudss at runtime; --no-deps keeps it from pulling
# CUDA wheels that clash with JetPack's CUDA 12.6
uv pip install --python $PY --no-deps nvidia-cudss-cu12

# torchcodec built against the Orin's FFmpeg 4 (shipped in the repo)
uv pip install --python $PY --force-reinstall --no-deps \
    scripts/deployment/orin/wheels/torchcodec-0.10.0a0-cp310-cp310-linux_aarch64.whl

# JetPack's TensorRT is a system package; expose it to the venv
echo /usr/lib/python3.10/dist-packages \
    > $UV_PROJECT_ENVIRONMENT/lib/python3.10/site-packages/jetpack-system-packages.pth
```

`scripts/deployment/orin/install_deps.sh` does the same plus an `apt-get
install ffmpeg`, which needs sudo. The steps above avoid it when FFmpeg is
already present (`ffmpeg -version`).

Then, in every new shell:

```bash
cd <repo>/models/groot_n1_7
source scripts/activate_orin.sh                   # CUDA 12.6 paths for torch/triton
export PATH=$UV_PROJECT_ENVIRONMENT/bin:$PATH
export PYTHONPATH=<repo>:<repo>/models/groot_n1_7 # foldquant + gr00t, no editable install
```

`scripts/deploy_groot_n17_jetson.sh` sets `PYTHONPATH` itself and uses
`models/groot_n1_7/.venv/bin/python` unless `PYTHON=` points elsewhere.

## 2. TensorRT plugins (once per board)

FoldQuant's quantized layers are TensorRT plugins compiled for one
`(GPU architecture, TensorRT version)` pair.

```bash
git -C <repo> submodule update --init third_party/cutlass   # or: export CUTLASS_INCLUDE_DIR=/path/to/cutlass/include
export TENSORRT_ROOT=/usr                                   # JetPack's own TensorRT headers
export CUDA_HOME=/usr/local/cuda-12.6

python -m foldquant.kernels build
python -m foldquant.kernels status
```

`status` must print `target: sm87-aarch64-trt10.3` and a path for each of the
three libraries. They are cached under `FOLDQUANT_CACHE_DIR` (default
`~/.cache/foldquant`); set the same value when building engines and serving.

## 3. Float pipeline (once per checkpoint)

GR00T N1.7 runs as seven engines. FoldQuant replaces two of them (LLM and
DiT); the other five come from upstream's bf16 pipeline.

```bash
python scripts/deployment/build_trt_pipeline.py \
    --model-path "$CKPT" --dataset-path "$DS" --embodiment-tag "$TAG" \
    --output-dir exports/float --steps export,build
ls exports/float/engines      # expect 7 *.engine files
```

This takes the longest of all steps on an Orin. `exports/float/engines` is
also a servable bf16 TensorRT arm on its own.

## 4. Export a FoldQuant arm

```bash
python -m foldquant_integration.export_foldquant \
    --model-path "$CKPT" --dataset-path "$DS" --embodiment-tag "$TAG" \
    --num-calib 128 --seed 0 \
    --llm-scheme w8a8_sr --dit-scheme w8a8_sh \
    --output-dir exports/w8a8
```

| arm | `--llm-scheme` | `--dit-scheme` | notes |
|---|---|---|---|
| `w8a8` | `w8a8_sr` | `w8a8_sh` | safest starting point |
| `w4a4` | `w4a4_srg` | `w4a4_shg` | GPTQ; keep `--num-calib` at 128 or more |
| `w4a4_res8` | `w4a4_srg` + `--llm-params '{"site_bits": {"o": 8, "down": 8}}'` | `w4a4_shg` | 4 bit with two 8 bit sites |

GPTQ's Cholesky falls back to the CPU on JetPack 6.2 (its cuSOLVER is older
than torch expects). The log says so; it only makes calibration slower.

## 5. Build the engines

```bash
python -m foldquant_integration.build_engines \
    --onnx-dir exports/w8a8/onnx --engine-dir exports/w8a8/engines \
    --float-onnx-dir exports/float/onnx --float-engine-dir exports/float/engines
```

`--float-engine-dir` copies the five untouched engines instead of rebuilding
them. The command fails if any of the seven is missing at the end.

## 6. Verify before serving

```bash
python -m foldquant_integration.verify \
    --model-path "$CKPT" --dataset-path "$DS" --embodiment-tag "$TAG" \
    --engine-dir exports/w8a8/engines --num-samples 32 --seed 42
```

It compares engine actions with the bf16 PyTorch policy on held-out samples
and writes `exports/w8a8/engines/verify.json`. Do not skip it on a new board
or a new checkpoint: an engine built from an incomplete graph, or served with
the wrong checkpoint, loads without any error and returns wrong actions.
Compare `actions.cos_mean` across arms; a value far below the float arm's
means the build is broken, not that the scheme is lossy.

## 7. Serve

TensorRT engines:

```bash
python -m foldquant_integration.serve \
    --model-path "$CKPT" --embodiment-tag "$TAG" \
    --engine-dir exports/w8a8/engines \
    --mode n17_full_pipeline --port 5556
```

bf16 PyTorch reference (drop `--engine-dir` and `--mode`):

```bash
python -m foldquant_integration.serve \
    --model-path "$CKPT" --embodiment-tag "$TAG" --port 5555
```

- `--model-path` is required with engines too: the server builds the upstream
  policy (processor, normalization, action decoding) and swaps the engines in.
- The server checks that all seven engine files exist before loading.
- Loading takes about 30 s; the port opens only after that. Ready when the log
  shows `listening on 0.0.0.0:<port>`, or `ss -ltn | grep <port>` lists it.
- Several servers can run side by side on different ports, but they share one
  GPU, so latency measured that way is not representative.
- `FOLDQUANT_TRT_CUDA_GRAPH=1` has no effect on this family; see `jetson.md`.

### Client

The wire protocol is upstream GR00T's (ZeroMQ). Keys and shapes come from the
checkpoint; ask the server rather than guessing:

```python
import numpy as np
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="<orin-ip>", port=5556)
print(client.get_modality_config())

# Keys below are those of an SO101 checkpoint (top/wrist cameras, single_arm/gripper).
obs = {
    "video": {
        "top":   np.zeros((1, 1, 480, 640, 3), np.uint8),
        "wrist": np.zeros((1, 1, 480, 640, 3), np.uint8),
    },
    "state": {
        "single_arm": np.zeros((1, 1, 5), np.float32),
        "gripper":    np.zeros((1, 1, 1), np.float32),
    },
    "language": {"annotation.human.task_description": [["pick up the cube"]]},
}
action, info = client.get_action(obs)      # returns a tuple
for k, v in action.items():
    print(k, v.shape)                      # (batch, horizon, dim)
```

| client error | cause |
|---|---|
| `Observation must contain a 'video' key` | flat keys such as `video.top`; use the nested dict above |
| `... must be ... np.float32. Got float64` | `np.zeros` defaults to float64 |
| `'tuple' object has no attribute 'items'` | `get_action` returns `(action, info)` |

Actions come back in the robot's absolute joint space: relative action
channels are converted back using the state you sent.

### Stopping a server

Stop the one you started, by its port, so other servers on the board keep
running:

```bash
ss -ltnp | grep ':5556'                    # shows pid=<PID>
kill <PID>
```

Avoid `pkill -f foldquant_integration.serve`: it stops every server on the
machine.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `torch.cuda.is_available()` is False | torch came from PyPI; rebuild the venv from `scripts/deployment/orin` |
| `No module named 'foldquant'` / `'gr00t'` | `PYTHONPATH` not set (section 1) |
| plugin library not found | `python -m foldquant.kernels build` not run on this board, or a different `FOLDQUANT_CACHE_DIR` |
| `ImportError: torchcodec is not available` | torchcodec wheel not installed; or pass `--video-backend decord` (`VIDEO_BACKEND=decord` for the script) |
| float pipeline ends with fewer than 7 engines | read `exports/float/pipeline.log`; the build now fails loudly instead of skipping a component |
| `Failed to deserialize the cuda engine` | engines built on another device or TensorRT version; rebuild on this board |
| `port ... already in use` | another server holds that port; pick another `--port` |
| engines load, actions are wrong | engine directory does not match the checkpoint, or a graph's `.onnx.data` is missing; run section 6 |
