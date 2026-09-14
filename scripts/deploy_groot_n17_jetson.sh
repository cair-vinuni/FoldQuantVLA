#!/usr/bin/env bash
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.
#
# GR00T N1.7 on a Jetson AGX Orin, end to end: plugins -> float pipeline ->
# FoldQuant export -> engines -> verify -> serve. docs/deploy/jetson_serve.md
# walks through the same steps one command at a time.
#
#   CKPT=/path/to/checkpoint DS=/path/to/lerobot_dataset TAG=new_embodiment \
#       scripts/deploy_groot_n17_jetson.sh
#
#   # only rebuild and serve another arm, reusing the float pipeline
#   ARM=w4a4 LLM_SCHEME=w4a4_srg DIT_SCHEME=w4a4_shg STEPS=export,build,verify,serve \
#   CKPT=... DS=... TAG=... scripts/deploy_groot_n17_jetson.sh
#
#   # NVIDIA ModelOpt INT8 SmoothQuant comparison baseline (needs nvidia-modelopt, see the guide)
#   ARM=modelopt_w8a8_sq LLM_SCHEME=modelopt_w8a8_smoothquant DIT_SCHEME=modelopt_w8a8_smoothquant \
#   NUM_CALIB=64 CKPT=... DS=... TAG=... scripts/deploy_groot_n17_jetson.sh
#
#   # serve an arm that is already built
#   STEPS=serve PORT=5556 CKPT=... TAG=... scripts/deploy_groot_n17_jetson.sh
#
# Required
#   CKPT         checkpoint directory (a local path)
#   DS           LeRobot dataset used for calibration and verify
#                (not needed when STEPS is only `serve`)
# Optional
#   TAG          embodiment tag (default: auto-detected by upstream)
#   OUT          export root (default: models/groot_n1_7/exports)
#   ARM          arm name, the subdirectory of OUT (default: w8a8)
#   LLM_SCHEME   default w8a8_sr          DIT_SCHEME   default w8a8_sh
#                (modelopt_w8a8_smoothquant: the ModelOpt Q/DQ baseline instead of a FoldQuant graph)
#   LLM_PARAMS   JSON, e.g. '{"site_bits": {"o": 8, "down": 8}}'   (default: {})
#   NUM_CALIB    calibration samples (default 128; keep >= 128 for a *_g scheme)
#   NUM_VERIFY   held-out verify samples (default 32)
#   VIDEO_BACKEND  torchcodec | decord | torchvision_av (default: upstream's)
#   PYTHON       interpreter (default: models/groot_n1_7/.venv/bin/python, else python)
#   HOST PORT    serve address (default 0.0.0.0:5555)
#   STEPS        comma list of: check,kernels,float,export,build,verify,serve
#                (default: all of them, in that order)
#   FORCE=1      redo export/build even when their outputs already exist
#
# Each step is skipped when its output is already complete, so re-running after
# a failure resumes where it stopped. Logs go to $OUT/<arm>/logs/.

set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAM="$R/models/groot_n1_7"
. "$R/scripts/_family_env.sh"

: "${CKPT:?set CKPT to the GR00T N1.7 checkpoint directory}"
TAG="${TAG:-}"
OUT="${OUT:-$FAM/exports}"
ARM="${ARM:-w8a8}"
LLM_SCHEME="${LLM_SCHEME:-w8a8_sr}"
DIT_SCHEME="${DIT_SCHEME:-w8a8_sh}"
LLM_PARAMS="${LLM_PARAMS:-}"
NUM_CALIB="${NUM_CALIB:-128}"
NUM_VERIFY="${NUM_VERIFY:-32}"
VIDEO_BACKEND="${VIDEO_BACKEND:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5555}"
STEPS="${STEPS:-check,kernels,float,export,build,verify,serve}"
FORCE="${FORCE:-0}"
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$FAM/.venv/bin/python" ]; then PYTHON="$FAM/.venv/bin/python"; else PYTHON=python; fi
fi

OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
FLOAT="$OUT/float"
ARMDIR="$OUT/$ARM"
LOGS="$ARMDIR/logs"
mkdir -p "$LOGS"

# The seven engines upstream's n17_full_pipeline mode loads.
ENGINES=(vit_bf16 llm_bf16 vl_self_attention state_encoder action_encoder dit_bf16 action_decoder)

export PYTHONPATH
PYTHONPATH="$(family_pythonpath "$R" "$FAM")"

say()  { printf '\n=== %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
has_step() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
engines_complete() {
  local d="$1" e
  for e in "${ENGINES[@]}"; do [ -s "$d/$e.engine" ] || return 1; done
}
need_ds() { [ -n "${DS:-}" ] || die "set DS to a LeRobot dataset (needed by the '$1' step)"; }

model_args=(--model-path "$CKPT")
[ -n "$TAG" ] && model_args+=(--embodiment-tag "$TAG")
backend_args=()
[ -n "$VIDEO_BACKEND" ] && backend_args=(--video-backend "$VIDEO_BACKEND")

cd "$FAM"

if has_step check; then
  say "check: platform and environment"
  [ "$(uname -m)" = aarch64 ] || echo "  warning: not aarch64 -- this script targets Jetson Orin"
  [ -d "$CKPT" ] || die "CKPT=$CKPT is not a directory"
  "$PYTHON" - <<'EOF' || die "the environment is incomplete -- see docs/deploy/jetson_serve.md, step 1"
import importlib.util, sys
missing = [m for m in ("torch", "tensorrt", "gr00t", "foldquant", "foldquant_integration")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"  cannot import: {', '.join(missing)}")
import torch, tensorrt
if not torch.cuda.is_available():
    sys.exit("  torch has no CUDA device (a PyPI aarch64 wheel? use the Jetson index)")
print(f"  torch {torch.__version__}, CUDA device {torch.cuda.get_device_name(0)}, TensorRT {tensorrt.__version__}")
EOF
  case "$LLM_SCHEME $DIT_SCHEME" in
    *modelopt_*)
      "$PYTHON" -c "import modelopt.torch.quantization, onnx_graphsurgeon" 2>/dev/null \
        || die "a modelopt_* scheme needs nvidia-modelopt==0.45.0 and onnx-graphsurgeon -- see docs/deploy/jetson_serve.md"
      echo "  nvidia-modelopt present" ;;
  esac
fi

if has_step kernels; then
  say "kernels: FoldQuant TensorRT plugins for this device"
  if "$PYTHON" -m foldquant.kernels status >"$LOGS/kernels.log" 2>&1; then
    echo "  already built: $(head -1 "$LOGS/kernels.log")"
  else
    echo "  building (needs nvcc and CUTLASS; see the guide) -> $LOGS/kernels.log"
    "$PYTHON" -m foldquant.kernels build >>"$LOGS/kernels.log" 2>&1 || die "plugin build failed -- see $LOGS/kernels.log"
    "$PYTHON" -m foldquant.kernels status >>"$LOGS/kernels.log" 2>&1 || die "plugins still unresolved -- see $LOGS/kernels.log"
  fi
fi

if has_step float; then
  say "float: upstream bf16 pipeline (the modules FoldQuant does not replace)"
  if engines_complete "$FLOAT/engines"; then
    echo "  already complete: $FLOAT/engines"
  else
    need_ds float
    echo "  export + build -> $FLOAT (log: $LOGS/float.log)"
    "$PYTHON" scripts/deployment/build_trt_pipeline.py \
        "${model_args[@]}" --dataset-path "$DS" "${backend_args[@]}" \
        --output-dir "$FLOAT" --steps export,build >"$LOGS/float.log" 2>&1 \
      || die "float pipeline failed -- see $LOGS/float.log"
    engines_complete "$FLOAT/engines" || die "float pipeline finished without all seven engines -- see $LOGS/float.log"
  fi
fi

if has_step export; then
  say "export: FoldQuant graphs ($LLM_SCHEME / $DIT_SCHEME)"
  if [ "$FORCE" != 1 ] && [ -s "$ARMDIR/onnx/foldquant_export.json" ]; then
    echo "  already exported: $ARMDIR/onnx (FORCE=1 to redo)"
  else
    need_ds export
    rm -rf "$ARMDIR/onnx"
    args=("${model_args[@]}" --dataset-path "$DS" "${backend_args[@]}"
          --num-calib "$NUM_CALIB" --seed 0
          --llm-scheme "$LLM_SCHEME" --dit-scheme "$DIT_SCHEME" --output-dir "$ARMDIR")
    [ -n "$LLM_PARAMS" ] && args+=(--llm-params "$LLM_PARAMS")
    echo "  calibrating on $NUM_CALIB samples (log: $LOGS/export.log)"
    "$PYTHON" -m foldquant_integration.export_foldquant "${args[@]}" >"$LOGS/export.log" 2>&1 \
      || die "export failed -- see $LOGS/export.log"
  fi
fi

if has_step build; then
  say "build: TensorRT engines for $ARM"
  if [ "$FORCE" != 1 ] && engines_complete "$ARMDIR/engines"; then
    echo "  already complete: $ARMDIR/engines (FORCE=1 to redo)"
  else
    [ -s "$ARMDIR/onnx/foldquant_export.json" ] || die "no export at $ARMDIR/onnx -- run the export step first"
    engines_complete "$FLOAT/engines" || die "float engines incomplete at $FLOAT/engines -- run the float step first"
    rm -rf "$ARMDIR/engines"
    echo "  building (log: $LOGS/build.log)"
    "$PYTHON" -m foldquant_integration.build_engines \
        --onnx-dir "$ARMDIR/onnx" --engine-dir "$ARMDIR/engines" \
        --float-onnx-dir "$FLOAT/onnx" --float-engine-dir "$FLOAT/engines" >"$LOGS/build.log" 2>&1 \
      || die "engine build failed -- see $LOGS/build.log"
    engines_complete "$ARMDIR/engines" || die "build finished without all seven engines -- see $LOGS/build.log"
  fi
fi

if has_step verify; then
  say "verify: engines against the bf16 policy on $NUM_VERIFY held-out samples"
  need_ds verify
  "$PYTHON" -m foldquant_integration.verify \
      "${model_args[@]}" --dataset-path "$DS" "${backend_args[@]}" \
      --engine-dir "$ARMDIR/engines" --num-samples "$NUM_VERIFY" --seed 42 \
      --output "$ARMDIR/verify.json" >"$LOGS/verify.log" 2>&1 \
    || die "verify failed -- see $LOGS/verify.log"
  "$PYTHON" - "$ARMDIR/verify.json" <<'EOF' || true
import json, sys
d = json.load(open(sys.argv[1]))
nan = float("nan")
a, f = d.get("actions", {}), d.get("backbone_features", {})
print(f"  action cosine    mean {a.get('cos_mean', nan):.5f}  min {a.get('cos_min', nan):.5f}")
print(f"  backbone cosine  mean {f.get('cos_mean', nan):.5f}  min {f.get('cos_min', nan):.5f}")
EOF
  echo "  report: $ARMDIR/verify.json"
fi

if has_step serve; then
  say "serve: $ARMDIR/engines on $HOST:$PORT"
  engines_complete "$ARMDIR/engines" || die "engines incomplete at $ARMDIR/engines"
  # Refuse an occupied port rather than failing after the ~30 s checkpoint load,
  # and never disturb whatever is already listening there.
  if command -v ss >/dev/null && ss -ltn | awk '{print $4}' | grep -qE "[:.]$PORT\$"; then
    die "port $PORT is already in use -- pick another with PORT=..."
  fi
  echo "  Ctrl+C to stop. The port opens once the checkpoint and engines are loaded."
  exec "$PYTHON" -m foldquant_integration.serve \
      "${model_args[@]}" --engine-dir "$ARMDIR/engines" \
      --mode n17_full_pipeline --host "$HOST" --port "$PORT"
fi
