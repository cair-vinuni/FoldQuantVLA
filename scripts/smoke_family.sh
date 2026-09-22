#!/usr/bin/env bash
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.
#
# One family, end to end, small: export -> build -> verify, on 8 calibration
# observations and 8 held-out ones. It answers one question (does this
# family's chain run on this machine and produce a record?) and deliberately
# not "are the paper's numbers right", which needs the full 128/32 protocol.
#
#   scripts/smoke_family.sh groot_n1_7
#   scripts/smoke_family.sh            # every family that has a .venv
#
# Paths differ per machine. Set the ones for the families you want to run;
# a family whose checkpoint or dataset is unset is skipped with a message
# rather than failed, so a reviewer holding two of four checkpoints still
# gets a useful report.
#
#   N17_MODEL N16_MODEL N15_MODEL   GR00T checkpoints
#   PI05_CKPT                       openpi checkpoint
#   GROOT_DATA                      LeRobot dataset for the GR00T + pi05 families
#   N17_VIDEO_BACKEND N16_VIDEO_BACKEND N15_VIDEO_BACKEND
#                                   optional; e.g. "decord" where torchcodec does not load
#
# Each family runs in its OWN virtualenv, from its OWN directory: the
# integrations are pinned to their upstream's environment and share nothing
# but the foldquant package.

set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${SMOKE_OUT:-$R/.smoke}"
CALIB="${SMOKE_CALIB:-8}"
SAMPLES="${SMOKE_SAMPLES:-8}"
mkdir -p "$OUT"

FAMILIES=("$@")
[ ${#FAMILIES[@]} -eq 0 ] && FAMILIES=(groot_n1_7 groot_n1_6 groot_n1_5 pi05)

# A policy server or a training job on the same card will fail this in a way that
# looks like a defect: TensorRT reports "execute_async_v3() failed" with no mention
# of memory, and known-good engines fail it exactly as a fresh build does. Say so
# up front rather than let the report blame the code.
busy_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
# nvidia-smi answers "[N/A]" for memory on Tegra too; see scripts/_gpu_busy.sh
# for why the PID check needs the same care.
. "$R/scripts/_gpu_busy.sh"
. "$R/scripts/_family_env.sh"
busy_pids=$(gpu_busy_pids)
case "$busy_mib" in ''|*N/A*|*Supported*) busy_mib="an unknown amount of" ;; esac
busy_n=$(printf '%s\n' "$busy_pids" | grep -c . || true)
if [ "${busy_n:-0}" -gt 0 ]; then
  echo "warning: ${busy_n} process(es) already on the GPU using ${busy_mib} MiB."
  echo "         The larger families need the card to themselves; a failure below may be"
  echo "         contention, not a defect. Set SMOKE_ALLOW_BUSY_GPU=1 to run anyway."
  [ "${SMOKE_ALLOW_BUSY_GPU:-0}" = 1 ] || exit 2
fi

pass=0; fail=0; skip=0
note () { printf '  %-9s %s\n' "$1" "$2"; }

run_family () {
  local fam="$1" venv="$R/models/$fam/.venv/bin/python"
  local log="$OUT/$fam.log" dir="$R/models/$fam"
  # The repo root carries the `foldquant` package and the family directory carries
  # `gr00t` and `foldquant_integration`. `-m` puts the family directory on sys.path
  # but not the root, and a script run by path gets neither -- so both imports fail
  # for anyone whose venv was built without `uv pip install -e .`, which the
  # per-family install_deps.sh does but a hand-built environment need not.
  local pp
  pp=$(family_pythonpath "$R" "$dir")
  echo "═══ $fam"
  [ -x "$venv" ] || { note SKIP "no .venv; see models/$fam/foldquant_integration/README.md"; skip=$((skip+1)); return; }

  # per-family arguments; an unset path means skip, never a wrong-path failure
  local ckpt=() data=() extra=()
  case "$fam" in
    groot_n1_7) ckpt=(--model-path "${N17_MODEL:-}") ; data=(--dataset-path "${GROOT_DATA:-}")  ; extra=(--embodiment-tag "${N17_TAG:-libero_panda}") ;;
    groot_n1_6) ckpt=(--model-path "${N16_MODEL:-}") ; data=(--dataset-path "${GROOT_DATA:-}")  ; extra=(--embodiment-tag "${N16_TAG:-libero_panda}") ;;
    groot_n1_5) ckpt=(--model-path "${N15_MODEL:-}") ; data=(--dataset-path "${GROOT_DATA:-}")  ; extra=(--embodiment-tag "${N15_TAG:-new_embodiment}") ;;
    pi05)       ckpt=(--checkpoint-dir "${PI05_CKPT:-}") ; data=(--dataset-path "${GROOT_DATA:-}") ;;
  esac
  # The GR00T integrations default to video_backend="torchcodec". Where torchcodec
  # does not load -- N1.5 on a Jetson, whose Orin wheels are built against a
  # different torch and FFmpeg -- export fails with "torchcodec is not available"
  # before a frame is read. Let the caller pick per family, in the same N1x_*
  # convention as the embodiment tags; unset keeps each family's own default.
  local backend=""
  case "$fam" in
    groot_n1_7) backend="${N17_VIDEO_BACKEND:-}" ;;
    groot_n1_6) backend="${N16_VIDEO_BACKEND:-}" ;;
    groot_n1_5) backend="${N15_VIDEO_BACKEND:-}" ;;
  esac
  [ -n "$backend" ] && extra+=(--video-backend "$backend")

  for a in "${ckpt[@]}" "${data[@]}"; do
    [ -z "$a" ] && { note SKIP "a required path is unset (see the header of this script)"; skip=$((skip+1)); return; }
  done

  # GR00T N1.7 and N1.6 quantize two modules of eight; the rest of the pipeline has
  # to come from a float export, so build_engines refuses without one. The other four
  # families leave their untouched modules in PyTorch and need no such directory.
  local float_args=()
  case "$fam" in
    groot_n1_7|groot_n1_6)
      # Only N1.7 accepts --float-engine-dir; N1.6 takes the ONNX directory alone and
      # builds the untouched components from it. Passing both to N1.6 is rejected.
      local reuse=0
      [ "$fam" = groot_n1_7 ] && reuse=1
      if [ -d "$dir/exports/float/onnx" ]; then
        float_args=(--float-onnx-dir exports/float/onnx)
        [ "$reuse" = 1 ] && [ -d "$dir/exports/float/engines" ] && float_args+=(--float-engine-dir exports/float/engines)
        note ok "float pipeline found, reusing exports/float"
      else
        note ..   "building the float pipeline first (needed for the untouched modules)"
        if [ "$fam" = groot_n1_7 ]; then
          ( cd "$dir" && PYTHONPATH="$pp" "$venv" scripts/deployment/build_trt_pipeline.py \
              "${ckpt[@]}" "${data[@]}" "${extra[@]}" --output-dir .smoke_float --steps export,build ) >>"$log" 2>&1 \
            || { note FAIL "float pipeline: see $log"; fail=$((fail+1)); return; }
        else
          # N1.6 ships no build_trt_pipeline.py: its float arm is the DiT alone, from
          # export_onnx_n1d6.py, which takes argparse underscore flags and writes the
          # ONNX directory directly (build_engines builds the engine from it).
          # GR00T_ONNX_EXPORTER_MODE=legacy is required, not optional -- the default
          # dynamo exporter specialises vl_seq_len and hands back a reference that
          # runs and is wrong; see foldquant_integration/README.md.
          ( cd "$dir" && PYTHONPATH="$pp" GR00T_ONNX_EXPORTER_MODE=legacy \
              "$venv" scripts/deployment/export_onnx_n1d6.py \
              --model_path "${N16_MODEL:-}" --dataset_path "${GROOT_DATA:-}" \
              --embodiment_tag "${N16_TAG:-libero_panda}" --output_dir .smoke_float/onnx ) >>"$log" 2>&1 \
            || { note FAIL "float DiT export: see $log"; fail=$((fail+1)); return; }
        fi
        float_args=(--float-onnx-dir .smoke_float/onnx)
        [ "$reuse" = 1 ] && float_args+=(--float-engine-dir .smoke_float/engines)
      fi
      ;;
  esac

  local exp="$dir/.smoke_export"
  rm -rf "$exp"
  ( cd "$dir" && PYTHONPATH="$pp" "$venv" -m foldquant_integration.export_foldquant \
      "${ckpt[@]}" "${data[@]}" "${extra[@]}" --num-calib "$CALIB" --seed 0 \
      --output-dir .smoke_export ) >>"$log" 2>&1 \
    || { note FAIL "export: see $log"; fail=$((fail+1)); return; }
  note ok "export"

  ( cd "$dir" && PYTHONPATH="$pp" "$venv" -m foldquant_integration.build_engines \
      --onnx-dir .smoke_export/onnx --engine-dir .smoke_export/engines \
      "${float_args[@]}" ) >>"$log" 2>&1 \
    || { note FAIL "build_engines: see $log"; fail=$((fail+1)); return; }
  note ok "build_engines"

  ( cd "$dir" && PYTHONPATH="$pp" "$venv" -m foldquant_integration.verify \
      "${ckpt[@]}" "${data[@]}" "${extra[@]}" --engine-dir .smoke_export/engines \
      --num-samples "$SAMPLES" --seed 42 --output "$OUT/$fam.verify.json" ) >>"$log" 2>&1 \
    || { note FAIL "verify: see $log"; fail=$((fail+1)); return; }

  "$venv" - "$OUT/$fam.verify.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
a = d["actions"]
print(f"  ok        verify    action cos median {a['cos_median']:.5f} "
      f"worst |d| {a['max_abs']:.4f}  ({d['num_samples']} held-out)")
PY
  pass=$((pass+1))
}

for fam in "${FAMILIES[@]}"; do run_family "$fam"; done
echo
echo "smoke: $pass passed, $fail failed, $skip skipped  (records under $OUT)"
[ "$fail" -eq 0 ]
