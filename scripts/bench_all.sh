#!/usr/bin/env bash
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.
#
# Latency for every family, on this device, into results/<family>/.
#
#   scripts/bench_all.sh                 # every family that has engines built
#   scripts/bench_all.sh smolvla evo_1   # only these
#
# Each family runs in its OWN virtualenv and from its OWN directory, because
# the integrations are pinned to their upstream's environment (Python 3.10 to
# 3.12, four torch versions between them). Nothing here is quantized or
# rebuilt: the engines must already exist under
# models/<family>/exports/<arm>/engines.
#
# Arms are discovered, not hardcoded — whatever <family>/exports/*/engines
# holds is timed, so this script does not go stale when an arm is added.
#
# One invocation per family times the eager PyTorch arm ONCE and every engine
# arm against it, which is why the output is results/<family>/benchmark.json
# rather than one file per arm: a shared baseline makes the speedup column
# comparable across arms, and re-timing eager per arm would both waste Orin
# minutes and let the baseline drift between rows. GR00T N1.7 is the exception
# — it wraps upstream's benchmark_inference.py, which takes one engine
# directory and prints to stdout, so it gets one benchmark.log per arm.
#
# Paths differ per machine, so every checkpoint/dataset root is an environment
# variable; the script refuses to guess. Optional knobs have defaults below.
: "${N17_MODEL:?set N17_MODEL to the GR00T N1.7 LIBERO checkpoint directory}"
: "${N16_MODEL:?set N16_MODEL to the GR00T N1.6 LIBERO checkpoint directory}"
# N1.6's processor_config.json names several embodiments; the LIBERO one is
# what the verify rows in results/groot_n1_6 were measured with.
: "${N16_EMBODIMENT:=libero_panda}"
: "${N15_MODEL:?set N15_MODEL to the GR00T N1.5 LIBERO checkpoint directory}"
: "${PI05_CKPT:?set PI05_CKPT to the converted pi05_libero PyTorch checkpoint directory}"
: "${GROOT_DATA:?set GROOT_DATA to the LIBERO 4-suite calibration dataset (LeRobot layout)}"
: "${SMOLVLA_DATA:=HuggingFaceVLA/libero}"
: "${SMOLVLA_EPISODES:=0-149}"
: "${EVO1_CKPT:?set EVO1_CKPT to the Evo1_LIBERO snapshot directory}"
: "${EVO1_DATA:?set EVO1_DATA to a local LeRobot LIBERO snapshot directory}"
: "${ITERS:=20}"
: "${WARMUP:=5}"

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAMILIES=("$@")
[ ${#FAMILIES[@]} -eq 0 ] && FAMILIES=(groot_n1_7 groot_n1_6 groot_n1_5 pi05 smolvla evo_1)

# LABEL=DIR for every built arm of a family; empty when none exist. Emitted as
# ONE `--arms a=... b=...` group: tyro's list flag keeps only the last `--arms`
# when the flag is repeated, so `--arms a=x --arms b=y` would time b alone.
arms_of() {
  local fam="$1" spec=() d
  for d in "$REPO/models/$fam/exports"/*/engines; do
    [ -d "$d" ] || continue
    spec+=("$(basename "$(dirname "$d")")=$d")
  done
  [ ${#spec[@]} -gt 0 ] && printf '%s\n' "--arms" "${spec[@]}"
  return 0
}

. "$(dirname "${BASH_SOURCE[0]}")/_gpu_busy.sh"
. "$(dirname "${BASH_SOURCE[0]}")/_family_env.sh"
busy() {
  local n
  n=$(gpu_busy_pids | grep -c . || true)
  [ "$n" -gt 0 ] && { echo "  SKIP: $n process(es) already on the GPU"; return 0; }
  return 1
}

base_pythonpath="${PYTHONPATH:-}"
for fam in "${FAMILIES[@]}"; do
  venv="$REPO/models/$fam/.venv/bin/python"
  out="$REPO/results/$fam"
  echo "=== $fam"
  [ -x "$venv" ] || { echo "  SKIP: no venv at models/$fam/.venv"; continue; }
  busy && continue
  mkdir -p "$out"
  cd "$REPO/models/$fam"
  # from the caller's PYTHONPATH each time, so families do not pile onto each other
  export PYTHONPATH
  PYTHONPATH=$(PYTHONPATH="$base_pythonpath" family_pythonpath "$REPO" "$REPO/models/$fam")

  if [ "$fam" = groot_n1_7 ]; then
    # upstream's own script: one engine directory per call, stdout is the record
    for d in exports/*/engines; do
      [ -d "$d" ] || continue
      arm="$(basename "$(dirname "$d")")"
      mkdir -p "$out/$arm"
      echo "  arm $arm -> results/$fam/$arm/benchmark.log"
      "$venv" -m foldquant_integration.benchmark \
        --model-path "$N17_MODEL" \
        --trt-engine-path "$d" --trt-mode n17_full_pipeline \
        2>&1 | tee "$out/$arm/benchmark.log" || echo "  FAIL: $fam/$arm (see results/$fam/$arm/benchmark.log)"
    done
    cd "$REPO"; continue
  fi

  mapfile -t ARMS < <(arms_of "$fam")
  [ ${#ARMS[@]} -eq 0 ] && { echo "  SKIP: no engines under models/$fam/exports/*/engines"; cd "$REPO"; continue; }
  echo "  arms: ${ARMS[*]}"

  case "$fam" in
    groot_n1_6) set -- --model-path "$N16_MODEL" --dataset-path "$GROOT_DATA" --embodiment-tag "$N16_EMBODIMENT" ;;
    groot_n1_5) set -- --model-path "$N15_MODEL" --dataset-path "$GROOT_DATA" ;;
    pi05)       set -- --checkpoint-dir "$PI05_CKPT" --dataset-path "$GROOT_DATA" ;;
    smolvla)    set -- --dataset-path "$SMOLVLA_DATA" --episodes "$SMOLVLA_EPISODES" ;;
    evo_1)      set -- --checkpoint-dir "$EVO1_CKPT" --dataset-path "$EVO1_DATA" ;;
  esac

  "$venv" -m foldquant_integration.benchmark "$@" "${ARMS[@]}" \
      --num-iterations "$ITERS" --warmup "$WARMUP" \
      --output "$out/benchmark.json" 2>&1 | tee "$out/benchmark.log" \
      || echo "  FAIL: $fam (see results/$fam/benchmark.log)"
  cd "$REPO"
done

echo
echo "done — results under $REPO/results/"
