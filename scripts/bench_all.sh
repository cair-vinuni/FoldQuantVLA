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
# Paths differ per machine; override any of these:
: "${N17_MODEL:=$HOME/vr_repos/VLA-OPT/weights/nvidia/GR00T-N1.7-LIBERO-4suite}"
: "${N16_MODEL:=$HOME/vr_repos/VLA-OPT/weights/nvidia/GR00T-N1.6-LIBERO}"
: "${N15_MODEL:=$HOME/vr_repos/VLA-OPT/weights/nvidia/GR00T-N1.5-LIBERO-4suite}"
: "${PI05_CKPT:=$HOME/vr_repos/VLA-OPT/weights/openpi/pi05_libero_pytorch}"
: "${GROOT_DATA:=$HOME/vr_repos/VLA-OPT/tmp/data/libero_4suites_calib}"
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

# --arms LABEL=DIR for every built arm of a family; empty when none exist.
arms_of() {
  local fam="$1" spec=() d
  for d in "$REPO/models/$fam/exports"/*/engines; do
    [ -d "$d" ] || continue
    spec+=("--arms" "$(basename "$(dirname "$d")")=$d")
  done
  printf '%s\n' "${spec[@]:-}"
}

busy() {
  local n
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  [ "$n" -gt 0 ] && { echo "  SKIP: $n process(es) already on the GPU"; return 0; }
  return 1
}

for fam in "${FAMILIES[@]}"; do
  venv="$REPO/models/$fam/.venv/bin/python"
  out="$REPO/results/$fam"
  echo "=== $fam"
  [ -x "$venv" ] || { echo "  SKIP: no venv at models/$fam/.venv"; continue; }
  busy && continue
  mkdir -p "$out"
  cd "$REPO/models/$fam"

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
        2>&1 | tee "$out/$arm/benchmark.log"
    done
    cd "$REPO"; continue
  fi

  mapfile -t ARMS < <(arms_of "$fam")
  [ ${#ARMS[@]} -eq 0 ] && { echo "  SKIP: no engines under models/$fam/exports/*/engines"; cd "$REPO"; continue; }
  echo "  arms: ${ARMS[*]}"

  case "$fam" in
    groot_n1_6) set -- --model-path "$N16_MODEL" --dataset-path "$GROOT_DATA" ;;
    groot_n1_5) set -- --model-path "$N15_MODEL" --dataset-path "$GROOT_DATA" ;;
    pi05)       set -- --checkpoint-dir "$PI05_CKPT" --dataset-path "$GROOT_DATA" ;;
    smolvla)    set -- --dataset-path "$SMOLVLA_DATA" --episodes "$SMOLVLA_EPISODES" ;;
    evo_1)      set -- --checkpoint-dir "$EVO1_CKPT" --dataset-path "$EVO1_DATA" ;;
  esac

  "$venv" -m foldquant_integration.benchmark "$@" "${ARMS[@]}" \
      --num-iterations "$ITERS" --warmup "$WARMUP" \
      --output "$out/benchmark.json" 2>&1 | tee "$out/benchmark.log"
  cd "$REPO"
done

echo
echo "done — results under $REPO/results/"
