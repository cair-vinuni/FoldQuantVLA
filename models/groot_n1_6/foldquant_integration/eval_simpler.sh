#!/usr/bin/env bash
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.
#
# SimplerEnv sweep for one arm of a GR00T N1.6 checkpoint: starts
# `foldquant_integration.serve --use-sim-policy-wrapper` (bf16 when ENGINE_DIR is
# empty, a FoldQuant arm otherwise), runs upstream's client over the task list,
# then stops the server. One log per task under OUT_DIR, plus summary.tsv.
#
#   ARM=bf16 bash foldquant_integration/eval_simpler.sh
#   ARM=w4a4 ENGINE_DIR=exports/bridge_w4a4/engines bash foldquant_integration/eval_simpler.sh
#
# Environment (defaults in brackets):
#   MODEL_PATH   [checkpoints/GR00T-N1.6-bridge]   EMBODIMENT [OXE_WIDOWX]
#   ENGINE_DIR   []  omit for the bf16 reference    PORT       [5555]
#   N_EPISODES   [200]  N_ENVS [5]  N_ACTION_STEPS [4]  MAX_STEPS [300]
#   TASKS        [the seven Bridge tasks of examples/SimplerEnv/README.md]
#   OUT_DIR      [exports/simpler/$ARM]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARM="${ARM:?set ARM (a label, e.g. bf16 / w8a8 / w4a4 / head4_lm8)}"
MODEL_PATH="${MODEL_PATH:-checkpoints/GR00T-N1.6-bridge}"
EMBODIMENT="${EMBODIMENT:-OXE_WIDOWX}"
ENGINE_DIR="${ENGINE_DIR:-}"
PORT="${PORT:-5555}"
N_EPISODES="${N_EPISODES:-200}"
N_ENVS="${N_ENVS:-5}"
N_ACTION_STEPS="${N_ACTION_STEPS:-4}"
MAX_STEPS="${MAX_STEPS:-300}"
OUT_DIR="${OUT_DIR:-exports/simpler/$ARM}"
TASKS="${TASKS:-widowx_spoon_on_towel widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_stack_cube widowx_put_eggplant_in_sink widowx_close_drawer widowx_open_drawer}"

SIM_PY="$ROOT/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
[ -x "$SIM_PY" ] || { echo "SimplerEnv venv missing: run gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh"; exit 1; }
[ -x "$ROOT/.venv/bin/python" ] || { echo "gr00t venv missing: uv sync in $ROOT"; exit 1; }
mkdir -p "$OUT_DIR"

SERVE_ARGS=(--model-path "$MODEL_PATH" --embodiment-tag "$EMBODIMENT" --port "$PORT" --use-sim-policy-wrapper)
[ -n "$ENGINE_DIR" ] && SERVE_ARGS+=(--engine-dir "$ENGINE_DIR")

echo "[$(date +%T)] server: ${SERVE_ARGS[*]}" | tee "$OUT_DIR/run.log"
"$ROOT/.venv/bin/python" -m foldquant_integration.serve "${SERVE_ARGS[@]}" > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  grep -q "listening on" "$OUT_DIR/server.log" 2>/dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited early; see $OUT_DIR/server.log"; exit 1; }
  sleep 2
done
grep -q "listening on" "$OUT_DIR/server.log" || { echo "server did not come up in 6 min; see $OUT_DIR/server.log"; exit 1; }
sleep 5

printf "task\tn_episodes\tsuccess_rate\tseconds\n" > "$OUT_DIR/summary.tsv"
for TASK in $TASKS; do
  LOG="$OUT_DIR/$TASK.log"
  echo "[$(date +%T)] $TASK" | tee -a "$OUT_DIR/run.log"
  T0=$(date +%s)
  "$SIM_PY" gr00t/eval/rollout_policy.py \
    --policy_client_host 127.0.0.1 --policy_client_port "$PORT" \
    --env_name "simpler_env_widowx/$TASK" \
    --n_episodes "$N_EPISODES" --n_envs "$N_ENVS" \
    --n_action_steps "$N_ACTION_STEPS" --max_episode_steps "$MAX_STEPS" > "$LOG" 2>&1 || echo "  client failed; see $LOG" | tee -a "$OUT_DIR/run.log"
  SR=$(grep -E "^success rate: " "$LOG" | tail -1 | awk '{print $3}')
  printf "%s\t%s\t%s\t%s\n" "$TASK" "$N_EPISODES" "${SR:-NA}" "$(( $(date +%s) - T0 ))" | tee -a "$OUT_DIR/summary.tsv"
done
echo "[$(date +%T)] done: $OUT_DIR/summary.tsv" | tee -a "$OUT_DIR/run.log"
