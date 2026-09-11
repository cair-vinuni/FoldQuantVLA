#!/usr/bin/env bash
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.
#
# Does each family's policy server start and bind? One family at a time, on a
# port nothing else uses, killed as soon as it answers.
#
#   scripts/smoke_serve.sh groot_n1_7
#   scripts/smoke_serve.sh                # every family whose paths are set
#
# This checks the half a robot depends on and the export smoke does not touch:
# that serve.py assembles the policy, installs the engines if asked, and listens.
# It does NOT drive a rollout — that is eval_libero, which needs the simulator.
#
# Paths as in smoke_family.sh; add ENGINE_<FAM> to serve a built arm instead of
# the bf16 policy, e.g. ENGINE_GROOT_N1_7=exports/w8a8/engines.

set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SMOKE_PORT:-5599}"
WAIT="${SMOKE_SERVE_WAIT:-180}"
FAMILIES=("$@")
[ ${#FAMILIES[@]} -eq 0 ] && FAMILIES=(groot_n1_7 groot_n1_6 groot_n1_5 pi05 smolvla evo_1)

busy_n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
if [ "${busy_n:-0}" -gt 0 ] && [ "${SMOKE_ALLOW_BUSY_GPU:-0}" != 1 ]; then
  echo "warning: ${busy_n} process(es) already on the GPU; a server needs the room."
  echo "         SMOKE_ALLOW_BUSY_GPU=1 to run anyway."
  exit 2
fi

pass=0; fail=0; skip=0
note () { printf '  %-9s %s\n' "$1" "$2"; }

serve_one () {
  local fam="$1" venv="$R/models/$fam/.venv/bin/python" dir="$R/models/$fam"
  local log="${SMOKE_OUT:-$R/.smoke}/serve_$fam.log"
  mkdir -p "$(dirname "$log")"
  echo "═══ $fam"
  [ -x "$venv" ] || { note SKIP "no .venv"; skip=$((skip+1)); return; }

  local args=()
  case "$fam" in
    groot_n1_7) args=(--model-path "${N17_MODEL:-}" --embodiment-tag "${N17_TAG:-libero_panda}") ;;
    groot_n1_6) args=(--model-path "${N16_MODEL:-}" --embodiment-tag "${N16_TAG:-libero_panda}") ;;
    groot_n1_5) args=(--model-path "${N15_MODEL:-}" --embodiment-tag "${N15_TAG:-new_embodiment}") ;;
    pi05)       args=(--checkpoint-dir "${PI05_CKPT:-}") ;;
    evo_1)      args=(--checkpoint-dir "${EVO1_CKPT:-}") ;;
    smolvla)    args=() ;;   # builds its policy when a client connects; binding is the check
  esac
  for a in "${args[@]}"; do
    [ -z "$a" ] && { note SKIP "a required path is unset"; skip=$((skip+1)); return; }
  done
  local eng_var="ENGINE_$(echo "$fam" | tr 'a-z.' 'A-Z_')"
  local eng="${!eng_var:-}"
  [ -n "$eng" ] && args+=(--engine-dir "$eng")

  ( cd "$dir" && "$venv" -m foldquant_integration.serve "${args[@]}" --port "$PORT" ) >"$log" 2>&1 &
  local pid=$!
  # Wait on the socket, not on the log. Each family announces readiness in its own
  # words — "listening on", "port %d", "ready", "running at ws://" — and matching
  # those cost a false failure on N1.5, whose server was up and serving while the
  # check looked for a phrase it never prints. A bound port is the thing a client
  # actually needs, and it reads the same for all six.
  local up=0
  for _ in $(seq 1 "$WAIT"); do
    ss -ltn 2>/dev/null | grep -q ":$PORT\b" && { up=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if [ "$up" = 1 ]; then
    note ok "listening on :$PORT${eng:+  (engines: $eng)}"
    pass=$((pass+1))
  else
    note FAIL "no socket on :$PORT within ${WAIT}s — see $log"
    fail=$((fail+1))
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 3
}

for fam in "${FAMILIES[@]}"; do serve_one "$fam"; done
echo
echo "serve: $pass passed, $fail failed, $skip skipped"
[ "$fail" -eq 0 ]
