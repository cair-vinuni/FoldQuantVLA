#!/usr/bin/env bash
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.
#
# Does the LIBERO rollout run? One suite, one episode per task — ten episodes,
# two to four minutes a family. It answers whether eval_libero drives the
# family's upstream loop on this machine and writes a summary, and nothing about
# success rate: ten episodes ranks nothing, and the published sweeps are 800.
#
#   scripts/smoke_eval.sh groot_n1_7
#   scripts/smoke_eval.sh                       # every family that can run it
#   EVAL_ARM=exports/w8a8/engines scripts/smoke_eval.sh groot_n1_7
#
# Only the families whose rollout runs in-process are covered. π₀.₅ and Evo-1
# drive their upstream client from a second environment against a running
# server, which is two processes and a different check; they are skipped with
# that reason rather than silently.

set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${SMOKE_OUT:-$R/.smoke}"
SUITE="${EVAL_SUITE:-libero_spatial}"
EPISODES="${EVAL_EPISODES:-1}"
mkdir -p "$OUT"

FAMILIES=("$@")
[ ${#FAMILIES[@]} -eq 0 ] && FAMILIES=(groot_n1_7 groot_n1_6 groot_n1_5 smolvla)

. "$(dirname "${BASH_SOURCE[0]}")/_gpu_busy.sh"
busy_n=$(gpu_busy_pids | grep -c . || true)
if [ "${busy_n:-0}" -gt 0 ] && [ "${SMOKE_ALLOW_BUSY_GPU:-0}" != 1 ]; then
  echo "warning: ${busy_n} process(es) already on the GPU; the simulator needs the room."
  echo "         SMOKE_ALLOW_BUSY_GPU=1 to run anyway."
  exit 2
fi

pass=0; fail=0; skip=0
note () { printf '  %-9s %s\n' "$1" "$2"; }

eval_one () {
  local fam="$1" venv="$R/models/$fam/.venv/bin/python" dir="$R/models/$fam"
  local log="$OUT/eval_$fam.log" res="$OUT/eval_$fam"
  echo "═══ $fam"
  [ -x "$venv" ] || { note SKIP "no .venv"; skip=$((skip+1)); return; }
  case "$fam" in
    pi05|evo_1) note SKIP "rollout runs from a separate client environment against a server"; skip=$((skip+1)); return ;;
  esac
  ( cd "$dir" && "$venv" -c "import importlib.util as u,sys; sys.exit(0 if u.find_spec('robosuite') else 1)" ) 2>/dev/null \
    || { note SKIP "LIBERO not installed — see the family's integration README"; skip=$((skip+1)); return; }

  # Each family's rollout takes its own upstream loop's arguments, so the flags are
  # not the same set: only N1.7 and N1.6 accept --n-envs, N1.5 wants an embodiment
  # tag, SmolVLA names its checkpoint --checkpoint and vectorizes elsewhere.
  local args=()
  case "$fam" in
    groot_n1_7) args=(--model-path "${N17_MODEL:-}" --n-envs 1) ;;
    groot_n1_6) args=(--model-path "${N16_MODEL:-}" --n-envs 1) ;;
    groot_n1_5) args=(--model-path "${N15_MODEL:-}" --embodiment-tag "${N15_TAG:-new_embodiment}") ;;
    smolvla)    args=(--checkpoint "${SMOLVLA_CKPT:-HuggingFaceVLA/smolvla_libero}") ;;
  esac
  for a in "${args[@]}"; do
    [ -z "$a" ] && { note SKIP "a required path is unset"; skip=$((skip+1)); return; }
  done
  [ -n "${EVAL_ARM:-}" ] && args+=(--engine-dir "$EVAL_ARM")

  # The N1.5 release pins no LIBERO checkout -- it is the operator's to supply --
  # while its two siblings pin one each. LIBERO is a fixed benchmark and this check
  # only asks whether the rollout runs, so borrowing a sibling's copy answers that
  # without a second clone. FOLDQUANT_LIBERO_DIR overrides.
  local env_pass=() sib
  if [ "$fam" = groot_n1_5 ] && [ -z "${FOLDQUANT_LIBERO_DIR:-}" ]; then
    for sib in groot_n1_7 groot_n1_6; do
      if [ -d "$R/models/$sib/external_dependencies/LIBERO/libero" ]; then
        env_pass=(FOLDQUANT_LIBERO_DIR="$R/models/$sib/external_dependencies/LIBERO")
        note using "${sib}'s pinned LIBERO (set FOLDQUANT_LIBERO_DIR to override)"
        break
      fi
    done
    if [ ${#env_pass[@]} -eq 0 ]; then
      note SKIP "this release pins no LIBERO; point FOLDQUANT_LIBERO_DIR at a checkout"
      skip=$((skip+1)); return
    fi
  fi

  rm -rf "$res"
  ( cd "$dir" && env MUJOCO_GL=egl ${env_pass[@]+"${env_pass[@]}"} \
      "$venv" -m foldquant_integration.eval_libero "${args[@]}" \
      --suites "$SUITE" --n-episodes "$EPISODES" --output "$res" ) >"$log" 2>&1 \
    || { note FAIL "rollout -- see $log"; fail=$((fail+1)); return; }

  if "$venv" - "$res/summary.json" <<'PY'
import json, sys

try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:  # any unreadable summary is the same failure
    print(f"  FAIL      no readable summary: {exc}")
    raise SystemExit(1)

# The families write two shapes: GR00T counts episodes itself under "per_suite",
# SmolVLA keeps upstream's percentage under "suites". Either way an empty rollout
# is a failure -- "0/0 succeeded" is not a pass, it is a check that never ran, and
# a harness that prints ok for one is worse than no harness.
if "per_suite" in d:
    suites = d["per_suite"]
    n = sum(v["num_episodes"] for v in suites.values())
    ok = sum(v["successes"] for v in suites.values())
    what = f"{ok}/{n} episodes succeeded across {len(d.get('tasks', {}))} tasks"
else:
    suites = d.get("suites", {})
    n = sum(v.get("n_episodes") or 0 for v in suites.values())
    tasks = sum(len(v.get("per_task", [])) for v in suites.values())
    rates = [v["pc_success"] for v in suites.values() if v.get("pc_success") is not None]
    pct = f"{sum(rates) / len(rates):.0f}%" if rates else "n/a"
    what = f"{pct} success over {n} episodes across {tasks} tasks"

if n == 0:
    print(f"  FAIL      rollout produced no episodes ({what}) -- see the log")
    raise SystemExit(1)
print(f"  ok        rollout   {what}")
PY
  then pass=$((pass+1)); else fail=$((fail+1)); fi
}

for fam in "${FAMILIES[@]}"; do eval_one "$fam"; done
echo
echo "eval: $pass passed, $fail failed, $skip skipped"
[ "$fail" -eq 0 ]
