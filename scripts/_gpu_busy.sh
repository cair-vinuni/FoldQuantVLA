# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.
#
# Sourced, not executed. Defines gpu_busy_pids, which prints the PIDs of the
# processes currently holding a CUDA context, one per line (nothing when idle).
#
# `nvidia-smi --query-compute-apps=pid` answers "[N/A]" on Tegra (L4T): the iGPU
# exposes no per-process accounting. Every script here used to count that line
# with `grep -c .`, so on a Jetson they all saw one process on an idle card and
# skipped or refused -- on the platform docs/deploy/jetson.md targets. Keeping
# the check in one place is what stops the copies drifting apart again.

gpu_busy_pids() {
  local pids
  # dGPU: nvidia-smi reports real PIDs; keep only lines that are one.
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
         | grep -oE '^[0-9]+$' || true)
  if [ -z "$pids" ] && [ -d /dev/nvgpu ]; then
    # Tegra: CUDA opens /dev/nvgpu, not /dev/nvidia* (that is only the display
    # stack). The desktop session holds the same node, so keep only processes
    # that mapped the CUDA driver -- otherwise gnome-shell counts as busy.
    local pid
    for pid in $(fuser /dev/nvgpu/*/* 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$'); do
      [ "$pid" = "$$" ] && continue
      grep -qE 'libcuda\.so|libcudart' "/proc/$pid/maps" 2>/dev/null && pids="${pids}${pid}"$'\n'
    done
  fi
  printf '%s' "$pids" | grep -E '^[0-9]+$' || true
}
