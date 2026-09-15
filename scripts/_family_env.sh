# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.
#
# Sourced, not executed. Defines family_pythonpath ROOT FAMILY_DIR, which prints
# the PYTHONPATH a family's `python -m foldquant_integration.*` call needs.
#
# Every family imports the shared `foldquant` package from the repo root next to
# its own `gr00t` / `openpi` / ... package and `foldquant_integration`. `-m` puts
# the working directory on sys.path but not the root, and a script run by path
# gets neither -- so both imports fail for anyone whose venv was built without
# `uv pip install -e .`, which the per-family install_deps.sh does but a
# hand-built environment need not. smoke_family.sh computed this inline while
# smoke_serve.sh, smoke_eval.sh and bench_all.sh set nothing and died on
# `No module named 'foldquant'`; one definition keeps them from drifting.
family_pythonpath() {
  local root="$1" dir="$2" pp src
  pp="$root:$dir"
  # pi05 uses a src/ layout (src/openpi) and vendors its client as
  # packages/*/src. Those directories, not the family directory, are what its
  # imports resolve from.
  for src in "$dir/src" "$dir"/packages/*/src; do
    [ -d "$src" ] && pp="$pp:$src"
  done
  printf '%s' "$pp${PYTHONPATH:+:$PYTHONPATH}"
}
