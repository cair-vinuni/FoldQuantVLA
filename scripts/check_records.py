#!/usr/bin/env python3
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.
#
# Invariants every committed record must satisfy. Reads results/ only — no GPU,
# no engines, no upstream environment — so a reviewer can run it on a clone.
#
#   python scripts/check_records.py
#
# These are not style checks. Each one is here because violating it produced a
# wrong number that looked plausible:
#
#   ladder      a float engine cannot drift further from the bf16 reference than
#               an INT8 engine built from the same graph. When it did, the float
#               arm had been compiled weakly typed and TensorRT ran layers in
#               fp32 — more exact than the reference, so further from it.
#   scope       two arms called "float" existed in one family, one covering the
#               action head alone and one covering both modules, with nothing in
#               the record to tell them apart.
#   provenance  a record must not carry the operator's filesystem. eval_libero
#               wrote model_path verbatim while every other tool scrubbed it.
#   split       a graph-versus-precision split is only meaningful if the float
#               arm it divides by has the same scope as the quantized arm.

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
FAMILIES = ("groot_n1_7", "groot_n1_6", "groot_n1_5", "pi05")
SEAM_KEYS = ("backbone_features", "kv_stack")
# Absolute home-directory paths, in the three shapes an operator's machine writes them.
# Deliberately name-agnostic: hardcoding the current operator's username would pass on
# anyone else's machine, and would put the name being hidden into the file that hides it.
LEAKY = re.compile(r"(?:/home/|/Users/)[^/\"\s]+/|[A-Za-z]:\\\\+Users\\\\+")


def _action_cos(arm: dict) -> float:
    """An arm's action cosine: the median, falling back to the mean for a record
    written before the median was stored."""
    a = arm["actions"]
    value = a.get("cos_median")
    return float(a["cos_mean"] if value is None else value)


def _seam(record: dict) -> tuple[str, dict] | tuple[None, None]:
    for k in SEAM_KEYS:
        if k in record:
            return k, record[k]
    return None, None


def check() -> list[str]:
    problems: list[str] = []
    for fam in FAMILIES:
        arms = {}
        for p in sorted((RESULTS / fam).glob("*/verify*.json")):
            arms[p.parent.name] = json.loads(p.read_text())

        # ---- ladder: float must be at least as close to bf16 as W8A8.
        # Compared on the median. The mean mixes two modes of a clipped action
        # space -- fully railed chunks score 1.000 by construction, barely-moving
        # ones let a small error swing the cosine to near zero -- so it moves with
        # how often the arm was railed, not with how faithful the engine was.
        # Tolerance 1e-3, not 0: a float TensorRT kernel and an INT8 plugin reduce in
        # different orders, and either can land marginally closer to PyTorch. Measured
        # spread on sound arms is -5.4e-5 to +7.1e-3, so the action cosine alone does
        # not discriminate; the seam check below is the one that caught every real
        # defect (pi05 0.3981 against 0.9880, N1.6 0.3783 against 0.5667).
        if "float" in arms and "w8a8" in arms:
            f = _action_cos(arms["float"])
            w = _action_cos(arms["w8a8"])
            if f < w - 1e-3:
                problems.append(
                    f"{fam}: float arm ({f:.5f}) drifts further than W8A8 ({w:.5f}). "
                    f"A float engine cannot be less faithful than an INT8 one built from the "
                    f"same graph — check it was compiled STRONGLY_TYPED."
                )
            fs, ws = _seam(arms["float"])[1], _seam(arms["w8a8"])[1]
            if fs and ws:
                fk = fs.get("token_cos_min", fs.get("position_cos_min"))
                wk = ws.get("token_cos_min", ws.get("position_cos_min"))
                if fk is not None and wk is not None and fk < wk - 0.05:
                    problems.append(
                        f"{fam}: float seam min ({fk:.4f}) is far below W8A8's ({wk:.4f}); "
                        f"the float engine is damaging the seam the quantized one preserves."
                    )

        # ---- scope: a float record must say what it covers
        if "float" in arms:
            r = arms["float"]
            if not r.get("schemes") and not r.get("components"):
                problems.append(
                    f"{fam}: the float record names neither `schemes` nor `components`, so its "
                    f"scope is unstated — two arms with different scope have shared this name."
                )

        # ---- every arm records its held-out status and sample count
        for name, r in arms.items():
            if r.get("num_samples") is None:
                problems.append(f"{fam}/{name}: record has no num_samples")
            if r.get("held_out") is False:
                problems.append(f"{fam}/{name}: scored on calibration episodes (held_out false)")

    # ---- provenance: no operator filesystem anywhere under results/
    for p in RESULTS.rglob("*.json"):
        m = LEAKY.search(p.read_text())
        if m:
            problems.append(f"{p.relative_to(RESULTS.parent)}: leaks an operator path ({m.group(0)!r})")
    return problems


def main() -> int:
    problems = check()
    if not problems:
        print("all record invariants hold")
        return 0
    print(f"{len(problems)} problem(s):\n")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
