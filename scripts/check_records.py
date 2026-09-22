#!/usr/bin/env python3
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.
#
# Check committed results without a GPU, engines, or model dependencies.
#
#   python scripts/check_records.py
#
# Checks cover the float/W8A8 ladder where a float arm is committed, declared
# scope, held-out samples, and local paths.

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
FAMILIES = ("groot_n1_7", "groot_n1_6", "groot_n1_5", "pi05")
SEAM_KEYS = ("backbone_features", "kv_stack")
# Match Linux, macOS, and Windows home paths without depending on a username.
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

        # ladder: float must be at least as close to bf16 as W8A8.
        # Clipped actions produce a bimodal cosine distribution; compare medians.
        # Allow 1e-3 for reduction-order differences, then check the backbone
        # output separately to catch errors that action clipping can obscure.
        if "float" in arms and "w8a8" in arms:
            f = _action_cos(arms["float"])
            w = _action_cos(arms["w8a8"])
            if f < w - 1e-3:
                problems.append(
                    f"{fam}: float arm ({f:.5f}) drifts further than W8A8 ({w:.5f}). "
                    f"A float engine cannot be less faithful than an INT8 one built from the "
                    f"same graph; check it was compiled STRONGLY_TYPED."
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

        # scope: a float record must say what it covers
        if "float" in arms:
            r = arms["float"]
            if not r.get("schemes") and not r.get("components"):
                problems.append(
                    f"{fam}: the float record names neither `schemes` nor `components`, so its "
                    f"scope is unstated; two arms with different scope have shared this name."
                )

        # every arm records its held-out status and sample count
        for name, r in arms.items():
            if r.get("num_samples") is None:
                problems.append(f"{fam}/{name}: record has no num_samples")
            if r.get("held_out") is False:
                problems.append(f"{fam}/{name}: scored on calibration episodes (held_out false)")

    # provenance: no operator filesystem anywhere under results/
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
