#!/usr/bin/env python3
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.
#
# Check committed results without a GPU, engines, or model dependencies.
#
#   python scripts/check_records.py
#
# Every family must carry a w8a8 and a w4a4 verify record and a latency record.
# Each verify record must name its dataset, seed and schemes, be scored held
# out, hold exactly num_samples finite per-sample cosines whose minimum,
# median and mean equal the stored aggregates, and contain no operator path.

from __future__ import annotations

import json
import math
import re
import statistics
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


ARMS = ("w8a8", "w4a4")
REQUIRED = ("dataset_path", "seed", "num_samples", "held_out", "schemes", "actions", "samples")


def _finite_unit(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and -1.0 <= value <= 1.0


def check() -> list[str]:
    problems: list[str] = []
    for fam in FAMILIES:
        arms = {}
        for arm in ARMS:
            p = RESULTS / fam / arm / "verify.json"
            if not p.is_file():
                problems.append(f"{fam}/{arm}: verify.json is missing")
                continue
            arms[arm] = json.loads(p.read_text())
        for p in sorted((RESULTS / fam).glob("*/verify*.json")):
            arms.setdefault(p.parent.name, json.loads(p.read_text()))
        if not (RESULTS / fam / "benchmark.json").is_file() and not list((RESULTS / fam).glob("*/benchmark.log")):
            problems.append(f"{fam}: no latency record (benchmark.json or <arm>/benchmark.log)")

        for name, r in arms.items():
            where = f"{fam}/{name}"
            missing = [k for k in REQUIRED if k not in r]
            if missing:
                problems.append(f"{where}: record lacks {missing}")
                continue
            if r["held_out"] is not True:
                problems.append(f"{where}: held_out is {r['held_out']!r}; the arm must be scored on episodes the calibration never saw")
            samples = r["samples"]
            if len(samples) != r["num_samples"] or not samples:
                problems.append(f"{where}: num_samples {r['num_samples']} but {len(samples)} samples recorded")
                continue
            cos = [s.get("action_cos") for s in samples]
            if not all(_finite_unit(c) for c in cos):
                problems.append(f"{where}: a per-sample action cosine is missing, non-finite or outside [-1, 1]")
                continue
            a = r["actions"]
            for key, ref in (("cos_min", min(cos)), ("cos_median", statistics.median(cos)), ("cos_mean", sum(cos) / len(cos))):
                if key in a and a[key] is not None and abs(float(a[key]) - ref) > 1e-6:
                    problems.append(f"{where}: actions.{key} = {a[key]} does not equal the {key[4:]} of the samples ({ref:.7f})")
            if len({(s.get("episode"), s.get("step")) for s in samples}) != len(samples):
                problems.append(f"{where}: two samples share an (episode, step)")

        # ladder: a float arm, where committed, must be at least as close to bf16 as W8A8.
        # Clipped actions produce a bimodal cosine distribution; compare medians, with
        # 1e-3 for reduction-order differences. This is a plausibility check on the
        # float control, not a theorem: it catches the weakly-typed build that once
        # ran layers in fp32 and drifted further than the INT8 engine.
        if "float" in arms and "w8a8" in arms:
            f = _action_cos(arms["float"])
            w = _action_cos(arms["w8a8"])
            if f < w - 1e-3:
                problems.append(
                    f"{fam}: float arm ({f:.5f}) drifts further than W8A8 ({w:.5f}); "
                    f"check the float engine was compiled STRONGLY_TYPED."
                )
            fs, ws = _seam(arms["float"])[1], _seam(arms["w8a8"])[1]
            if fs and ws:
                fk = fs.get("token_cos_min", fs.get("position_cos_min"))
                wk = ws.get("token_cos_min", ws.get("position_cos_min"))
                if fk is not None and wk is not None and fk < wk - 0.05:
                    problems.append(f"{fam}: float seam min ({fk:.4f}) is far below W8A8's ({wk:.4f})")
            r = arms["float"]
            if not r.get("schemes") and not r.get("components"):
                problems.append(f"{fam}: the float record names neither `schemes` nor `components`")

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
