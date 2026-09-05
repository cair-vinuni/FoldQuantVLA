#!/usr/bin/env python3
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.
#
# Does damage to the backbone output predict which observation's chunk breaks?
#
#   python scripts/prefix_action_correlation.py            # every family's w4a4 arm
#   python scripts/prefix_action_correlation.py --arm w8a8
#
# Reads only committed records (results/<family>/<arm>/verify.json) and prints
# the Pearson correlation between the per-sample worst backbone position cosine
# and the action cosine, plus what the single most-damaged prefix decodes to.
# This regenerates the six numbers quoted in the Drift section of
# results/README.md; it needs no GPU, no engines and no upstream environment.
#
# The field differs by family because the representation the action module
# consumes does: a token sequence for GR00T, a KV stack for pi05 / SmolVLA,
# fused tokens for Evo-1. The first one present is used.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
FAMILIES = ("groot_n1_7", "groot_n1_6", "groot_n1_5", "pi05", "smolvla", "evo_1")
FIELDS = ("backbone_token_cos_min", "kv_stack_position_cos_min", "fused_tokens_position_cos_min")


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="w4a4")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    args = ap.parse_args()

    print(f"{'family':12} {'field':32} {'n':>3} {'pearson':>8}  worst prefix -> action cos")
    for family in args.families:
        record = RESULTS / family / args.arm / "verify.json"
        if not record.exists():
            print(f"{family:12} no {args.arm}/verify.json")
            continue
        samples = json.loads(record.read_text())["samples"]
        field = next((f for f in FIELDS if f in samples[0]), None)
        if field is None:
            print(f"{family:12} no backbone position field in {args.arm}/verify.json")
            continue
        prefix = [s[field] for s in samples]
        action = [s["action_cos"] for s in samples]
        worst = min(range(len(prefix)), key=lambda i: prefix[i])
        print(
            f"{family:12} {field:32} {len(samples):3} {pearson(prefix, action):+8.3f}"
            f"  {prefix[worst]:.4f} -> {action[worst]:.4f}"
        )


if __name__ == "__main__":
    main()
