#!/usr/bin/env python3
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.
#
# Print the paper's drift and latency tables from the committed records.
#
#   python scripts/results_tables.py            # every table
#   python scripts/results_tables.py --table drift
#
# It reads only results/ (no GPU, no engines, no upstream environment), so any
# reader can reproduce every cell from the records.
#
# Tables:
#   drift        per-arm action cosine from <family>/<arm>/verify.json
#   latency      per-arm e2e median from <family>/benchmark.json, and for
#                GR00T N1.7 from <family>/<arm>/benchmark.log, which upstream's
#                script writes instead (one file per arm, eager re-timed in each)
#   correlation  per-sample backbone damage against action cosine

from __future__ import annotations

import argparse
import glob
import json
import statistics
import math
import os
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
FAMILIES = ("groot_n1_7", "groot_n1_6", "groot_n1_5", "pi05")
ARMS = ("w8a8", "w4a4")
LABEL = {
    "groot_n1_7": "GR00T N1.7",
    "groot_n1_6": "GR00T N1.6",
    "groot_n1_5": "GR00T N1.5",
    "pi05": "π₀.₅",
}
# the per-sample worst backbone position cosine, named per family by what the
# action module actually consumes
PREFIX_FIELDS = ("backbone_token_cos_min", "kv_stack_position_cos_min")
N17_BLOCK = re.compile(
    r"^(PyTorch Eager|torch\.compile|TensorRT \(n17_full_pipeline\)):\s*\n"
    r"\s*E2E:\s*median=([\d.]+).*?\n\s*Data Processing:\s*([\d.]+).*?\n"
    r"\s*Backbone:\s*([\d.]+).*?\n\s*Action Head:\s*([\d.]+)",
    re.M | re.S,
)


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def n17_logs() -> dict[str, dict[str, tuple[float, float, float, float]]]:
    """``{arm: {section: (e2e, data, backbone, head)}}`` from the per-arm logs."""
    out: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for f in sorted(glob.glob(str(RESULTS / "groot_n1_7" / "*" / "benchmark.log"))):
        arm = os.path.basename(os.path.dirname(f))
        sections = {}
        for m in N17_BLOCK.finditer(Path(f).read_text()):
            name = "TensorRT" if m.group(1).startswith("TensorRT") else m.group(1)
            sections[name] = tuple(float(m.group(i)) for i in (2, 3, 4, 5))
        if sections:
            out[arm] = sections
    return out


def drift_table() -> str:
    """Action-cosine drift, reported as the median.

    The mean and the minimum are still in every record, and they are not the
    number to read. A VLA action space is clipped, so a chunk is often railed on
    every channel at once; a cosine between two railed vectors compares their
    signs and comes out at 1.000 whatever the arm did, while a chunk with two
    channels railed and the rest near zero lets a small absolute error swing the
    cosine to 0.02. The distribution is bimodal and the mean averages across the
    two modes -- on a GR00T N1.6 bridge W4A4 arm it read 0.842 while the median
    read 0.9985 and the arm's LIBERO success rate was within a point of bf16.
    The median answers the question the table is asking: is the typical action
    the same action?
    """
    rows = ["| family | arm | n | action cos (median) | median worst \\|Δ\\| |", "|---|---|---|---|---|"]
    for fam in FAMILIES:
        for arm in ARMS:
            d = _load(RESULTS / fam / arm / "verify.json")
            if d is None:
                continue
            a = d["actions"]
            med = "-" if a.get("cos_median") is None else f"{a['cos_median']:.5f}"
            rows.append(f"| {LABEL[fam]} | `{arm}` | {d['num_samples']} | {med} | {_median_worst(d)} |")
    return "\n".join(rows)


def _median_worst(record: dict) -> str:
    """Median over observations of each observation's largest coordinate discrepancy.

    This is the paper's "Worst |Δ|" column. It is *not* ``actions.max_abs``,
    which is the single largest discrepancy over the whole held-out set and is
    dominated by one event -- on N1.7 W8A8 that is 0.540 against a median of
    0.011. A median of per-observation maxima says how far the typical action
    moves on its worst channel; the global maximum says only that some
    observation once did.
    """
    per = [s["action_max_abs"] for s in record.get("samples", []) if s.get("action_max_abs") is not None]
    return f"{statistics.median(per):.3f}" if per else "-"


def latency_table() -> str:
    rows = [
        "| family | eager | W8A8 | W4A4 | W4A4 vs eager |",
        "|---|---|---|---|---|",
    ]
    logs = n17_logs()
    if logs:
        cell = {a: logs[a]["TensorRT"][0] for a in logs}
        eager = {a: logs[a]["PyTorch Eager"][0] for a in logs}
        row = [LABEL["groot_n1_7"], f"{eager['w4a4']:.2f}"]
        row += [f"{cell[a]:.2f}" if a in cell else "-" for a in ARMS]
        row.append(f"{eager['w4a4'] / cell['w4a4']:.2f}x")
        rows.append("| " + " | ".join(row) + " |")
    for fam in FAMILIES:
        d = _load(RESULTS / fam / "benchmark.json")
        if d is None:
            continue
        arms = d["arms"]
        e = arms["PyTorch Eager"]["median_ms"]["e2e"]
        row = [LABEL[fam], f"{e:.2f}"]
        row += [f"{arms[a]['median_ms']['e2e']:.2f}" if a in arms else "-" for a in ARMS]
        row.append(f"{e / arms['w4a4']['median_ms']['e2e']:.2f}x" if "w4a4" in arms else "-")
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return math.nan if sx == 0 or sy == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def correlation_table() -> str:
    rows = ["| family | field | n | Pearson | worst prefix → action cos |", "|---|---|---|---|---|"]
    for fam in FAMILIES:
        d = _load(RESULTS / fam / "w4a4" / "verify.json")
        if d is None:
            continue
        samples = d["samples"]
        field = next((f for f in PREFIX_FIELDS if f in samples[0]), None)
        if field is None:
            continue
        prefix = [s[field] for s in samples]
        action = [s["action_cos"] for s in samples]
        w = min(range(len(prefix)), key=lambda i: prefix[i])
        rows.append(
            f"| {LABEL[fam]} | `{field}` | {len(samples)} | {_pearson(prefix, action):+.2f} "
            f"| {prefix[w]:.4f} → {action[w]:.4f} |"
        )
    return "\n".join(rows)


TABLES = {"drift": drift_table, "latency": latency_table, "correlation": correlation_table}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", choices=[*TABLES, "all"], default="all")
    args = ap.parse_args()
    names = list(TABLES) if args.table == "all" else [args.table]
    for i, name in enumerate(names):
        if i:
            print()
        print(f"## {name}\n")
        print(TABLES[name]())


if __name__ == "__main__":
    main()
