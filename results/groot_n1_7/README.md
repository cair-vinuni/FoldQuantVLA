# GR00T N1.7 — records

Four arms (`float`, `w8a8`, `w4a4`, `w4a4_cascade`) under the protocol in
[`../README.md`](../README.md). Each arm directory holds `verify.json` (drift),
`benchmark.log` (latency) and, for the quantized arms, `foldquant_export.json`.
`w4a4/positions.json` additionally records the sixteen worst prefix positions
per observation — a depth of detail only this family has.

## Drift

32 held-out observations, seeded; PyTorch repeatability under the same seeds
is 1.000000, so these deficits are the arm's.

| arm | action mean | median | min | worst \|Δ\| | backbone cos min |
|---|---|---|---|---|---|
| `float` | 0.99977 | 0.99999 | 0.99415 | 0.436 | 0.99991 |
| `w8a8` | 0.99965 | 0.99996 | 0.99086 | 0.540 | 0.99990 |
| `w4a4` | 0.98575 | 0.99817 | 0.80213 | 1.000 | 0.99924 |
| `w4a4_cascade` | 0.98591 | 0.99825 | 0.80649 | 1.000 | 0.99924 |

## Latency

RTX 4070 Ti SUPER, 20 timed chunks after 5 warm-up, e2e median ms. Upstream's
`benchmark_inference.py` re-times eager and `torch.compile` inside every
invocation, so each row carries its own baseline — pair an arm with the eager
on its own row, not across rows.

| arm | eager | `torch.compile` | TensorRT | TRT vs eager |
|---|---|---|---|---|
| `float` | 70.00 | 56.10 | 41.50 | 1.69x |
| `w8a8` | 67.80 | 55.20 | 36.50 | 1.86x |
| `w4a4` | 67.80 | 56.20 | 32.60 | 2.08x |
| `w4a4_cascade` | 69.10 | 55.40 | 33.20 | 2.08x |

Compiling the graph with nothing quantized accounts for about three quarters
of what the W4A4 arm saves against eager; the protocol file's Latency section
carries that split and the reasoning.

## Note: prefix damage does not predict which observation flips

The W4A4 arm carries two full gripper flips, samples 18 and 21. The commit
message of `a73ebea` calls these "the same two observations `positions.json`
shows taking the worst prefix damage" — **the files do not say that**, and the
sentence should not be quoted:

| sample | ep/step | worst-position cos | rank of 32 | action cos | worst channel |
|---|---|---|---|---|---|
| 25 | 72/154 | 0.7321 | **1** | 0.9995 | z, \|Δ\| 0.027 |
| 19 | 103/60 | 0.7481 | **2** | 0.9780 | gripper, \|Δ\| 0.291 |
| 18 | 2/28 | 0.7507 | 3 | 0.8021 | **gripper, \|Δ\| 1.000** |
| 21 | 100/107 | 0.7813 | 14 | 0.8714 | **gripper, \|Δ\| 1.000** |

The two most damaged prefixes do not flip; the worst of them decodes one of
the cleanest chunks in the arm. Pearson over the 32 is +0.30 — the direction
the claim assumed, far too weak to carry it, and weak in all six families
(+0.08 to +0.31, see the protocol file). The flips are real and both land on
channel 6; what does not follow is a per-observation link between the two
depths. Pushed history is left as it is; this note is the correction of record.

## Reproducing

```bash
python scripts/results_tables.py --table drift        # the drift table above
python scripts/results_tables.py --table correlation  # the +0.30, and the other five
```
