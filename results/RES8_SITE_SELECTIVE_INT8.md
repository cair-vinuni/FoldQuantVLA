# Site-selective INT8 (`o_proj` / `down_proj` at INT8 inside W4A4) across families

The paper measures this arm on all four checkpoints (its closed-loop success table and its
held-out fidelity table) and recommends it. This note gathers those figures in one place beside the
checks run with this release. Recipe per family: LLM `w4a4_srg` with the family's ARC constants
plus `site_bits: {o: 8, down: 8}`; action module as stated per row. The N1.6 and N1.7 arms were
built on the dense-rotation (`w4a4_sr`) calibration preset; the N1.5 arm shares the preset of the
printed uniform-W4A4 row.

## Held-out fidelity (protocol P2)

32 held-out mid-trajectory observations against BF16 PyTorch, disjoint from the 128-observation
four-suite calibration set (seed 0). Worst |Δ| is the median over observations of each
observation's largest coordinate discrepancy.

| checkpoint | arm | mean | median | min | worst \|Δ\| |
|---|---|---:|---:|---:|---:|
| GR00T N1.7 | W8A8 | 0.99965 | 0.99996 | 0.99086 | 0.011 |
| | W4A4 | 0.98575 | 0.99817 | 0.80213 | 0.099 |
| | **W4A4 + o/d INT8** | **0.99540** | **0.99928** | **0.91244** | **0.058** |
| GR00T N1.6 | W8A8 | 0.99996 | 0.99998 | 0.99948 | 0.007 |
| | W4A4 | 0.97411 | 0.99876 | 0.46067 | 0.074 |
| | **W4A4 + o/d INT8** | **0.99205** | **0.99967** | **0.84984** | **0.039** |
| GR00T N1.5 | W8A8 | 0.99996 | 0.99999 | 0.99944 | 0.007 |
| | W4A4 | 0.99585 | 0.99885 | 0.96967 | 0.072 |
| | **W4A4 + o/d INT8** | **0.99779** | **0.99933** | **0.97276** | **0.058** |
| π₀.₅ | W8A8 | 1.00000 | 1.00000 | 0.99999 | 0.005 |
| | W4A4 | 0.99450 | 0.99942 | 0.84749 | 0.054 |
| | **W4A4 + o/d INT8** | **0.99973** | **0.99981** | **0.99850** | **0.032** |

The selective-INT8 arm raises the minimum from 0.461 to 0.850 on N1.6, 0.802 to 0.912 on N1.7,
0.970 to 0.973 on N1.5 and 0.847 to 0.998 on π₀.₅, and halves the median worst-coordinate
discrepancy on N1.6 and N1.7, while the medians stay at 0.9993–0.9998. Its cosine remains below
the W8A8 engine's on every checkpoint; what it recovers is most of the four-bit gap.

A release-harness check on 16 observations of π₀.₅ (a different sample from P2) moved the same way:
mean 0.99935 -> 0.99973, min 0.99835 -> 0.99942, prefix KV-stack cosine 0.9665 -> 0.9864.

## Closed-loop success (protocol P3)

Four suites, ten tasks, twenty initial states per task: 800 episodes per arm, on one H100 MIG
3g.40gb partition. Success rate in percent; paired over the 40 tasks. "W4A4, same preset" is the
uniform W4A4 arm built on the same calibration preset as the o/d INT8 arm. On N1.5 and π₀.₅ that is
the printed uniform-W4A4 row; on N1.7 and N1.6 it is the dense-rotation build, so it differs from the
printed W4A4 row (95.38% and 95.75%).

| checkpoint | W4A4, same preset | W4A4 + o/d INT8 | paired p |
|---|---:|---:|---:|
| GR00T N1.7 | 94.62% | 95.00% | 0.77 |
| GR00T N1.6 | 95.38% | **96.62%** | 0.17 |
| GR00T N1.5 | 87.38% | 87.00% | 0.78 |
| π₀.₅ | 97.12% | 97.62% | 0.42 |

No detectable success-rate cost and no detectable gain in any pair; the offline improvements are
resolved, the closed-loop differences are not.

Two further campaigns ran through this release's harness rather than the paper's closed-loop one, and pair
the same way: GR00T N1.7 on NVIDIA's per-suite checkpoints, cap 720 (94.88% -> 95.62%, p = 0.57;
`groot_n1_7/HOLOQ_LIBERO.md`), and π₀.₅ with the openpi step profile (97.25% -> 97.75%, p = 0.42;
`pi05/HOLOQ_LIBERO.md`).

## Latency (E2E medians, ms; head-only in parentheses where the harness reports it)

| checkpoint | eager | torch.compile | TRT bf16 | W4A4 | W4A4 + o/d INT8 | INT8 cost |
|---|---:|---:|---:|---:|---:|---|
| GR00T N1.7, nvidia `libero_10` fine-tune, `shg` head | 69.5 (39.1) | 55.7 (25.0) | 40.8 (21.4) | 32.3 (15.9) | 32.9 (15.9) | +0.6 ms, backbone |
| GR00T N1.7, four-suite fine-tune, `sr` fold-before head (the paper's closed-loop recipe) | 68 (38) | — | — | — | 34 (17) | head `sr` is 1.1 ms slower than `shg` |
| GR00T N1.5, four-suite fine-tune, 4 steps, K=1 | 54.2 (20.5) | — | — | 32.7 (12.1) | 33.3 (12.1) | +0.6 ms, backbone |
| pi0.5, 10 steps | 168.6--170.8 | 100.5 | — | 77.3--78.6 | 82.3--84.4 | +4--6 ms, prefix LLM |

Engine bytes for the LLM: N1.7 442 -> 577 MB, pi0.5 948 -> 1269 MB (+34% in both); action-module
engines unchanged. The paper's latency figure carries the canonical controls (torch.compile / TRT bf16);
dashes mark arms not emitted by that family's release benchmark in the same run.
