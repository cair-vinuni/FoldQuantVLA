# Site-selective INT8 (`o_proj` / `down_proj` at INT8 inside W4A4) across families

The paper measures this arm on GR00T N1.6 only (Table V: 773/800; Section VI-C: 35.3 ms). This
note collects the same arm on the other checkpoints, built with this release. All engines were
built and timed on one RTX 4070 Ti SUPER (sm89, TensorRT 10.15.1); latency is one repeat of 60
iterations after 10 warm-ups (the paper's protocol uses three repeats), so treat these as
preliminary. Recipe per family: LLM `w4a4_srg` with the family's ARC constants plus
`site_bits: {o: 8, down: 8}`; action module as stated per row.

## Held-out fidelity (Table VI protocol: release `verify`, 32 held-out mid-trajectory observations vs BF16 PyTorch, seed 42, four-suite calibration set, 128 samples, seed 0)

| checkpoint | arm | action cosine mean / median / min |
|---|---|---|
| GR00T N1.6 (`nvidia/GR00T-N1.6-LIBERO`) | W4A4 (Table VI) | 0.97411 / 0.99876 / 0.461 |
| | W8A8 (Table VI) | 0.99996 / 0.99998 / 0.99948 |
| | **W4A4 + o/d INT8** (LLM sq 0.6, clip 0.85; DiT `w4a4_sr` fold-before sq 0.5 — the Table V 773 recipe) | **0.99205 / 0.99967 / 0.850** |
| GR00T N1.7 (paper's four-suite fine-tune) | W4A4 (Table VI) | 0.98575 / 0.99817 / 0.802 |
| | W8A8 (Table VI) | 0.99965 / 0.99996 / 0.99086 |
| | **W4A4 + o/d INT8** (LLM sq 0.5, clip 0.85; DiT `w4a4_sr` fold-before sq 0.5) | **0.99540 / 0.99928 / 0.912** |
| pi0.5 (`pi05_libero`) | W4A4 (16 obs) | 0.99935 / 0.99955 / 0.99835 |
| | W4A4 + o/d INT8 (16 obs) | 0.99973 / 0.99978 / 0.99942 |

Median deficit falls by 73% (N1.6) and 61% (N1.7); the worst observation moves from 0.46 -> 0.85
and 0.80 -> 0.91. On pi0.5 the prefix KV-stack cosine rises 0.9665 -> 0.9864.

## Closed-loop success

| checkpoint | protocol | W4A4 | W4A4 + o/d INT8 |
|---|---|---|---|
| GR00T N1.6 | Table V campaign (H100, cap 520) | 763 (`sr` fold-before + ARC) | **773** |
| GR00T N1.7 | Table V campaign | 757 (`sr` fold-before + ARC) | pending (server run) |
| GR00T N1.7 | HoloQ-port protocol, four nvidia per-suite checkpoints, cap 720 (`groot_n1_7/HOLOQ_LIBERO.md`) | 759 | 765 |
| pi0.5 | this release's harness, openpi step profile (`pi05/HOLOQ_LIBERO.md`) | 778 | 782 |

None of the paired differences is significant at 800 episodes (N1.6 p = 0.17; N1.7 HoloQ-port p = 0.57;
pi0.5 p = 0.42); the offline improvements are.

## Latency (E2E medians, ms; head-only in parentheses where the harness reports it)

| checkpoint | eager | torch.compile | TRT bf16 | W4A4 | W4A4 + o/d INT8 | INT8 cost |
|---|---:|---:|---:|---:|---:|---|
| GR00T N1.7, nvidia `libero_10` fine-tune, `shg` head | 69.5 (39.1) | 55.7 (25.0) | 40.8 (21.4) | 32.3 (15.9) | 32.9 (15.9) | +0.6 ms, backbone |
| GR00T N1.7, four-suite fine-tune, `sr` fold-before head (Table V recipe) | 68 (38) | — | — | — | 34 (17) | head `sr` is 1.1 ms slower than `shg` |
| GR00T N1.5, four-suite fine-tune, 4 steps, K=1 | 54.2 (20.5) | — | — | 32.7 (12.1) | 33.3 (12.1) | +0.6 ms, backbone |
| pi0.5, 10 steps | 168.6--170.8 | 100.5 | — | 77.3--78.6 | 82.3--84.4 | +4--6 ms, prefix LLM |

Engine bytes for the LLM: N1.7 442 -> 577 MB, pi0.5 948 -> 1269 MB (+34% in both); action-module
engines unchanged. Table II's controls (torch.compile / TRT bf16) remain the canonical ones;
dashes mark arms not emitted by that family's release benchmark in the same run.
