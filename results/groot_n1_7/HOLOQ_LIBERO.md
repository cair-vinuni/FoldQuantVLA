# GR00T N1.7, LIBERO (four suites): FoldQuant W4A4 engines against a HoloQ-style W4A4 emulation

Same checkpoint, same suite, same episode budget. The checkpoints are NVIDIA's public
per-suite fine-tunes under `nvidia/GR00T-N1.7-LIBERO` (revision `2ea293aa…`), the
ones the HoloQ-style emulation was evaluated on; the campaign in the paper uses a
different, four-suite N1.7 fine-tune, so none of the paper's N1.7 rows apply here.

## Arms

Both emulated arms can now be rebuilt from this release: `models/groot_n1_7/foldquant_integration/baseline_w4a4.py` (`--method holoq | duquant`), served or rolled out with `--baseline-pack`; see that integration's README.

| arm | LLM | DiT | how it runs |
|---|---|---|---|
| HoloQ-style W4A4 (external) | zigzag + SVD-Hadamard rotation, GPTQ, per-token A4 | same rotation, RTN, static per-step per-channel A4 | fake-quant: dequantise, then `F.linear` in BF16 |
| DuQuant-style W4A4 (emulated, `baseline_w4a4 --method duquant`) | zigzag + SVD-only rotation (eigvecs of WᵀW), GPTQ, static per-channel q99.9 A4 | same rotation, RTN, static per-channel A4 | fake-quant, same runtime as the row above |
| FoldQuant W4A4 | `w4a4_srg`: SmoothQuant + block-64 Hadamard, GPTQ | `w4a4_shg`: SmoothRot fold-before, butterfly, GPTQ | native INT4 TensorRT plugins (sm89 s4 tensor cores) |
| FoldQuant W4A4 + o/d INT8 | as above, `site_bits {o: 8, down: 8}` | as above | native |

Scope differs: the HoloQ-style scope is 112 LLM + 192 DiT linears (attn1 + ff in all 32
blocks) and leaves AdaLN modulation, the 4-layer `vl_self_attention` and the
cross-attention encoder KV in BF16; FoldQuant quantises those too (AdaLN weight-only INT4,
encoder KV pre-quant). Both calibrate each specialist on its own suite's demonstrations
(`IPEC-COMMUNITY/libero_<suite>_no_noops_1.0.0_lerobot`; FoldQuant: 128 seeded samples, seed 0;
HoloQ-style: 10 trajectories).

## Closed-loop success, four suites, 20 episodes per task

One specialist checkpoint per suite (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
of `nvidia/GR00T-N1.7-LIBERO`, revision `2ea293aa…`), each calibrated on its own suite's IPEC
LeRobot demonstrations (FoldQuant: 128 seeded samples, seed 0; HoloQ-style port: 10 trajectories).
Protocol matched to the port's report: 10 tasks x 20 episodes per suite, `n_action_steps` 8, served
action horizon 16 (the checkpoints' `processor_config.json` for `libero_sim`), `max_episode_steps`
720. FoldQuant arms ran through this release's `eval_libero` (upstream `MultiStepWrapper`, unseeded
flow noise) on an RTX 4070 Ti SUPER; the BF16 baseline and the HoloQ-style W4A4 are as reported by
the port's own harness on an L4, so the two left columns are **cross-harness** and no paired test is
possible against them. The two FoldQuant arms are paired within one harness.

| suite | BF16 PyTorch† | HoloQ-style W4A4† | FQ W4A4 | FQ W4A4 + o/d INT8 |
|---|---:|---:|---:|---:|
| spatial | 197 | 195 | 197 | 193 |
| object | 197 | 194 | 194 | 197 |
| goal | 185 | 178 | 191 | 188 |
| long (LIBERO-10) | 187 | 178 | 177 | 187 |
| **total /800** | **766** | **745** | **759** | **765** |
| % | 95.75 | 93.12 | 94.88 | 95.62 |
| Wilson 95% | [94.1, 96.9] | [91.2, 94.7] | [93.1, 96.2] | [94.0, 96.8] |

FoldQuant W4A4 + o/d INT8 vs FoldQuant W4A4, paired over the 40 tasks: +6 episodes,
t(39) = 0.58, p = 0.57; McNemar 36 vs 30 discordant episodes, p = 0.54.
The two FoldQuant arms are not separable at this budget; the largest per-suite gap is on LIBERO-10
(+10 for the INT8 sites), the others are within four episodes either way. Per-task counts and
per-episode outcomes are in each arm's `summary.json`.

† HoloQ-VLA's authors publish GR00T results only for N1.5, for which NVIDIA ships no public
checkpoint, so a same-checkpoint comparison on N1.5 is not possible. The "HoloQ-style W4A4" column
is therefore **our own port of the HoloQ recipe to N1.7** (fork branch `duc-quan`, `quant-report.md`),
evaluated in that fork's upstream LIBERO harness on an L4 together with the unquantised BF16
checkpoint (the BF16 column is the plain baseline, not a HoloQ arm). The port was traced against the
Omega-QVLA reference: identical LLM and DiT scope regexes (112 + 192 linears), zigzag weight-energy
permutation with block-64 randomised SVD-Hadamard rotation, LLM GPTQ (block 128, damping 0.01),
DiT round-to-nearest weights, per-token dynamic A4 in the LLM and a static per-step per-channel A4
table (99.9th percentile) in the DiT, fake-quant execution. Two knobs of the reference are not in the
port: the permutation-energy blend `lambda_smooth = 0.15` (permutation only), and the reference's
optional SmoothQuant scale / low-rank residual variants, which belong to its SVDQuant-style DiT
builder rather than to the A2-lite rotation path the port follows. We did not rerun BF16 in our
harness.

## Offline fidelity, 32 held-out observations per suite vs BF16 PyTorch (seed 42)

Action cosine mean / median / min.

| suite | FQ W4A4 | FQ W4A4 + o/d INT8 |
|---|---|---|
| spatial | 0.9847 / 0.99899 / 0.597 | 0.9866 / 0.99964 / 0.601 |
| object | 0.9911 / 0.99958 / 0.817 | 0.9899 / 0.99977 / 0.690 |
| goal | 0.9978 / 0.99942 / 0.963 | 0.9977 / 0.99968 / 0.964 |
| long (LIBERO-10) | 0.9857 / 0.99901 / 0.780 | 0.9922 / 0.99961 / 0.888 |

The INT8 sites raise the median on every suite and the mean on three of four; the LIBERO-10
pair shows the largest gain (mean deficit -45%, median deficit -61%).

## Memory (identical engine shapes on all four checkpoints; LIBERO-10 shown)

| | float TensorRT | FQ W4A4 | FQ W4A4 + o/d INT8 |
|---|---:|---:|---:|
| LLM engine | 1614 MB | 442 MB | 577 MB |
| DiT engine | 2188 MB | 567 MB | 567 MB |
| all 7 engines | 6.5 GB | 3.5 GB | 3.6 GB |
| device memory used during rollout (nvidia-smi, 5 s samples, includes the simulator) | -- | 9.8 GB | 9.8 GB |

The HoloQ-style report gives torch peak allocated 5.97 -> 4.46 GiB for its fake-quant
runtime on an L4 (INT4 codes stored, dequantised on the fly); that is a different
accounting from serialized engine size or device peak and is not comparable
number-for-number. The comparable statement is the weight footprint of the quantised
modules: FoldQuant's LLM + DiT engines are 1.01 GB against 3.80 GB in float (3.8x).

## Latency (RTX 4070 Ti SUPER, batch 1)

Upstream `benchmark_inference.py` through this release's `benchmark` wrapper, 60 iterations
after 10 warm-ups, seed 42, medians in ms. Eager is the BF16 PyTorch policy in the same
process (it varies 66.5--69.5 ms across the runs; one number is quoted); the `torch.compile` row is
this runtime's, measured in a separate run (float engine 41.0 ms there).

| arm | data proc. | backbone | action head | end to end | Hz |
|---|---:|---:|---:|---:|---:|
| PyTorch eager (bf16) | 2.7 | 27.5 | 39.1 | 69.5 | 14.4 |
| PyTorch `torch.compile` | 2.7 | 28.0 | 25.0 | 55.7 | 17.9 |
| float TensorRT | 2.7 | 16.6 | 21.4 | 40.8 | 24.5 |
| FQ W4A4 | 2.7 | 13.7 | 15.9 | 32.3 | 31.0 |
| FQ W4A4 + o/d INT8 | 2.7 | 14.3 | 15.9 | 32.9 | 30.4 |

Holding the two sites at INT8 costs 0.6 ms end to end (+1.7%), all of it in the language
backbone. The float TensorRT engine supplies (69.5 - 40.8) / (69.5 - 32.3) = 77% of the
eager-to-W4A4 reduction, in line with the paper's attribution on the four-suite checkpoint.
The HoloQ-style report has no native-kernel latency; its runtime is emulation.

## Files

`exports/n17l10_*/` (LIBERO-10) and `exports/n17_{spatial,object,goal}_*/` in the working tree (engines, `verify.json`,
`libero10_cap720/summary.json` with per-episode outcomes, `gpu_mem_used_mib.log`);
run script `exports/run_n17_libero10_holoq_cmp.sh`.
