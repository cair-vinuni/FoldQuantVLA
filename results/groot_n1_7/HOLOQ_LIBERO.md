# GR00T N1.7, LIBERO (four suites): FoldQuant W4A4 engines against HoloQ-style and DuQuant-style W4A4 emulations

Same checkpoint, same suite, same episode budget. The checkpoints are NVIDIA's public
per-suite fine-tunes under `nvidia/GR00T-N1.7-LIBERO` (revision `2ea293aa…`), the
ones the HoloQ-style emulation was evaluated on; the campaign in the paper uses a
different, four-suite N1.7 fine-tune, so none of the paper's N1.7 rows apply here.

## Arms

Both emulated arms can now be rebuilt from this release: `models/groot_n1_7/foldquant_integration/baseline_w4a4.py` (`--method holoq | duquant`), served or rolled out with `--baseline-pack`; see that integration's README.

| arm | LLM | DiT | how it runs |
|---|---|---|---|
| HoloQ-style W4A4 (emulated, `baseline_w4a4 --method holoq`) | zigzag + SVD-Hadamard rotation, GPTQ, per-token A4 | same rotation, RTN, static per-step per-channel A4 | fake-quant: dequantise, then `F.linear` in BF16 |
| DuQuant-style W4A4 (emulated, `baseline_w4a4 --method duquant`) | zigzag + SVD-only rotation (eigvecs of WᵀW), GPTQ, static per-channel q99.9 A4 | same rotation, RTN, static per-channel A4 | fake-quant, same runtime as the row above |
| FoldQuant W4A4 | `w4a4_srg`: SmoothQuant + block-64 Hadamard, GPTQ | `w4a4_shg`: SmoothRot fold-before, butterfly, GPTQ | native INT4 TensorRT plugins (sm89 s4 tensor cores) |
| FoldQuant W4A4 + o/d INT8 | as above, `site_bits {o: 8, down: 8}` | as above | native |

Scope differs: the HoloQ-style scope is 112 LLM + 192 DiT linears (attn1 + ff in all 32
blocks) and leaves AdaLN modulation, the 4-layer `vl_self_attention` and the
cross-attention encoder KV in BF16; FoldQuant quantises those too (AdaLN weight-only INT4,
encoder KV pre-quant). Every arm calibrates each specialist on its own suite's demonstrations
(`IPEC-COMMUNITY/libero_<suite>_no_noops_1.0.0_lerobot`), 128 seeded samples, seed 0, same sampler.

## Closed-loop success, four suites, 20 episodes per task

One specialist checkpoint per suite (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
of `nvidia/GR00T-N1.7-LIBERO`, revision `2ea293aa…`), each calibrated on its own suite's IPEC
LeRobot demonstrations (128 seeded samples, seed 0). Protocol: 10 tasks x 20 episodes per suite,
`n_action_steps` 8, served action horizon 16 (the checkpoints' `processor_config.json` for
`libero_sim`), `max_episode_steps` 720. The HoloQ-style, DuQuant-style and FoldQuant arms all ran
through this release's `eval_libero` (upstream `MultiStepWrapper`, unseeded flow noise) on one RTX
4070 Ti SUPER, so they are paired within one harness. The BF16 baseline was measured with upstream
Isaac-GR00T's own LIBERO harness on an L4 (†), so it is **cross-harness** and no paired test is
possible against it.

| suite | BF16 PyTorch† | HoloQ-style W4A4 | DuQuant-style W4A4 | FoldQuant W4A4 | FoldQuant W4A4 + o/d INT8 |
|---|---:|---:|---:|---:|---:|
| spatial | 197 | 188 | 194 | 197 | 193 |
| object | 197 | 197 | 195 | 194 | 197 |
| goal | 185 | 179 | 190 | 191 | 188 |
| long (LIBERO-10) | 187 | 180 | 167 | 177 | 187 |
| **total /800** | **766** | **744** | **746** | **759** | **765** |
| % | 95.75 | 93.00 | 93.25 | 94.88 | 95.62 |
| Wilson 95% | [94.1, 96.9] | [91.0, 94.6] | [91.3, 94.8] | [93.1, 96.2] | [94.0, 96.8] |

Paired over the 40 tasks (task-clustered t(39), unadjusted):

| comparison | Δ episodes | t(39) | p |
|---|---:|---:|---:|
| FoldQuant W4A4 vs HoloQ-style | +15 | 1.24 | 0.22 |
| FoldQuant W4A4 + o/d INT8 vs HoloQ-style | +21 | 1.37 | 0.18 |
| FoldQuant W4A4 vs DuQuant-style | +13 | 1.45 | 0.16 |
| FoldQuant W4A4 + o/d INT8 vs DuQuant-style | +19 | 1.60 | 0.12 |
| DuQuant-style vs HoloQ-style | +2 | 0.13 | 0.90 |
| FoldQuant W4A4 + o/d INT8 vs FoldQuant W4A4 | +6 | 0.58 | 0.57 |

For the last pair, McNemar gives 36 vs 30 discordant episodes, p = 0.54. No pair is separable at this
budget. Both FoldQuant arms are ahead of both emulated recipes, and the largest gaps are on LIBERO-10
for DuQuant-style (−20 against o/d INT8) and on goal for HoloQ-style (−12 against FoldQuant W4A4).

† HoloQ-VLA's authors publish GR00T results only for N1.5, for which NVIDIA ships no public
checkpoint, so a same-checkpoint comparison on N1.5 is not possible. The HoloQ-style and
DuQuant-style columns are therefore **our own implementations of those recipes for N1.7**, shipped in
this release (`foldquant_integration/baselines/`, see that integration's README). The BF16
column comes from upstream Isaac-GR00T's LIBERO harness on an L4 (it is the plain baseline, not a
quantized arm), and we did not rerun BF16 in our harness. The implementation was traced against the
HoloQ-VLA reference: identical LLM and DiT scope regexes (112 + 192 linears), zigzag weight-energy
permutation with block-64 randomised SVD-Hadamard rotation, LLM GPTQ (block 128, damping 0.01),
DiT round-to-nearest weights, per-token dynamic A4 in the LLM and a static per-step per-channel A4
table (99.9th percentile) in the DiT, fake-quant execution. Two knobs of the reference are not
implemented: the permutation-energy blend `lambda_smooth = 0.15` (permutation only), and the
reference's optional SmoothQuant scale / low-rank residual variants, which belong to its
SVDQuant-style DiT builder rather than to the rotation path followed here.

## Offline fidelity, 32 held-out observations per suite vs BF16 PyTorch (seed 42)

Action cosine mean / median / min.

| suite | FoldQuant W4A4 | FoldQuant W4A4 + o/d INT8 |
|---|---|---|
| spatial | 0.9847 / 0.99899 / 0.597 | 0.9866 / 0.99964 / 0.601 |
| object | 0.9911 / 0.99958 / 0.817 | 0.9899 / 0.99977 / 0.690 |
| goal | 0.9978 / 0.99942 / 0.963 | 0.9977 / 0.99968 / 0.964 |
| long (LIBERO-10) | 0.9857 / 0.99901 / 0.780 | 0.9922 / 0.99961 / 0.888 |

The INT8 sites raise the median on every suite and the mean on three of four; the LIBERO-10
pair shows the largest gain (mean deficit -45%, median deficit -61%).

## Memory (identical engine shapes on all four checkpoints; LIBERO-10 shown)

| | float TRT | FoldQuant W4A4 | FoldQuant W4A4 + o/d INT8 |
|---|---:|---:|---:|
| LLM engine | 1614 MB | 442 MB | 577 MB |
| DiT engine | 2188 MB | 567 MB | 567 MB |
| all 7 engines | 6.5 GB | 3.5 GB | 3.6 GB |
| device memory used during rollout (nvidia-smi, 5 s samples, includes the simulator) | -- | 9.8 GB | 9.8 GB |

A fake-quant runtime (INT4 codes stored, dequantised on the fly) reports torch peak allocated
5.97 -> 4.46 GiB on an L4; that is a different
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
| float TRT | 2.7 | 16.6 | 21.4 | 40.8 | 24.5 |
| FoldQuant W4A4 | 2.7 | 13.7 | 15.9 | 32.3 | 31.0 |
| FoldQuant W4A4 + o/d INT8 | 2.7 | 14.3 | 15.9 | 32.9 | 30.4 |

Holding the two sites at INT8 costs 0.6 ms end to end (+1.7%), all of it in the language
backbone. The float TRT engine supplies (69.5 - 40.8) / (69.5 - 32.3) = 77% of the
eager-to-W4A4 reduction, in line with the paper's attribution on the four-suite checkpoint.
The HoloQ-style and DuQuant-style arms have no native-kernel latency; they run as emulation.

## Reproducing

`baseline_w4a4.py --method holoq | duquant` builds the emulated packs and `eval_libero
--baseline-pack` rolls them out; the FoldQuant arms come off the release export path with
`site_bits {o: 8, down: 8}` for the o/d INT8 arm. The N1.7 integration README lists the
commands.
