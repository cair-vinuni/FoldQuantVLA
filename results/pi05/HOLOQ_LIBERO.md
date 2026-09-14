# pi0.5, LIBERO: FoldQuant engines against the W4A4 methods tabulated by HoloQ-VLA

Same public checkpoint on both sides: OpenPI `pi05_libero`, converted to PyTorch. HoloQ-VLA
(arXiv 2605.28803 v3, Table 2) reports it and four W4A4 baselines it ran itself, all as
**fake quantisation**; the FoldQuant rows are the paper's closed-loop campaign on the same
checkpoint, executed on native INT8/INT4 TensorRT engines (H100 MIG, INT4 lowered to the INT8
datapath there). Percentages are converted to successes out of 200 per suite; every HoloQ-VLA
percentage is a multiple of 0.5, consistent with 20 trials per task.

| method | runs as | spatial | object | goal | long | total /800 | % |
|---|---|---:|---:|---:|---:|---:|---:|
| FP16 baseline (HoloQ-VLA harness) | PyTorch | 198 | 195 | 197 | 187 | 777 | 97.1 |
| SmoothQuant W4A4 (HoloQ-VLA harness) | fake-quant | 166 | 176 | 80 | 52 | 474 | 59.3 |
| DuQuant W4A4 (HoloQ-VLA harness) | fake-quant | 192 | 198 | 188 | 176 | 754 | 94.3 |
| QuantVLA W4A4 (HoloQ-VLA harness) | fake-quant, sensitive layers float | 188 | 196 | 160 | 112 | 656 | 82.0 |
| HoloQ-VLA W4A4 (HoloQ-VLA harness) | fake-quant | 198 | 194 | 200 | 192 | 784 | 98.0 |
| BF16 PyTorch (our harness) | PyTorch | 199 | 196 | 196 | 181 | 772 | 96.5 |
| TRT bf16 (our harness) | TensorRT float | 200 | 200 | 198 | 186 | 784 | 98.0 |
| ModelOpt W8A8 SmoothQuant (our harness) | native INT8 | 197 | 197 | 194 | 186 | 774 | 96.8 |
| ModelOpt W4A16 AWQ (our harness) | native weight-only | 199 | 197 | 197 | 191 | 784 | 98.0 |
| FoldQuant W8A8 (our harness) | native INT8 | 199 | 199 | 193 | 188 | 779 | 97.4 |
| FoldQuant head W4A4 / LM W8A8 (our harness) | native | 199 | 199 | 196 | 187 | 781 | 97.6 |
| FoldQuant W4A4 (our harness) | native INT4 | 197 | 198 | 196 | 186 | 777 | 97.1 |

## What is and is not comparable

* Checkpoint: identical (`pi05_libero`). Executed prefix: identical (5 actions per call).
* Harness: different. HoloQ-VLA uses OpenPI's LIBERO evaluator with per-suite step caps
  220 / 280 / 600 / 1000 (its `groot` profile); our campaign caps every suite at 520. Our cap is
  shorter on goal and long-horizon, longer on spatial and object. Initial states come from LIBERO's
  shipped init-state files on both sides. Flow-matching noise is unseeded in our harness.
* Two BF16 references, 777 and 772, differ by 5 episodes across harnesses; the two float
  export controls of our harness sit at 784. All quantised rows above 770 lie inside the spread of
  the unquantised references, on either side, so the ordering among them is not resolved by 800
  episodes (see the paper's noise-floor discussion).
* The three "other method" rows are HoloQ-VLA's own re-runs of those methods as fake
  quantisation; SmoothQuant and QuantVLA collapse on goal and long-horizon there, DuQuant loses
  23 episodes. FoldQuant W4A4 executes natively and loses none against its own BF16 reference
  (+5). We have not re-run those methods ourselves on pi0.5.

## Site-selective INT8 and a second harness (this release, RTX 4070 Ti SUPER)

The same checkpoint run through this release's `eval_libero` (upstream openpi LIBERO client,
openpi max-steps profile 220 / 280 / 300 / 520, 20 trials per task, `replan_steps` 5, seed 7),
engines built here with 128 four-suite calibration samples (seed 0). Two arms, paired within
this harness; cross-harness to every row above.

| method | spatial | object | goal | long | total /800 | % | Wilson 95% |
|---|---:|---:|---:|---:|---:|---:|---|
| FoldQuant W4A4 (this harness, sm89 native INT4) | 196 | 199 | 192 | 191 | 778 | 97.2 | [95.9, 98.2] |
| FoldQuant W4A4 + `o_proj`/`down_proj` INT8 | 199 | 198 | 197 | 188 | 782 | 97.8 | [96.5, 98.6] |

Paired over the 40 tasks: +4 episodes for the INT8 sites, t(39) = 0.81, p = 0.42; McNemar 17 vs 13
discordant episodes, p = 0.59. Not separable at this budget, as on N1.7 (+6, p = 0.57) and N1.6
(+10, p = 0.17) in the paper. The W4A4 row lands within one episode of the campaign's W4A4 row on
H100 (778 vs 777), so the two harnesses agree on this checkpoint.

Offline, 16 held-out observations vs BF16 PyTorch: action cosine mean 0.99935 -> 0.99973 and
min 0.99835 -> 0.99942; prefix KV-stack cosine 0.9665 -> 0.9864. Engine bytes: LLM 948 -> 1269 MB
(+34%), expert 401 MB unchanged. A campaign-harness build of the same res8 arm on H100 gave the same
LLM growth (904 -> 1210 MB, +34%) and loaded the INT8 per-row plugin beside the INT4 one; its
rollout was stopped after 73 episodes and is not a result.

Latency on the same GPU (release `benchmark`, 60 iterations after 10 warm-ups, one repeat, 10
denoising steps, E2E medians): eager 168.6--170.8 ms, `torch.compile(max-autotune)` 100.5 ms,
W4A4 77.3--78.6 ms, W4A4 + `o_proj`/`down_proj` INT8 82.3--84.4 ms. The INT8 sites cost about
5 ms, all in the PaliGemma prefix pass (17.1 -> 22.9 ms); the ten-step denoise loop is unchanged
(31.0 ms). Ranges are two runs of the same engines.
