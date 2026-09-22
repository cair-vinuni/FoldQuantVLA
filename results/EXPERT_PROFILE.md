# Where a denoising step goes: per-layer profile of the expert engines

TensorRT `IProfiler` over the expert engine alone (opt shapes, 100-200 iterations after warm-up),
RTX 4070 Ti SUPER. "Wall" is the un-profiled time of one `execute_async_v3` (no CUDA graph);
per-kind sums carry per-layer profiler overhead and are read as shares, not as absolute ms.

| engine | wall / step | TensorRT fused MHA | FoldQuant plugins + elementwise | float GEMMs | of which adaRMS `dense` |
|---|---|---|---|---|---|
| π₀.₅ float | 3.34 ms | 18 × 87 µs = 1.56 ms (**47 %**) | 0.33 | 1.34 | 0.38 (one fused GEMM, all 54 norms) |
| π₀.₅ W4A4 | 3.02 ms | 18 × 87 µs = 1.57 ms (**52 %**) | 1.74 | - | (inside the plugin path) |

**π₀.₅: half of every expert step is attention, and precision never touches it.** The expert
attends ten action queries to a 968-token prefix through one KV head; TensorRT's fused
multi-head-attention kernel (`_gemm_mha_v2`) takes 87 µs per layer at that decode-like shape,
about 60× the bandwidth floor of the 1 MB of K/V it reads. Four bits cut the rest of the step
from 1.78 to 1.44 ms and the step as a whole by 10 %. This is why `torch.compile` (100.3 ms), whose
attention is PyTorch SDPA, is the stronger floating-point control on this family and the float
engine (111.8) is not: the gap is the attention kernel, on every arm alike. CUDA-graph replay
confirmed the same from the other side: it removed ≤1.5 ms of a 30 ms loop. An
attention emission fit to the ten-query shape (a plugin, or a decomposed MatMul-Softmax-MatMul
that lets the builder pick GEMV-class kernels) would move every π₀.₅ arm by a similar amount and
is the next lever on this family; it is not a precision change, but it is a new engine, so it is
future work rather than a re-timing.

## Follow-up: taking the fused attention kernel out of the π₀.₅ float expert

`Softmax` rewritten as `ReduceMax → Sub → Exp → ReduceSum → Div` (the same function; the builder
can no longer form its fused-MHA pattern), expert engine rebuilt, everything else unchanged
Per-layer profile: `_gemm_mha_v2` 18 → 0, step 3.34 → **2.82 ms**.
Held-out verify (n = 32): actions cosine 0.99035 / min 0.91525 against 0.99037 / 0.91556 for the
fused engine, identical to the fourth digit, as expected from a kernel change. Benchmark in one
process: eager 164.6, float 110.5 (loop 34.4), float-no-MHA **105.0** (loop 28.9); `torch.compile`
is 100.3.

The remaining 5 ms is scope, not kernels: upstream's `torch.compile` wraps the whole of
`sample_actions`, vision tower included (`pi0_pytorch.py:113`), while every engine arm here leaves
the SigLIP tower in eager PyTorch at 16.7 ms. A float vision engine would close that on every arm
alike. Both changes are engine changes rather than re-timings: the paper reports the fused-attention
engines it measured in closed loop, and cites this experiment as the measured direction of the fix.
