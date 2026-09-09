# Float-engine arms (`--llm-scheme float --dit-scheme float`)

The unquantized engine of every module, traced from the live policy under the runtime's
binding names and built weakly typed with no plugins. It is the floor of each ladder and the
compiled control the paper's latency table divides by — and, since it comes off the same
`export_foldquant` → `build_engines` → `install_engines` path as W8A8 and W4A4, the arms
differ in nothing but the precision of the projections.

Held-out verify (n = 32, seeded, vs bf16 PyTorch) and single-process benchmark medians on
one RTX 4070 Ti SUPER (sm_89), batch 1:

| family | verify: actions cos (mean / min) | eager | **float** | W8A8 | W4A4 | note |
|---|---|---|---|---|---|---|
| GR00T N1.6 | 0.99987 / 0.99950 | 69 | **44** | 41 | 36 | framework runtime measured 44.1 |
| GR00T N1.5 | 0.9951 / 0.981 | 54.6 | **41.2** | 37.3 | 33.1 | framework runtime measured 40.2 |
| GR00T N1.7 | see `groot_n1_7/float` | 70.0 | **41.5** | — | — | upstream full-pipeline export, `build_engines --float-onnx-dir` |
| Evo-1 | 0.99870 / 0.98081 | 174.5 | **135.9** | 123.3 | 115.6 | flash-parity attention re-implemented for the trace |
| SmolVLA | 0.99374 / 0.95629 (kv_stack 0.99995) | 214.8 | **36.8** | 39.5 | 37.0 | float and W4A4 within run-to-run noise: SmolLM2 projections are too small for the plugins to pay back |
| π₀.₅ | 0.99037 / 0.91556 (kv_stack 0.99279) | 166.5 | **111.8** | 89.9 | 76.5 | `torch.compile` reaches 100.7 here: on this family the compiled-PyTorch control is the stronger one, which is what the paper's table uses |

Two of these needed more than a plain trace, and both are recorded where they bit:

- **Evo-1** runs `flash_attention_2`, under which the model hands its layers no mask at
  all; a traced eager attention must rebuild causality itself, add the key-padding bias at
  a quarter of `finfo.min` (so a padded query's own key, both causally blocked and padded,
  stays finite), and zero each layer's attention output at padded query positions before
  `o_proj`. Missing the first of those gave actions cosine 0.79; with it, 0.9987.
- **SmolVLA / π₀.₅** run their prefix through a hand-written forward
  (`smolvlm_with_expert.py`, `paligemma_with_expert.py`) whose RoPE convention HF's
  decoder does not share: tracing `LlamaModel.forward` reproduced V exactly and K at
  cosine 0.74. The float arm therefore traces upstream's own prefix forward.

`groot_n1_6/exports/float_head/` is the earlier head-only arm (upstream's float DiT with
the LLM left in PyTorch); `results/groot_n1_6/float/verify.json` was produced by it. It
is kept because the paper's first draft cited it; the end-to-end arm is `exports/float/`.

## CUDA-graph replay (`FOLDQUANT_TRT_CUDA_GRAPH=1`) on the two KV-stack families

Same benchmark, same engines, three repeats of sixty iterations, medians in ms (denoise loop in
parentheses). The runtime's opt-in graph replay removes per-call enqueue work from the ten-step
expert loop; the paper's table is measured without it, as is every other family.

| family | eager | torch.compile | float | W8A8 | W4A4 |
|---|---|---|---|---|---|
| π₀.₅, no replay | 169.4 | 100.3 | 111.8 (34.4) | 89.9 (34.4) | 76.5 (30.8) |
| π₀.₅, replay | 166.1 | 100.4 | 111.6 (33.4) | 89.7 (32.9) | 75.0 (29.4) |
| SmolVLA, no replay | 210.0 | 38.93 | 36.8 (~20) | 38.7 (21.8) | 36.7 (20.2) |
| SmolVLA, replay | 212.5 | 38.8 | 32.3 (13.4) | 34.8 (18.2) | 32.6 (16.6) |

Two readings. On π₀.₅ replay changes nothing (≤1.5 ms on a 30 ms loop): the 3.3 ms per
expert step is the engine itself, not launch overhead, and `torch.compile` stays the stronger
floating-point control there. On SmolVLA replay is worth 5–7 ms and reverses the order inside the
loop — float 13.4 < W4A4 16.6 < W8A8 18.2 — which is the same statement as the row above:
SmolLM2's projections are too small for the INT4/INT8 plugins to pay back their own overhead.
