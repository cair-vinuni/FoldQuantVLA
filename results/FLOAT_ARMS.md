# Float-engine arms (`--llm-scheme float --dit-scheme float`)

The unquantized engine of every module, traced from the live policy under the runtime's
binding names and built weakly typed with no plugins. It is the floor of each ladder and the
compiled control the paper's latency table divides by. Since it comes off the same
`export_foldquant` → `build_engines` → `install_engines` path as W8A8 and W4A4, the arms
differ in nothing but the precision of the projections.

Held-out verify (n = 32, seeded, vs bf16 PyTorch) and single-process benchmark medians on
one RTX 4070 Ti SUPER (sm_89), batch 1:

| family | verify: actions cos (mean / min) | eager | **float** | W8A8 | W4A4 | note |
|---|---|---|---|---|---|---|
| GR00T N1.6 | 0.99987 / 0.99950 | 69 | **44** | 41 | 36 | framework runtime measured 44.1 |
| GR00T N1.5 | 0.9951 / 0.981 | 54.6 | **41.2** | 37.3 | 33.1 | framework runtime measured 40.2 |
| GR00T N1.7 | see `groot_n1_7/float` | 70.0 | **41.5** | - | - | upstream full-pipeline export, `build_engines --float-onnx-dir` |
| π₀.₅ | 0.99037 / 0.91556 (kv_stack 0.99279) | 166.5 | **111.8** | 89.9 | 76.5 | `torch.compile` reaches 100.7 here: on this family the compiled-PyTorch control is the stronger one, which is what the paper's table uses |

One of these needed more than a plain trace, recorded where it bit: **π₀.₅** runs its
prefix through a hand-written forward (`paligemma_with_expert.py`) whose RoPE convention
HF's decoder does not share. The float arm therefore traces upstream's own prefix forward.

`groot_n1_6/exports/float_head/` is the earlier head-only arm (upstream's float DiT with
the LLM left in PyTorch); `results/groot_n1_6/float/verify.json` was produced by it. It
is kept because the paper's first draft cited it; the end-to-end arm is `exports/float/`.

## CUDA-graph replay (`FOLDQUANT_TRT_CUDA_GRAPH=1`) on the KV-stack family

Same benchmark, same engines, three repeats of sixty iterations, medians in ms (denoise loop in
parentheses). The runtime's opt-in graph replay removes per-call enqueue work from the ten-step
expert loop; the paper's table is measured without it, as is every other family.

| family | eager | torch.compile | float | W8A8 | W4A4 |
|---|---|---|---|---|---|
| π₀.₅, no replay | 169.4 | 100.3 | 111.8 (34.4) | 89.9 (34.4) | 76.5 (30.8) |
| π₀.₅, replay | 166.1 | 100.4 | 111.6 (33.4) | 89.7 (32.9) | 75.0 (29.4) |

On π₀.₅ replay changes nothing (≤1.5 ms on a 30 ms loop): the 3.3 ms per expert step is the
engine itself, not launch overhead, and `torch.compile` stays the stronger floating-point control
there.
