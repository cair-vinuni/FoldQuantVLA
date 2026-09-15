# Does the PyTorch emulation track the W4A4 engine on the KV-stack family?

`validate_emulation.py` (π₀.₅) compares the emulated prefix `kv_stack` — the fold, the
rotation, GPTQ and the fake-quantized projections run in PyTorch — against the W4A4 cascade
engine, layer by layer, with the float PyTorch model as the control for what *no* emulation gives.
8 held-out observations; the figure of merit is how much of the float→engine gap the emulation
closes.

| family | LLM | calibration | emulated-vs-engine | float-vs-engine (control) | gap closed |
|---|---|---|---|---|---|
| π₀.₅ | Gemma-2B, 18 layers | **32 samples** (engine: 128) | 0.92179 (worst layer 13 V, min 0.76084) | 0.94763 | **−49.3 %** |

Reading. The run is not a matched control — the emulation was calibrated on 32 observations to
fit beside the engine and the model on a 16 GB card (the fp64 Cholesky still fell back to CPU),
against the engine's 128 — so the emulated GPTQ rounding differs from the engine's by
construction and the negative number cannot be read as "emulation hurts". What it does establish
is that a cascade calibration on this family should not be assumed to see the engine it
calibrates for. The cascade arm's success rate is unaffected — it measures the engines as
built — but the emulation is a weak proxy here, and a matched 128-sample Gemma control needs a
card with room for engine, model and Hessians at once (or the CPU path, hours per build).
