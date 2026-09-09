# Does the PyTorch emulation track the W4A4 engine on the KV-stack families?

`validate_emulation.py` (SmolVLA, π₀.₅) compares the emulated prefix `kv_stack` — the fold, the
rotation, GPTQ and the fake-quantized projections run in PyTorch — against the W4A4 cascade
engine, layer by layer, with the float PyTorch model as the control for what *no* emulation gives.
8 held-out observations; the figure of merit is how much of the float→engine gap the emulation
closes.

| family | LLM | calibration | emulated-vs-engine | float-vs-engine (control) | gap closed |
|---|---|---|---|---|---|
| SmolVLA | SmolLM2, 32 layers | 128 samples (= engine) | 0.98718 (worst layer 30 V, min 0.94463) | 0.98489 | **15.1 %** |
| π₀.₅ | Gemma-2B, 18 layers | **32 samples** (engine: 128) | 0.92179 (worst layer 13 V, min 0.76084) | 0.94763 | **−49.3 %** |

Reading. On SmolLM2 the emulation moves the right way but closes only a seventh of the gap:
most of what separates the engine from the float model is not reproduced by fake-quantizing the
projections, i.e. it lives in the plugin's rounding and accumulation path rather than in the
weights. On Gemma the run is not a matched control — the emulation was calibrated on 32
observations to fit beside the engine and the model on a 16 GB card (the fp64 Cholesky still fell
back to CPU), against the engine's 128 — so the emulated GPTQ rounding differs from the engine's
by construction and the negative number cannot be read as "emulation hurts". What the two runs do
establish together is that the SmolLM2 gate (`smollm_llama`, plain-RMSNorm path) is not an
outlier: on neither KV-stack family should a cascade calibration be assumed to see the engine it
calibrates for. The cascade arms' success rates are unaffected — they measure the engines as
built — but the emulation is a weak proxy here, and a matched 128-sample Gemma control needs a
card with room for engine, model and Hessians at once (or the CPU path, hours per build).
