# Evo-1: does the calibration image resolution explain the four-bit loss?

Two defects were found in the calibration of every Evo-1 closed-loop arm of the campaign: the
calibration set was the 256×256 LIBERO bundle while Evo-1 evaluates at 448×448 (its ViT input is
fixed at 448, so a 256-px frame enters upsampled — the mismatch is image detail, not tensor shape),
and the calibration manifest carried the `libero_panda` embodiment tag although the artifact
declares only `libero_robot` (the tag selects the normalization statistics used during replay). The
resolution mismatch had been pre-registered in the campaign ledger as a confound to report if the
four-bit arms fell materially below the eight-bit band; they did (648/800 and 381/800 against a
724–737 band for every other Evo-1 arm).

## Matched A/B screening (RTX 4070 Ti SUPER, TensorRT 11.2.1.2)

Arm `act_w4a4_shg_llm_w4a4_srg`, both calibrations with the correct `libero_robot` tag, the same
96 frames, 64 samples, seed 42 (identical sample indices); drift vs the bf16 PyTorch reference on
32 held-out observations (the same reference validates the float TensorRT arm at e2e 0.999985).

| calibration | frames | e2e cosine | LLM | head |
|---|---|---|---|---|
| A — native 448 px | `libero_evo1_calib` (sha256 7e5bc061…) | **0.9695** | 0.9810 | 0.9971 |
| B — same frames, 448→256→448 | `libero_evo1_calib_256up` (3b704899…) | **0.9717** | 0.9803 | 0.9972 |
| campaign arm — 256 px bundle, `libero_panda` tag (H100, P1) | `libero_4suites_calib` | 0.9720 | — | — |

Neither the image detail nor the embodiment tag moves the four-bit *fidelity*: three calibrations land
within 0.003 of each other, which makes a large calibration effect less likely. It does not clear the
pre-registered confound, whose endpoint is closed-loop success: in this cosine band the campaign's own
data show that differences of a few thousandths in cosine coexist with hundreds of episodes of
difference in success (0.9655 → 381/800 against 0.9718 → 648/800), and the screening used 64 samples
from 96 frames where the campaign arms used 128 from 53,635. The Evo-1 four-bit results are therefore
reported as confounded per the pre-registration until a closed-loop rerun with a 448-pixel calibration
exists; this screening is recorded as the evidence that the effect, if any, is not visible offline.
