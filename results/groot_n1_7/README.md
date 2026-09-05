# GR00T N1.7 — records

Four arms (`float`, `w8a8`, `w4a4`, `w4a4_cascade`) over one seeded held-out
set of 32 observations, scored under the protocol in [`../README.md`](../README.md).
`w4a4/positions.json` additionally records, per observation, the sixteen worst
prefix positions of the LLM output the action head consumes.

## Correction: prefix damage does not predict which observation flips

The commit message of `a73ebea` states that the two full gripper flips in the
W4A4 arm — samples 18 and 21 — are "the same two observations `positions.json`
shows taking the worst prefix damage". **That is not what the file says**, and
the sentence should not be quoted. Re-derived from the two committed files:

| sample | ep/step | worst-position cos | rank of 32 | positions < 0.9 | rank of 32 | action cos | worst channel |
|---|---|---|---|---|---|---|---|
| 25 | 72/154 | 0.7321 | **1** | 10 | 14 | 0.9995 | z, \|Δ\| 0.027 |
| 19 | 103/60 | 0.7481 | **2** | 13 | 4 | 0.9780 | gripper, \|Δ\| 0.291 |
| 18 | 2/28 | 0.7507 | 3 | 11 | 9 | 0.8021 | **gripper, \|Δ\| 1.000** |
| 21 | 100/107 | 0.7813 | 14 | 8 | 22 | 0.8714 | **gripper, \|Δ\| 1.000** |

The two most damaged prefixes in the set do not flip — the worst of them,
sample 25, decodes one of the cleanest chunks in the arm at 0.9995. One flip
(18) does sit near the damage tail; the other (21) sits mid-pack by worst
position and 22nd of 32 by count of positions below 0.9. Across all 32
observations the Pearson correlation between worst-position cosine and action
cosine is **+0.30** — the same direction the claim assumed, far too weak to
carry it.

What the records do support is unchanged: both flips are real, both land on
channel 6 (the gripper), both read \|Δ\| = 1.000 on this family's 0/1 gripper,
and the position-level damage is real where it is measured. What does not
follow is a per-observation link between the two depths. The position-level
finding and the action-level finding are two measurements of the same arm, not
two views of the same observations.

The pushed history is left as it is; this file is the correction of record.

## Reproducing the table

```bash
python - <<'PY'
import json
pos = json.load(open("results/groot_n1_7/w4a4/positions.json"))
ver = json.load(open("results/groot_n1_7/w4a4/verify.json"))
S, V = pos["samples"], ver["samples"]
worst = [min(w["cos"] for w in s["worst"]) for s in S]
below = [sum(1 for w in s["worst"] if w["cos"] < 0.9) for s in S]
rank = lambda vals, i, asc: sorted(range(len(S)), key=lambda k: vals[k] if asc else -vals[k]).index(i) + 1
for i in (25, 19, 18, 21):
    s, v, aw = S[i], V[i], V[i]["action_worst"]
    print(f'{i:2} ep{s["episode"]}/{s["step"]:<4} cos {worst[i]:.4f} rank {rank(worst, i, True):2}'
          f'  <0.9: {below[i]:2} rank {rank(below, i, False):2}'
          f'  action {v["action_cos"]:.4f}  {aw["label"]} |d|={aw["max_abs"]:.3f}')
PY
```
