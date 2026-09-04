# RoboTwin Evaluation for Evo-1

Full setup and run instructions live in the top-level
[README → 🧪 RoboTwin Benchmark](../README.md#-robotwin-benchmark).

This directory ships the Evo-1 **policy plugin** for RoboTwin:

```
policy/Evo1/
  deploy_policy.py   # RoboTwin policy interface: obs -> server -> smoothed 14-D action chunk
  deploy_policy.yml  # default config (horizon=37)
  eval.sh            # single-task launcher
```

Install it into a RoboTwin checkout:

```bash
cp -r policy/Evo1  /path/to/RoboTwin/policy/Evo1
```

The Evo-1 server reads `arm_key` / `dataset_key` from each client request, so **no server-side
edit is needed for RoboTwin** — the client sends `arm_key=aloha_joint` and the per-task
`dataset_key=robotwin_<task>` (RoboTwin `norm_stats.json` is keyed per task, 50 keys under
`aloha_joint`).

For checkpoints trained on
[StarVLA/RoboTwin-Randomized](https://huggingface.co/datasets/StarVLA/RoboTwin-Randomized),
`norm_stats.json` is keyed per task **and per data setting** (`robotwin_<task>_clean` /
`robotwin_<task>_rand`, 100 keys). Pass `auto` as the 8th argument of `eval.sh`
(`dataset_key_suffix`) and the client appends the suffix matching the task config
(`demo_clean` → `_clean`, `demo_randomized` → `_rand`).

## Inference Settings

The policy is served with:

1. **`horizon=37`** — actions executed per inference call (default in `eval.sh` / `deploy_policy.yml`).
2. **`num_inference_timesteps=50`** in `Evo_1/scripts/Evo1_server.py`.
3. **Gaussian action smoothing, kernel=9** — in `deploy_policy.py`.

## Results — RoboTwin-50 (clean-only training)

A **single multi-task policy trained on clean data only** — 50 RoboTwin tasks,
**50 `demo_clean` demonstrations per task** (no randomized / augmented data). At **test time**
each task is rolled out **100 times per setting** (50 tasks × 100 = 5000 evaluation episodes
per setting), `horizon=37`, using the released
[`MINT-SJTU/Evo1_RoboTwin2_clean`](https://huggingface.co/MINT-SJTU/Evo1_RoboTwin2_clean) checkpoint.

| Setting | Overall success rate |
|---|---|
| `demo_clean` | **65.7%** (3287 / 5000) |
| `demo_randomized` (zero-shot) | **12.5%** (623 / 5000) |

The `demo_randomized` number is **zero-shot domain generalization** — the policy never saw
randomized scenes (messy table, random background / lighting, table-height jitter) during
training. See the RoboTwin-550 section below for the effect of training on randomized data.

Per-task success rates (sorted by `demo_clean`):

| Task | clean | rand | | Task | clean | rand |
|---|---|---|---|---|---|---|
| adjust_bottle | 100% | 40% | | place_phone_stand | 73% | 3% |
| handover_mic | 100% | 2% | | place_object_stand | 72% | 11% |
| stack_blocks_two | 100% | 6% | | open_microwave | 70% | 4% |
| click_alarmclock | 98% | 18% | | place_cans_plasticbox | 70% | 13% |
| grab_roller | 98% | 17% | | move_can_pot | 69% | 3% |
| shake_bottle_horizontally | 98% | 28% | | put_bottles_dustbin | 69% | 45% |
| place_container_plate | 96% | 7% | | place_bread_basket | 63% | 19% |
| click_bell | 95% | 36% | | place_bread_skillet | 63% | 9% |
| dump_bin_bigbin | 95% | 46% | | blocks_ranking_size | 58% | 7% |
| stack_bowls_two | 93% | 30% | | place_can_basket | 50% | 4% |
| shake_bottle | 91% | 27% | | pick_diverse_bottles | 49% | 26% |
| place_burger_fries | 90% | 23% | | place_object_scale | 49% | 1% |
| stack_blocks_three | 87% | 3% | | place_a2b_left | 48% | 5% |
| stack_bowls_three | 85% | 14% | | put_object_cabinet | 39% | 10% |
| press_stapler | 83% | 42% | | place_a2b_right | 38% | 3% |
| beat_block_hammer | 82% | 8% | | place_fan | 34% | 0% |
| open_laptop | 82% | 2% | | place_shoe | 33% | 4% |
| lift_pot | 81% | 7% | | rotate_qrcode | 32% | 1% |
| move_playingcard_away | 81% | 1% | | scan_object | 32% | 0% |
| place_object_basket | 79% | 6% | | stamp_seal | 28% | 2% |
| place_empty_cup | 77% | 17% | | turn_switch | 28% | 8% |
| blocks_ranking_rgb | 76% | 21% | | place_mouse_pad | 17% | 0% |
| pick_dual_bottles | 76% | 40% | | hanging_mug | 9% | 1% |
| move_pillbottle_pad | 75% | 0% | | place_dual_shoes | 2% | 0% |
| handover_block | 74% | 3% | | move_stapler_pad | 0% | 0% |

## Results — RoboTwin-550 randomized training (`demo_clean` + `demo_randomized`)

A **single multi-task policy trained on randomized data** — the public
[StarVLA/RoboTwin-Randomized](https://huggingface.co/datasets/StarVLA/RoboTwin-Randomized)
dataset: the same 50 RoboTwin tasks, **550 demonstrations per task** (50 `demo_clean` +
500 `demo_randomized`), 27,500 demonstrations in total. At **test time** each task is rolled
out **100 times per setting** (2 × 50 × 100 = 10,000 evaluation episodes), `horizon=37`,
`dataset_key_suffix=auto`. Checkpoint release in preparation.

| Setting | Overall success rate |
|---|---|
| `demo_clean` | **85.0%** (4251 / 5000) |
| `demo_randomized` | **83.2%** (4161 / 5000) |

Per-task success rates (sorted by `demo_clean`):

| Task | clean | rand | | Task | clean | rand |
|---|---|---|---|---|---|---|
| adjust_bottle | 100% | 97% | | move_playingcard_away | 91% | 98% |
| click_alarmclock | 100% | 100% | | stack_bowls_two | 91% | 96% |
| click_bell | 100% | 100% | | move_can_pot | 89% | 77% |
| lift_pot | 100% | 100% | | place_bread_skillet | 88% | 90% |
| place_container_plate | 100% | 99% | | place_shoe | 87% | 98% |
| place_empty_cup | 100% | 99% | | put_bottles_dustbin | 87% | 34% |
| stack_blocks_two | 100% | 99% | | place_fan | 86% | 90% |
| place_burger_fries | 99% | 97% | | open_microwave | 85% | 91% |
| place_cans_plasticbox | 99% | 98% | | stack_bowls_three | 79% | 70% |
| blocks_ranking_rgb | 98% | 97% | | pick_dual_bottles | 78% | 45% |
| grab_roller | 98% | 97% | | place_object_basket | 78% | 72% |
| move_pillbottle_pad | 98% | 96% | | rotate_qrcode | 78% | 83% |
| open_laptop | 98% | 100% | | scan_object | 78% | 56% |
| place_object_stand | 98% | 92% | | place_a2b_left | 76% | 68% |
| shake_bottle | 98% | 100% | | place_can_basket | 75% | 56% |
| shake_bottle_horizontally | 98% | 100% | | handover_block | 71% | 61% |
| stack_blocks_three | 97% | 95% | | place_mouse_pad | 66% | 27% |
| place_bread_basket | 96% | 94% | | pick_diverse_bottles | 65% | 36% |
| press_stapler | 96% | 96% | | place_a2b_right | 62% | 68% |
| blocks_ranking_size | 95% | 94% | | stamp_seal | 61% | 78% |
| beat_block_hammer | 94% | 93% | | place_dual_shoes | 59% | 74% |
| handover_mic | 94% | 99% | | hanging_mug | 56% | 56% |
| place_object_scale | 93% | 96% | | move_stapler_pad | 47% | 59% |
| dump_bin_bigbin | 92% | 96% | | turn_switch | 45% | 57% |
| place_phone_stand | 92% | 99% | | put_object_cabinet | 40% | 88% |

## Demos

Inference examples:

<table>
  <tr>
    <td width="50%"><video src="https://github.com/user-attachments/assets/8c4bac58-93fe-4187-bcf0-700c0a1a46b3" controls muted width="100%"></video></td>
    <td width="50%"><video src="https://github.com/user-attachments/assets/526c0599-feaf-4c8c-b3f2-6fe710fcdd6f" controls muted width="100%"></video></td>
  </tr>
</table>


