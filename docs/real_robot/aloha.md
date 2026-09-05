# ALOHA

Bimanual [ALOHA](https://github.com/tonyzhaozh/aloha): two ViperX 300s follower
arms, two leader arms for teleoperation, four cameras, driven through Interbotix
under ROS 1 Noetic. State and action are **14-dimensional** —
`[left 6 joints, left gripper, right 6 joints, right gripper]`.

Two of the six families ship a real ALOHA client upstream, so both can drive
one through a FoldQuant server with no client change. Read
[`README.md`](README.md) first — in particular that a LIBERO checkpoint will
not drive an ALOHA no matter how well it quantizes.

## Which family

| family | ALOHA support upstream | what you need |
|---|---|---|
| π₀.₅ | `examples/aloha_real/` — full client, Docker, data converter | an ALOHA checkpoint (`pi05_aloha`, or fine-tune) |
| Evo-1 | `scripts/Evo1_client_aloha.py` + `scripts/aloha/` | an ALOHA checkpoint |
| GR00T N1.7/N1.6/N1.5 | none — no ALOHA embodiment tag | fine-tune under `new_embodiment`, write a client from the SO100 evaluator |
| SmolVLA | LeRobot v0.6.1 ships no ALOHA robot class, and this integration has no server | both |

π₀.₅ is the shortest path: openpi's ALOHA support is first-class, the
normalization statistics for the embodiment ship with the base checkpoint, and
its client already brokers action chunks.

## π₀.₅ on ALOHA

### 1. Checkpoint

Use an ALOHA-trained checkpoint. Upstream's configs are `pi05_aloha`,
`pi0_aloha`, and task-specific ones (`pi0_aloha_towel`,
`pi0_aloha_tupperware`, `pi05_aloha_pen_uncap`); to train your own, convert
demonstrations with `examples/aloha_real/convert_aloha_data_to_lerobot.py` and
follow that example's README. The config name you train under is the one you
must serve under.

### 2. Quantize it

From `models/pi05`, in that family's environment, calibrating on **ALOHA**
data — not LIBERO:

```bash
python -m foldquant_integration.export_foldquant \
    --checkpoint-dir <aloha-ckpt> --config pi05_aloha \
    --dataset-path <your aloha lerobot dataset> \
    --llm-scheme w8a8_sr --expert-scheme w8a8_sh \
    --output-dir exports/aloha_w8a8
python -m foldquant_integration.build_engines \
    --onnx-dir exports/aloha_w8a8/onnx --engine-dir exports/aloha_w8a8/engines
python -m foldquant_integration.verify \
    --checkpoint-dir <aloha-ckpt> --config pi05_aloha \
    --dataset-path <your aloha lerobot dataset> \
    --engine-dir exports/aloha_w8a8/engines
```

Read `verify`'s **min** action cosine and its worst channel before going
further. On a 14-dim bimanual action a saturated gripper is one channel out of
fourteen and will not move the mean much.

### 3. Serve

```bash
python -m foldquant_integration.serve \
    --checkpoint-dir <aloha-ckpt> --config pi05_aloha \
    --engine-dir exports/aloha_w8a8/engines --port 8000
```

Omit `--engine-dir` for the bf16 reference arm. The server publishes
`policy.metadata`, which is where the client reads `reset_pose`, so the
metadata handshake behaves as upstream's.

### 4. Run the robot

Upstream's client, unchanged, in its own environment on the robot host:

```bash
# robot host, terminal 1 — ROS nodes
roslaunch aloha ros_nodes.launch

# robot host, terminal 2 — upstream's ALOHA runtime
python -m examples.aloha_real.main --host <inference-host> --port 8000
```

`examples/aloha_real/main.py` takes `--action_horizon` (25 by default) and
brokers the chunk through `ActionChunkBroker`. That horizon is how long the
arm executes open loop; shorten it before the first runs.

Observation contract the client already satisfies: `state` `[14]`, images
`cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` as CHW uint8,
resized with padding. Actions come back `[horizon, 14]`.

## Evo-1 on ALOHA

Upstream's `scripts/README_Aloha.md` documents the hardware bring-up; the only
change is the address the client connects to.

```bash
# inference host
python -m foldquant_integration.serve \
    --checkpoint-dir <aloha-ckpt> --engine-dir exports/aloha_w8a8/engines --port 9000
```

The server resolves the normalizer keys from the checkpoint's `norm_stats.json`
on its own, so an ALOHA checkpoint needs no `--arm-key` / `--dataset-key`
unless the file holds more than one arm — in which case it lists the choices
and stops rather than guessing.

Then edit `IP` and `PORT` at the top of `scripts/Evo1_client_aloha.py` to point
at the inference host and run it. Its contract, for reference when adapting:

| field | value |
|---|---|
| `image` | `[cam_high, cam_left_wrist, cam_right_wrist]`, HWC uint8 lists |
| `image_mask` | `[1, 1, 1]` |
| `state` | raw state zero-padded to **24** |
| `action_mask` | `[[1]*len(raw_state) + [0]*pad]` |
| `prompt` | the task instruction string |
| response | action chunk; the client executes `chunk[:25]`, taking `act[:14]` |

Note Evo-1's client sends three cameras and π₀.₅'s four — `cam_low` is used by
one and not the other. Each family's client matches the data its checkpoint was
trained on; do not cross them.

## GR00T on ALOHA

No GR00T release declares an ALOHA embodiment tag — N1.7's are DROID, XDOF,
G1, R1-Pro, SimplerEnv, LIBERO and `new_embodiment`. So ALOHA means
fine-tuning under `new_embodiment` with a modality config describing the 14-dim
state and the four cameras (`getting_started/finetune_new_embodiment.md` in
N1.6, `3_0_new_embodiment_finetuning.md` in N1.5), then quantizing and serving
that checkpoint with `--embodiment-tag new_embodiment`.

The client has to be written; `gr00t/eval/real_robot/SO100/eval_so100.py` is
the template and shows the three things that change per robot — packing camera
frames into `obs["video"]`, building `obs["state"]` under the modality keys the
checkpoint declares, and decoding the returned chunk back to joint commands.
For ALOHA the state would carry the two arms and two grippers rather than
SO100's `single_arm` / `gripper` pair.
