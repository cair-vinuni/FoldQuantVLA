# UR10e

A Universal Robots UR10e: 6-DoF e-series arm, 10 kg payload, 1300 mm reach,
controlled over RTDE. State and action are **7-dimensional** — six joints plus
a gripper.

Unlike [ALOHA](aloha.md), **no family in this repository ships a UR client, and
none ships a UR checkpoint.** Both have to be produced. Read
[`README.md`](README.md) first; the embodiment rule there is the whole
difficulty of this page.

## What exists to build on

| | status |
|---|---|
| UR checkpoint | none in any family |
| UR client | none in any family |
| openpi UR5 fine-tuning recipe | `models/pi05/examples/ur5/README.md` — input/output transforms, data config, train config |
| UR normalization statistics | `pi0_base` ships an `ur5e` asset (`asset_id="ur5e"`), reusable when fine-tuning |
| nearest in-tree client | `models/evo_1/Evo_1/scripts/Evo1_client_xarm6.py` — a 6-DoF arm plus gripper, same shape as a UR |
| this guide's client | [`ur10e_client.py`](ur10e_client.py) — written for RTDE, **not run against hardware** |

π₀.₅ is the family to use: it is the only one with a documented UR recipe and
UR normalization statistics, and its websocket server is what the client here
speaks.

**UR5e versus UR10e.** openpi's recipe and assets are for the UR5e. The two
share the e-series kinematic structure and the RTDE interface, so the recipe
transfers, but they do not share link lengths, payload or reachable workspace.
Reused `ur5e` normalization statistics are a starting point for fine-tuning,
not a substitute for statistics computed on your own UR10e data — and a policy
fine-tuned on UR5e data will not transfer to a UR10e unchanged.

## 1. Collect data

Record teleoperated demonstrations in LeRobot format with, per frame:

- `joints` — 6 joint positions, radians, `RTDEReceiveInterface.getActualQ()`
- `gripper` — 1 value in a fixed convention (this guide's client assumes 0
  open, 1 closed)
- `base_rgb`, `wrist_rgb` — two camera streams
- `prompt` — the task instruction

Match this to the transform you will train under. openpi's UR5 `UR5Inputs`
concatenates `joints` and `gripper` into the state vector and maps the two
cameras onto `base_0_rgb` and `left_wrist_0_rgb`, leaving `right_wrist_0_rgb`
zeroed and masked off — so a two-camera UR rig fits the three-camera model
slot without retraining the architecture.

## 2. Fine-tune

Follow `models/pi05/examples/ur5/README.md`, which gives `UR5Inputs`,
`UR5Outputs` (`actions[:, :7]`), the `LeRobotUR5DataConfig` and a `TrainConfig`
verbatim. Two details from that file matter downstream:

- `delta_action_mask = make_bool_mask(6, -1)` — the six joints are trained as
  **delta** actions and the gripper as absolute. The client must apply them the
  same way, so pass `--action-space delta`; it defaults to `absolute` and
  getting this wrong turns a small correction into an absolute joint target.
- `assets=AssetsConfig(assets_dir=..., asset_id="ur5e")` reuses the base
  model's UR statistics.

Name the config something you will remember: it is what `--config` must be at
every later step.

## 3. Quantize and verify

From `models/pi05`, calibrating on **your UR10e data**:

```bash
python -m foldquant_integration.export_foldquant \
    --checkpoint-dir <ur10e-ckpt> --config <your-config> \
    --dataset-path <your ur10e lerobot dataset> \
    --llm-scheme w8a8_sr --expert-scheme w8a8_sh \
    --output-dir exports/ur10e_w8a8
python -m foldquant_integration.build_engines \
    --onnx-dir exports/ur10e_w8a8/onnx --engine-dir exports/ur10e_w8a8/engines
python -m foldquant_integration.verify \
    --checkpoint-dir <ur10e-ckpt> --config <your-config> \
    --dataset-path <your ur10e lerobot dataset> \
    --engine-dir exports/ur10e_w8a8/engines
```

W8A8 first. On the six families measured here it stays within a few 1e-4 of
bf16; W4A4 is the arm that carries a tail, and a 10 kg payload on a 1.3 m reach
is the wrong place to meet one. Move to W4A4 only after W8A8 has run the task
and only after reading W4A4's **min** cosine and worst channel on your own
held-out data.

## 4. Serve

```bash
python -m foldquant_integration.serve \
    --checkpoint-dir <ur10e-ckpt> --config <your-config> \
    --engine-dir exports/ur10e_w8a8/engines --port 8000
```

Omit `--engine-dir` for the bf16 reference. Run the same task both ways before
concluding anything about the quantized arm.

## 5. Drive the robot

[`ur10e_client.py`](ur10e_client.py) on the robot host:

```bash
pip install ur-rtde opencv-python numpy
pip install -e <openpi>/packages/openpi-client

python ur10e_client.py --robot-ip 192.168.1.10 --host <inference-host> \
    --prompt "pick up the red block" --dry-run
```

**The first run must be `--dry-run`**: it reads the arm and the cameras, queries
the policy, applies the clamps and logs what it *would* command without sending
motion. That is how you find out what the policy emits in your cell before the
arm can act on it.

The client is not validated against hardware — no UR was available here — so
treat it as a template whose safety envelope you must set:

| constant | default | set it to |
|---|---|---|
| `MAX_JOINT_STEP` | 0.03 rad/tick | your cell's tolerance; at 100 Hz this is ~3 rad/s per joint |
| `JOINT_LIMITS` | full controller range | your reachable box — the default constrains nothing |
| `CONTROL_HZ` | 100 | your `servoJ` rate |
| `GRIPPER_RANGE` | (0, 1) | your gripper's units and convention |
| `--action-horizon` | 10 | how long the arm may run open loop per chunk |
| `--action-space` | `absolute` | `delta` if you trained under openpi's UR5 recipe |

What it does with a chunk: rejects it outright if it contains NaN/inf or has
the wrong width, then rate-limits every step so no single action can be
executed quickly and the steps after it cannot escape the limit either. Joint
limits are applied after the rate clamp, per step.

`Arm.gripper()` and `Arm.set_gripper()` are deliberately unwired — gripper
plumbing is cell-specific (Robotiq over the UR controller, a digital output, a
separate socket), and a wrong guess there is a closing gripper. Wire them
before the gripper channel means anything.

## If you use a different family

- **GR00T** — fine-tune under `new_embodiment` with a modality config for the
  7-dim state and your cameras, serve with `--embodiment-tag new_embodiment`,
  and write the client against `PolicyClient`; `gr00t/eval/real_robot/SO100/`
  is the template and is also a 6-DoF-plus-gripper arm.
- **Evo-1** — `Evo1_client_xarm6.py` is structurally what a UR client is:
  6 joints plus gripper padded to a 24-dim state, `image_mask [1, 1, 0]` for a
  two-camera rig, `chunk[:25]` executed with `act[:6]` as joints and `act[6]`
  as the gripper. Swap `XArmAPI` for RTDE and mind the units — that client
  sends **degrees**, RTDE speaks radians.
