#!/usr/bin/env python3
# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.
#
# Reference client: a UR10e driven by a FoldQuant policy server.
#
# ─────────────────────────────────────────────────────────────────────────────
# THIS FILE HAS NOT BEEN RUN AGAINST A UR10e. No Universal Robots hardware was
# available to the authors. It is a template with the safety envelope written
# in, not a validated controller. Read every clamp below and set it for YOUR
# cell before the arm is powered, and make the first run --dry-run.
# ─────────────────────────────────────────────────────────────────────────────
#
# No family in this repository ships a UR client, so unlike ALOHA there is no
# upstream loop to point at a server. The protocol is openpi's websocket, so
# this runs against models/pi05's `foldquant_integration.serve`; the
# observation keys are the ones openpi's UR5 recipe defines
# (models/pi05/examples/ur5/README.md) and must match the transform the
# checkpoint was trained under.
#
# Runs on the ROBOT host, in its own environment:
#   pip install ur-rtde opencv-python numpy
#   pip install -e <openpi>/packages/openpi-client
#
#   python ur10e_client.py --robot-ip 192.168.1.10 --host <inference-host> \
#       --prompt "pick up the red block" --dry-run

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import numpy as np

log = logging.getLogger("ur10e")

# ── Safety envelope. These are deliberately conservative; raise them only after
# ── a dry run has shown you what the policy actually emits in your cell.
MAX_JOINT_STEP = 0.03
"""Radians a joint may move in one control tick. At 100 Hz this caps each joint
near 3 rad/s; a chunk that asks for more is clamped, not executed as asked."""

JOINT_LIMITS = [(-2 * np.pi, 2 * np.pi)] * 6
"""Per-joint (low, high) in radians. Replace with YOUR cell's reachable box —
the default is the controller's full range and constrains nothing."""

CONTROL_HZ = 100.0
SERVO_LOOKAHEAD = 0.1
SERVO_GAIN = 300
GRIPPER_RANGE = (0.0, 1.0)
"""Policy gripper convention: 0 open, 1 closed. Map to your gripper's units."""


class Stopped(Exception):
    pass


def _clamp_chunk(current: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Rate- and range-limit a chunk of joint targets, starting from ``current``.

    Each step may move a joint at most ``MAX_JOINT_STEP`` from where the
    previous step left it, so a single wild action cannot be executed quickly
    and cannot be escaped by the steps that follow it.
    """
    out = np.empty_like(targets)
    q = current.copy()
    for i, target in enumerate(targets):
        step = np.clip(target - q, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        q = q + step
        for j, (low, high) in enumerate(JOINT_LIMITS):
            q[j] = float(np.clip(q[j], low, high))
        out[i] = q
    return out


def _validate(actions: np.ndarray, dof: int) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[1] < dof + 1:
        raise Stopped(f"action chunk has shape {actions.shape}, expected [horizon, >={dof + 1}]")
    if not np.all(np.isfinite(actions)):
        raise Stopped("action chunk contains NaN or inf — refusing to execute")
    return actions


class Cameras:
    """Two OpenCV cameras. Replace with RealSense SDK calls if that is your rig."""

    def __init__(self, base: str, wrist: str, size: int = 224) -> None:
        import cv2

        self._cv2 = cv2
        self._size = size
        self._base = cv2.VideoCapture(base if not base.isdigit() else int(base))
        self._wrist = cv2.VideoCapture(wrist if not wrist.isdigit() else int(wrist))
        for name, cap in (("base", self._base), ("wrist", self._wrist)):
            if not cap.isOpened():
                raise Stopped(f"{name} camera did not open")

    def _grab(self, cap) -> np.ndarray:
        ok, frame = cap.read()
        if not ok:
            raise Stopped("camera read failed")
        frame = self._cv2.resize(frame, (self._size, self._size))
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB).astype(np.uint8)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        return self._grab(self._base), self._grab(self._wrist)

    def close(self) -> None:
        self._base.release()
        self._wrist.release()


class Arm:
    """UR10e through RTDE, plus whatever drives the gripper."""

    def __init__(self, ip: str, dry_run: bool) -> None:
        self.dry_run = dry_run
        import rtde_control
        import rtde_receive

        self.recv = rtde_receive.RTDEReceiveInterface(ip)
        self.ctrl = None if dry_run else rtde_control.RTDEControlInterface(ip)
        log.info("connected to %s%s", ip, " (dry run: no motion commands)" if dry_run else "")

    def joints(self) -> np.ndarray:
        return np.asarray(self.recv.getActualQ(), dtype=np.float32)

    def gripper(self) -> float:
        # Robotiq over the UR controller, a digital output, or a separate socket —
        # cell-specific. Returning a constant keeps the state vector well-formed
        # while making it obvious this is unwired.
        return 0.0

    def servo(self, q: np.ndarray, dt: float) -> None:
        if self.dry_run:
            log.info("dry run q=%s", np.round(q, 4).tolist())
            return
        self.ctrl.servoJ(q.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)

    def set_gripper(self, value: float) -> None:
        value = float(np.clip(value, *GRIPPER_RANGE))
        if self.dry_run:
            log.info("dry run gripper=%.3f", value)
            return
        # Wire to your gripper driver here.

    def stop(self) -> None:
        if self.ctrl is not None:
            self.ctrl.servoStop()
            self.ctrl.stopScript()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ip", required=True)
    ap.add_argument("--host", default="127.0.0.1", help="inference host running foldquant_integration.serve")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--base-camera", default="0")
    ap.add_argument("--wrist-camera", default="1")
    ap.add_argument("--action-horizon", type=int, default=10, help="steps executed per chunk, open loop")
    ap.add_argument(
        "--action-space",
        choices=("absolute", "delta"),
        default="absolute",
        help="how the checkpoint was trained to emit joints; openpi's UR5 recipe uses delta",
    )
    ap.add_argument("--max-chunks", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true", help="read the robot and the policy, command no motion")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from openpi_client import websocket_client_policy

    arm = Arm(args.robot_ip, args.dry_run)
    cams = Cameras(args.base_camera, args.wrist_camera)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    log.info("server metadata: %s", client.get_server_metadata())

    running = True

    def _sigint(_sig, _frm):
        nonlocal running
        running = False
        log.warning("interrupt — stopping after this step")

    signal.signal(signal.SIGINT, _sigint)

    dt = 1.0 / CONTROL_HZ
    try:
        for chunk in range(args.max_chunks):
            if not running:
                break
            base_rgb, wrist_rgb = cams.read()
            q = arm.joints()
            obs = {
                "joints": q,
                "gripper": np.asarray([arm.gripper()], dtype=np.float32),
                "base_rgb": base_rgb,
                "wrist_rgb": wrist_rgb,
                "prompt": args.prompt,
            }
            actions = _validate(np.asarray(client.infer(obs)["actions"], dtype=np.float32), dof=6)
            horizon = min(args.action_horizon, len(actions))
            joints = actions[:horizon, :6]
            if args.action_space == "delta":
                # openpi's UR5 recipe trains the six joints as deltas and the
                # gripper as absolute (make_bool_mask(6, -1)); deltas accumulate
                # from the position the chunk was observed at.
                joints = q + np.cumsum(joints, axis=0)
            targets = _clamp_chunk(q, joints)
            log.info(
                "chunk %d: %d steps, max clamped joint move %.4f rad",
                chunk,
                horizon,
                float(np.abs(np.diff(np.vstack([q, targets]), axis=0)).max()),
            )
            for step in range(horizon):
                if not running:
                    break
                start = time.perf_counter()
                arm.servo(targets[step], dt)
                arm.set_gripper(float(actions[step, 6]))
                time.sleep(max(0.0, dt - (time.perf_counter() - start)))
    except Stopped as exc:
        log.error("stopped: %s", exc)
        return 1
    finally:
        arm.stop()
        cams.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
