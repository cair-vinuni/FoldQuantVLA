# Deploying a FoldQuant arm on a real robot

A FoldQuant arm is served by the **upstream release's own policy server**, with
the quantized engines installed into the policy before the server starts. The
wire protocol, the observation and action dictionaries and the client are
upstream's, unchanged — so a robot loop that already talks to the bf16 policy
talks to a W4A4 engine by changing a host and a port and nothing else.

```
    robot host                              inference host (GPU)
 ┌──────────────────┐                   ┌───────────────────────────┐
 │ cameras, arm,    │   observation     │ foldquant_integration.    │
 │ gripper, control │ ────────────────► │   serve                   │
 │ loop             │                   │   ├ upstream policy       │
 │                  │ ◄──────────────── │   └ FoldQuant engines     │
 └──────────────────┘   action chunk    └───────────────────────────┘
   upstream's client,      ZMQ or ws        upstream's server,
   unchanged                                + install_engines
```

The two hosts may be the same machine. Nothing in the protocol knows the
policy is quantized.

## The rule that decides whether any of this works

**FoldQuant changes the arithmetic, not the policy.** An engine built from a
LIBERO checkpoint emits LIBERO Panda actions in LIBERO's normalization,
whatever robot is listening. Quantizing does not port a policy to a new
embodiment, and pointing a LIBERO arm at an ALOHA or a UR10e produces
confident, well-formed, meaningless motion — which on real hardware is worse
than an obvious failure.

So the deployment order is fixed:

1. **Get a checkpoint for *your* robot.** Fine-tune the upstream release on
   data from that embodiment — each family documents this
   ([GR00T](../../models/groot_n1_7/getting_started/),
   [openpi](../../models/pi05/examples/ur5/README.md),
   [Evo-1](../../models/evo_1/Evo_1/dataset/data_preparation.md)). None of the
   checkpoints this repository quantizes for the paper is such a checkpoint;
   they are all LIBERO.
2. **Quantize that checkpoint**, with calibration drawn from that robot's data
   — `export_foldquant` → `build_engines`, per the family's integration README.
3. **Verify before hardware.** `verify` scores the engine against the bf16
   policy on held-out observations from your data. A W4A4 arm that has not
   been verified on your embodiment has not been shown to do anything.
4. **Serve, then point the robot client at it.**

Step 3 is not a formality. On the paper's LIBERO checkpoints, W4A4 drift
ranges from a median action cosine of 0.9994 (π₀.₅) to 0.9297 with a saturated
gripper in the typical observation (SmolVLA) — the same scheme, the same
kernels, six different answers. Your embodiment is a seventh.

## Servers

Each is `python -m foldquant_integration.serve` from that family's directory,
in that family's environment. Omit `--engine-dir` to serve the bf16 policy —
the reference arm every quantized number should be read against.

| family | transport | default port | server flags |
|---|---|---|---|
| GR00T N1.7 | ZMQ | 5555 | `--model-path` `--embodiment-tag` `--engine-dir` `--mode` `--host` |
| GR00T N1.6 | ZMQ | 5555 | `--model-path` `--embodiment-tag` `--engine-dir` `--host` |
| GR00T N1.5 | ZMQ | 5555 | `--model-path` `--embodiment-tag` `--engine-dir` `--data-config` `--denoising-steps` `--api-token` |
| π₀.₅ | websocket | 8000 | `--checkpoint-dir` `--config` `--engine-dir` |
| Evo-1 | websocket (JSON) | 9000 | `--checkpoint-dir` `--engine-dir` `--arm-key` `--dataset-key` |
| SmolVLA | gRPC | 8080 | `--engine-dir` `--host` `--port` `--fps` — see below |

`--embodiment-tag` is required wherever the checkpoint declares more than one
(N1.7's release declares nine and refuses to guess). π₀.₅'s `--config` is the
upstream training config name and must be the one the checkpoint was trained
under — `pi05_libero` is only the default because that is what the paper
measures.

**SmolVLA's server installs its engines later than the other five**, and the
difference is worth knowing before you point a robot at it. Upstream's async
policy server (`src/lerobot/async_inference/policy_server.py`, gRPC) builds the
policy **lazily**: the server starts with no model, and the checkpoint is named
by the *client* in its `RemotePolicyConfig` handshake. There is therefore no
assembled policy to install engines into before the server starts, so
`foldquant_integration.serve` subclasses upstream's server and installs them
inside the handshake instead, after upstream's own method has built the policy
and before the first observation can arrive.

Two consequences. The server binds its port without touching the GPU, so a
listening SmolVLA server proves nothing about the engines yet — the first
client connection is when they load. And nothing can check that the engine
directory matches the checkpoint the client asks for: the engines carry shapes,
the checkpoint carries weights, and a mismatch shows up as wrong actions rather
than as an error.

    # inference host
    python -m foldquant_integration.serve --engine-dir exports/w8a8/engines --port 8080

    # robot host — upstream's client, unchanged
    python -m lerobot.async_inference.robot_client \
        --server_address=<host>:8080 --policy_type=smolvla \
        --pretrained_name_or_path=<ckpt> ...

## Clients

Every family's client is upstream's:

| family | client | real-robot example in-tree |
|---|---|---|
| GR00T | `gr00t.policy.server_client.PolicyClient` | `gr00t/eval/real_robot/SO100/eval_so100.py` |
| π₀.₅ | `openpi_client.websocket_client_policy.WebsocketClientPolicy` | `examples/aloha_real/main.py` |
| Evo-1 | plain `websockets` + JSON | `scripts/Evo1_client_aloha.py`, `scripts/Evo1_client_xarm6.py` |

Per-robot guides:

- **[ALOHA](aloha.md)** — bimanual ViperX 300s. Upstream clients exist in two
  families; this is the shortest path to a real-robot FoldQuant arm.
- **[UR10e](ur10e.md)** — 6-DoF Universal Robots e-series. No upstream client
  in any family; the guide covers the fine-tune and ships a reference client.

## Safety

A quantized policy fails differently from a float one. The drift tables show
the shape: the median observation is usually intact while a minority of
observations move a channel hard, and on several families the channel that
moves is the gripper, saturated. A tail like that is invisible in a mean and
arrives without warning mid-trajectory.

Before the first powered run, and again after every rebuild:

- Run `verify` on held-out observations from your own data and read the
  **min** and the worst-|Δ| channel, not the mean. Know which channel your
  arm damages worst before it damages it in front of you.
- Bring up W8A8 first. It is the arm that stays within a few 1e-4 of bf16 on
  every family measured here; W4A4 is the one with the tail.
- Cap velocity and acceleration in the robot controller, not in the client —
  a policy that emits a bad action should be unable to execute it quickly.
- Bound the workspace in the controller for the same reason.
- Keep a hand on the e-stop for the first runs, and know how long one action
  chunk takes: at a 25-step horizon the robot is executing an open-loop plan
  for that whole time, and a bad chunk is not interrupted by the next
  observation.
- Compare against the bf16 arm on the same task before trusting a number. The
  server serves it by omitting `--engine-dir`; nothing else changes.
