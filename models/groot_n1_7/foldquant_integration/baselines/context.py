# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Ported from the authors' Isaac-GR00T fork for the FoldQuant release.

"""Denoising-step context used by timestep-aware DiT quantization.

The fork sets the step inside the action head's Euler loop. This release does
not edit the vendored upstream model, so :func:`install_dit_step_context`
attaches the same context from the outside: a forward pre-hook on the DiT
module (``action_head.model``) counts its calls (upstream calls it exactly
once per denoising step) and publishes ``step = calls % num_inference_timesteps``
for the duration of that call.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from torch import nn


_STEP: ContextVar[int | None] = ContextVar("foldquant_baseline_dit_step", default=None)
_TOTAL_STEPS: ContextVar[int | None] = ContextVar(
    "foldquant_baseline_dit_total_steps", default=None
)


def get_dit_quant_step() -> int | None:
    """Return the active zero-based DiT denoising step, if any."""

    return _STEP.get()


def get_dit_quant_total_steps() -> int | None:
    """Return the number of steps in the active denoising loop, if any."""

    return _TOTAL_STEPS.get()


@contextmanager
def set_dit_quant_step(step: int, *, total_steps: int) -> Iterator[None]:
    """Set the active denoising step for quantized DiT linears.

    Context variables keep concurrent policy requests isolated and restore a
    surrounding context when the block exits.
    """

    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0 <= step < total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}")

    step_token = _STEP.set(step)
    total_token = _TOTAL_STEPS.set(total_steps)
    try:
        yield
    finally:
        _STEP.reset(step_token)
        _TOTAL_STEPS.reset(total_token)


class DiTStepContext:
    """Publish the denoising step around every forward of the DiT module.

    ``action_head.get_action`` runs ``for t in range(num_inference_timesteps)`` and
    calls ``action_head.model`` once per iteration, so the call counter modulo the
    step count is the step. The counter wraps after each full loop, so a policy that
    serves many requests keeps the same alignment; ``reset`` realigns it if a
    request is ever aborted mid-loop.
    """

    def __init__(self, action_head: nn.Module) -> None:
        total = int(getattr(action_head, "num_inference_timesteps", 0))
        if total <= 0:
            raise ValueError("action head exposes no positive num_inference_timesteps")
        self.total_steps = total
        self.calls = 0
        self._tokens: list[tuple[Any, Any]] = []
        dit = action_head.model
        self._handles = [
            dit.register_forward_pre_hook(self._enter),
            dit.register_forward_hook(self._exit),
        ]

    def _enter(self, _module: nn.Module, _args: tuple[Any, ...]) -> None:
        step = self.calls % self.total_steps
        self._tokens.append((_STEP.set(step), _TOTAL_STEPS.set(self.total_steps)))

    def _exit(self, _module: nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
        step_token, total_token = self._tokens.pop()
        _STEP.reset(step_token)
        _TOTAL_STEPS.reset(total_token)
        self.calls += 1

    def reset(self) -> None:
        self.calls = 0

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def install_dit_step_context(model: nn.Module) -> DiTStepContext:
    """Attach :class:`DiTStepContext` to ``model.action_head`` (a ``Gr00tN1d7``)."""

    action_head = getattr(model, "action_head", None)
    if action_head is None or not hasattr(action_head, "model"):
        raise ValueError("model has no action_head.model to attach the step context to")
    return DiTStepContext(action_head)
