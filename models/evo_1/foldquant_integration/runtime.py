# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Serve FoldQuant engines from inside the upstream ``EVO1`` model.

Evo-1's inference is two calls apart from the engines: the InternVL3 embedder
fuses image and text tokens and runs them through the Qwen2 language tower once
per observation, and the flow-matching head then integrates 50 Euler steps
against those tokens. FoldQuant swaps exactly those two collaborators on the
live model instance:

* the LLM engine takes ``embedder.model.language_model``
  (``inputs_embeds`` / ``attention_mask`` in, ``hidden_states`` out). The
  vision tower, the tile fusion and the prompt embedding stay in PyTorch — they
  produce ``inputs_embeds`` — and the engine returns what upstream reads off
  the call, which is the final hidden state: the release sets
  ``lm_head = Identity()``, so its ``logits`` *are* the hidden states;
* the head engine takes one denoise step (``action_seq`` / ``context_tokens`` /
  ``time_emb`` in, ``velocity`` out): the action encoder, the eight
  cross-attention blocks, ``norm_out``, ``seq_pool_proj`` and ``mlp_head`` all
  live in the graph.

Unlike every other family here, upstream gives that step no method of its own:
``FlowmatchingActionHead.get_action`` inlines it inside the Euler loop. So this
module rebinds ``get_action`` and reimplements the loop around the seam —
state encoding, the sinusoidal time table, the action mask and the Euler update
are all still upstream's own calls, and only the block stack is replaced. It is
the one place in this repository where an integration mirrors upstream control
flow instead of calling it, so :func:`_check_loop_contract` asserts the pieces
it depends on are still there, and a release that reshapes the loop must be
re-read against :meth:`get_action` before its numbers are quoted.
"""

from __future__ import annotations

import json
import logging
import types
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from foldquant.runtime.engine import TensorRTEngine
from foldquant.runtime.plugins import load_plugins

from ._upstream import COMPONENTS, ENGINES_RECORD_NAME, MANIFEST_NAME

logger = logging.getLogger("foldquant.evo_1.runtime")

LLM_INPUTS = {"inputs_embeds", "attention_mask"}
LLM_OUTPUT = "hidden_states"
HEAD_INPUTS = {"action_seq", "context_tokens", "time_emb"}
HEAD_OUTPUT = "velocity"


def plugin_libs_of(engine_dir: Path) -> list[str]:
    """Plugin libraries an engine directory needs, from its build record or export manifest."""
    for name in (ENGINES_RECORD_NAME, MANIFEST_NAME):
        path = engine_dir / name
        if path.is_file():
            return list(json.loads(path.read_text()).get("plugin_libs", []))
    raise FileNotFoundError(f"{engine_dir} holds neither {ENGINES_RECORD_NAME} nor {MANIFEST_NAME}")


def model_of(policy) -> torch.nn.Module:
    """The ``EVO1`` module inside a :class:`~.calibration.Deployed` or the model itself."""
    model = getattr(policy, "model", policy)
    if not hasattr(model, "action_head") or not hasattr(model, "embedder"):
        raise TypeError(f"expected an EVO1 model or calibration.Deployed, got {type(policy).__name__}")
    return model


def llm_module(policy) -> torch.nn.Module:
    """The InternVL3 language tower (Qwen2) the fused tokens run through."""
    return model_of(policy).embedder.model.language_model


def head_module(policy) -> torch.nn.Module:
    """The flow-matching action head, which the FoldQuant emitter reads directly.

    Upstream's ``FlowmatchingActionHead`` already has the layout the emitter
    expects (``action_encoder.W*.linear``, ``transformer_blocks.*``,
    ``norm_out``, ``seq_pool_proj``, ``mlp_head.fc*.linear``), so no view is
    needed — but only while the category-specific modules are plain linears,
    which is what ``num_categories <= 1`` builds.
    """
    head = model_of(policy).action_head
    categories = int(getattr(head.config, "num_categories", 1) or 1)
    if categories > 1:
        raise NotImplementedError(
            f"the FoldQuant Evo-1 head graph is emitted for plain linears; this checkpoint has "
            f"num_categories={categories}, whose CategorySpecificLinear keeps a stacked weight instead"
        )
    return head


def _check_loop_contract(head: torch.nn.Module) -> None:
    """Fail loudly if upstream's denoise loop no longer has the pieces the seam reuses."""
    missing = [
        name
        for name in ("_project_actions", "_expand_action_mask", "time_pos_enc", "transformer_blocks", "norm_out")
        if not hasattr(head, name)
    ]
    if missing:
        raise RuntimeError(
            f"FlowmatchingActionHead is missing {missing}: the FoldQuant seam reimplements get_action's loop "
            "around upstream's own pieces and must be re-read against this release"
        )


# ------------------------------------------------------------------ the seams


def _language_model_forward(engine: TensorRTEngine) -> Callable[..., Any]:
    """``language_model(...)`` returning the engine's hidden states under upstream's contract.

    Upstream reads ``outputs.logits if hasattr(outputs, "logits") else
    outputs[0]``, so the reply carries both: a namespace with ``logits`` that
    also indexes like a tuple.
    """

    class _Output(tuple):
        __slots__ = ()

        def __new__(cls, hidden: torch.Tensor) -> _Output:
            return super().__new__(cls, (hidden,))

        @property
        def logits(self) -> torch.Tensor:
            return self[0]

    def forward(
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> Any:
        if inputs_embeds is None or attention_mask is None:
            raise RuntimeError("the FoldQuant Evo-1 LLM engine needs inputs_embeds and attention_mask")
        out = engine(
            inputs_embeds=inputs_embeds.to(torch.bfloat16),
            attention_mask=attention_mask.to(torch.int64),
        )[LLM_OUTPUT]
        # The engine reuses its output buffer across calls: hand back a copy.
        return _Output(out.clone())

    return forward


def _get_action(head: torch.nn.Module, engine: TensorRTEngine) -> Callable[..., torch.Tensor]:
    """``FlowmatchingActionHead.get_action`` with the block stack served by the engine.

    Every line that is not the block stack is upstream's: the state token is
    encoded and concatenated by ``state_encoder``, the start sample is drawn
    from the same ``torch.rand`` against the global RNG, the sinusoidal table is
    ``time_pos_enc``, the mask is ``_expand_action_mask``, and the update is the
    same ``action + dt * pred``.
    """
    _check_loop_contract(head)

    def get_action(
        self: torch.nn.Module,
        fused_tokens: torch.Tensor,
        state: torch.Tensor | None = None,
        embodiment_id: torch.LongTensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = fused_tokens.size(0)
        if batch != 1:
            raise RuntimeError(f"the FoldQuant head graph is emitted at batch 1, got {batch}")
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(batch, dtype=torch.long, device=device)

        context_tokens = fused_tokens
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1)
            context_tokens = torch.cat([context_tokens, state_emb], dim=1)

        action_dim_total = getattr(self.config, "action_dim", None) or self.action_dim
        per_action_dim = (
            getattr(self.config, "per_action_dim", action_dim_total // self.horizon)
            if self.horizon > 1
            else action_dim_total
        )

        action = torch.rand(batch, action_dim_total, device=device, dtype=context_tokens.dtype) * 2 - 1
        action_seq = action.view(batch, max(self.horizon, 1), per_action_dim)
        action_mask = self._expand_action_mask(
            action_mask, batch_size=batch, per_action_dim=per_action_dim, device=device, dtype=action_seq.dtype
        )
        action_seq = action_seq * action_mask
        context_tokens = context_tokens.to(dtype=self.dtype)

        steps = int(getattr(self.config, "num_inference_timesteps", 32))
        if steps <= 0:
            raise ValueError(f"num_inference_timesteps must be positive, got {steps}")
        dt = 1.0 / steps
        table = self.time_pos_enc(1000)
        for i in range(steps):
            time_index = min(int((i / steps) * 999), 999)
            time_emb = table[:, time_index, :].to(device).squeeze(0).to(dtype=context_tokens.dtype)
            time_emb = time_emb.unsqueeze(0).repeat(batch, 1)
            action_seq = action_seq * action_mask
            pred = engine(
                action_seq=action_seq.to(torch.bfloat16),
                context_tokens=context_tokens.to(torch.bfloat16),
                time_emb=time_emb.to(torch.bfloat16),
            )[HEAD_OUTPUT]
            action = action + dt * pred.clone().to(action.dtype)
            action_seq = action.view(batch, max(self.horizon, 1), per_action_dim)

        action_seq = action_seq * action_mask
        return action_seq.reshape(batch, -1)

    return get_action


class InstalledEngines:
    """Handle over the engines installed by :func:`install_engines`; ``remove()`` restores PyTorch."""

    def __init__(self) -> None:
        self.engines: dict[str, TensorRTEngine] = {}
        self._restore: list[Callable[[], None]] = []

    def rebind(self, obj: Any, name: str, value: Any) -> None:
        # An instance attribute shadows the class method (nn.Module.__call__
        # and upstream's explicit calls both go through attribute lookup);
        # deleting it puts the PyTorch one back.
        setattr(obj, name, value)
        self._restore.append(lambda: obj.__dict__.pop(name, None))

    def remove(self) -> None:
        for restore in reversed(self._restore):
            restore()
        self._restore.clear()
        for engine in self.engines.values():
            engine.close()
        self.engines.clear()


def install_engines(policy, engine_dir: str | Path, *, components: Iterable[str] | None = None) -> InstalledEngines:
    """Load the plugins an engine directory needs and swap its engines into *policy*.

    ``components`` restricts the swap (``("action_head",)`` keeps the PyTorch
    tower); by default every engine present in the directory is installed. A
    module the directory has no engine for keeps running in PyTorch, untouched:
    unlike the KV-stack families there is no shared tensor to rebuild, because
    the seams meet on the fused tokens the tower already returns.
    """
    engine_dir = Path(engine_dir)
    model_of(policy)  # refuse anything that is not an EVO1 before touching the engines
    load_plugins(plugin_libs_of(engine_dir))
    wanted = None if components is None else set(components)
    installed = InstalledEngines()
    for name, _onnx, engine_name in COMPONENTS:
        if wanted is not None and name not in wanted:
            continue
        path = engine_dir / engine_name
        if not path.is_file():
            if wanted is not None:
                raise FileNotFoundError(f"{name} engine requested but {path} does not exist")
            logger.info("%s: no %s in %s, PyTorch module kept", name, engine_name, engine_dir)
            continue
        engine = TensorRTEngine(path)
        if name == "llm":
            engine.validate_binding_names(LLM_INPUTS, {LLM_OUTPUT})
        else:
            engine.validate_binding_names(HEAD_INPUTS, {HEAD_OUTPUT})
        installed.engines[name] = engine
        logger.info("%s: serving %s", name, path)
    if not installed.engines:
        raise FileNotFoundError(f"no FoldQuant engine found in {engine_dir}")

    if "llm" in installed.engines:
        tower = llm_module(policy)
        installed.rebind(tower, "forward", _language_model_forward(installed.engines["llm"]))
    if "action_head" in installed.engines:
        head = head_module(policy)
        installed.rebind(
            head, "get_action", types.MethodType(_get_action(head, installed.engines["action_head"]), head)
        )
    return installed


class ContextCapture:
    """Record the fused tokens the language tower returns, under PyTorch or the engine.

    Upstream calls ``self.model.language_model(`` explicitly and reads the
    result's ``logits``, so a forward hook on the tower's parent never fires;
    this wraps whatever ``forward`` is currently bound so the two passes compare
    tensor to tensor.
    """

    def __init__(self, policy) -> None:
        self._tower = llm_module(policy)
        self.hidden: list[torch.Tensor] = []
        self._had_override = False
        self._previous: Any = None

    def __enter__(self) -> ContextCapture:
        tower = self._tower
        self._had_override = "forward" in tower.__dict__
        self._previous = tower.__dict__.get("forward")
        inner = self._previous if self._had_override else types.MethodType(type(tower).forward, tower)

        def forward(*args: Any, **kwargs: Any) -> Any:
            out = inner(*args, **kwargs)
            hidden = out.logits if hasattr(out, "logits") else out[0]
            self.hidden.append(hidden.detach().clone())
            return out

        tower.forward = forward  # type: ignore[method-assign]
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._had_override:
            self._tower.forward = self._previous  # type: ignore[method-assign]
        else:
            self._tower.__dict__.pop("forward", None)
