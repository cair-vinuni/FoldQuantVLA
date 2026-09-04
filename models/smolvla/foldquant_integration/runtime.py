# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Serve FoldQuant engines from inside the upstream ``VLAFlowMatching`` model.

Upstream's ``sample_actions`` is two calls apart from the engines: it runs the
SmolVLM2 text stack once over the image + prompt + state prefix, keeps only the
KV cache that call returns, and hands that cache to ``denoise_step`` on every
Euler step. FoldQuant swaps exactly those two collaborators on the live model
instance:

* the LLM engine takes the prefix branch of ``vlm_with_expert.forward``
  (``prefix_embs`` / the BOOL block ``attention_mask`` / ``position_ids`` in,
  the stacked post-RoPE KV cache ``kv_stack`` out). The SigLIP vision tower and
  the token embedding stay in PyTorch — they produce ``prefix_embs``;
* the expert engine takes ``denoise_step`` (``x_t`` / ``timestep`` /
  ``prefix_pad_masks`` / ``kv_stack`` in, ``velocity`` out): the suffix
  embedding, the 16 alternating expert layers, the final expert RMSNorm and
  ``action_out_proj`` all live in the graph. The Euler loop stays upstream's.

The two seams meet on the KV cache. Upstream's attention works in
``[batch, seq, heads, head_dim]`` and only transposes into ``[batch, heads,
seq, head_dim]`` to satisfy ``DynamicCache``; the engine contract keeps the
module's own order, so the stack crosses as
``[layers, 2, 1, prefix_len, kv_heads, head_dim]`` and :func:`stack_cache` /
:func:`cache_from_stack` undo the cache's transpose rather than the module's.
Whichever side is PyTorch sees the layout it expects, so each engine can be
installed alone.

Everything else — the processor pipeline, the tokenizer, normalisation, the
rollout loop — is untouched, so the policy's public behaviour is exactly
upstream's with two modules swapped underneath.

The model must be eager: ``torch.compile`` traces ``sample_actions`` once,
inlining the prefix pass it found at trace time, so a ``forward`` rebound on
the instance afterwards is never called and tensors kept across CUDA-graph
replays are overwritten. Upstream's default is eager (``compile_model =
False``); :func:`install_engines` and :class:`PrefixCapture` refuse a
checkpoint that turns it on.
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

logger = logging.getLogger("foldquant.smolvla.runtime")

LLM_INPUTS = {"prefix_embs", "attention_mask", "position_ids"}
LLM_OUTPUT = "kv_stack"
EXPERT_INPUTS = {"x_t", "timestep", "prefix_pad_masks", "kv_stack"}
EXPERT_OUTPUT = "velocity"


def plugin_libs_of(engine_dir: Path) -> list[str]:
    """Plugin libraries an engine directory needs, from its build record or export manifest."""
    for name in (ENGINES_RECORD_NAME, MANIFEST_NAME):
        path = engine_dir / name
        if path.is_file():
            return list(json.loads(path.read_text()).get("plugin_libs", []))
    raise FileNotFoundError(f"{engine_dir} holds neither {ENGINES_RECORD_NAME} nor {MANIFEST_NAME}")


def model_of(policy) -> torch.nn.Module:
    """The ``VLAFlowMatching`` module inside a :class:`~.calibration.Deployed`, policy or model."""
    model = getattr(policy, "policy", policy)  # calibration.Deployed
    model = getattr(model, "model", model)  # SmolVLAPolicy
    if not hasattr(model, "vlm_with_expert"):
        raise TypeError(f"expected a SmolVLA policy or VLAFlowMatching, got {type(policy).__name__}")
    return model


def require_eager(model: torch.nn.Module) -> None:
    """Refuse a model whose ``sample_actions`` is ``torch.compile``d (see the module docstring)."""
    if getattr(model.config, "compile_model", False):
        raise RuntimeError(
            "the checkpoint's config sets compile_model=True; torch.compile keeps calling the prefix pass it "
            "traced, so a seam rebound on the instance is skipped — load the policy with "
            "calibration.load_policy(..., compile=False)"
        )


def llm_module(policy) -> torch.nn.Module:
    """The SmolVLM2 text decoder (SmolLM2: ``model_type == 'llama'``) the prefix runs through."""
    return model_of(policy).vlm_with_expert.get_vlm_model().text_model


class SmolVlaExpertView(torch.nn.Module):
    """The action expert as one module, the way the FoldQuant SmolVLA emitter reads it.

    Upstream splits the expert across two objects: the projections that bracket
    it (``action_in_proj``, the time MLP, ``action_out_proj``) sit on
    ``VLAFlowMatching`` while the transformer itself is
    ``vlm_with_expert.lm_expert``. The emitter expects them under one root, with
    the expert's own sizing (``expert_hidden_size``, ``self_attn_every_n_layers``)
    and the VLM layer count reachable from it. This view rebinds the live
    submodules (no copies): its state dict aliases the model's parameters, so
    an export always reads the weights the policy is about to run.

    ``backbone`` is the ``SmolVLMWithExpertModel`` module rather than the
    config: the config's ``num_vlm_layers`` may be 0, meaning "as many as the
    VLM has", and only the module carries the resolved count.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        vwe = model.vlm_with_expert
        self.lm_expert = vwe.lm_expert
        self.action_in_proj = model.action_in_proj
        self.action_time_mlp_in = model.action_time_mlp_in
        self.action_time_mlp_out = model.action_time_mlp_out
        self.action_out_proj = model.action_out_proj
        self.config = model.config
        self.backbone = vwe
        self.expert_hidden_size = int(vwe.expert_hidden_size)
        self.self_attn_every_n_layers = int(vwe.self_attn_every_n_layers)


def expert_view(policy) -> SmolVlaExpertView:
    """The action expert of *policy* under the emitter's module layout."""
    return SmolVlaExpertView(model_of(policy))


# ------------------------------------------------------------------ KV cache


def _layer_kv(cache: Any, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One layer's ``(keys, values)`` out of a ``DynamicCache``, across transformers versions.

    transformers 5 keeps them on a per-layer object (``cache.layers[i].keys``);
    4.x made the cache itself subscriptable. The pinned release is 5, and the
    fallback costs one attribute check.
    """
    layers = getattr(cache, "layers", None)
    if layers is not None:
        layer = layers[index]
        return layer.keys, layer.values
    return cache[index]


def stack_cache(cache: Any) -> torch.Tensor:
    """HF ``DynamicCache`` -> ``[layers, 2, 1, prefix_len, kv_heads, head_dim]`` (the engine layout).

    The cache holds ``[batch, heads, seq, head_dim]``; upstream's attention
    transposes back to ``[batch, seq, heads, head_dim]`` on every read, and the
    engine contract is that second layout, so the transpose happens here once.
    """
    pairs = (_layer_kv(cache, i) for i in range(len(cache)))
    layers = [torch.stack((k.transpose(1, 2), v.transpose(1, 2)), dim=0) for k, v in pairs]
    return torch.stack(layers, dim=0)


def cache_from_stack(kv_stack: torch.Tensor) -> Any:
    """The inverse: a ``DynamicCache`` the PyTorch expert's attention indexes per layer."""
    from transformers import DynamicCache

    cache = DynamicCache()
    for i in range(kv_stack.shape[0]):
        cache.update(kv_stack[i, 0].transpose(1, 2), kv_stack[i, 1].transpose(1, 2), i)
    return cache


# ------------------------------------------------------------------ the seams


def _prefix_forward(vwe: torch.nn.Module, engine: TensorRTEngine | None) -> Callable[..., Any]:
    """``vlm_with_expert.forward`` with the prefix branch returning a KV stack.

    With *engine* the prefix runs through the FoldQuant LLM engine; without it
    the PyTorch text model runs and its cache is stacked into the same layout —
    that is what lets the expert engine be installed alone.
    """
    original = type(vwe).forward

    def forward(
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: list[torch.Tensor | None] | None = None,
        use_cache: bool | None = None,
    ) -> tuple[Any, Any]:
        if inputs_embeds is None or inputs_embeds[1] is not None:
            # Suffix / joint branch: not a prefix pass, PyTorch as upstream wrote it.
            return original(
                vwe,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
            )
        if engine is None:
            hidden, cache = original(
                vwe,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
            return hidden, stack_cache(cache)
        if attention_mask is None or position_ids is None:
            raise RuntimeError("the FoldQuant LLM engine needs the 2-D block attention mask and position_ids")
        out = engine(
            prefix_embs=inputs_embeds[0].to(torch.bfloat16),
            attention_mask=attention_mask.to(torch.bool),
            position_ids=position_ids.to(torch.int64),
        )[LLM_OUTPUT]
        # The engine reuses its output buffer across calls: hand back a copy.
        return None, out.clone()

    return forward


def _denoise_step(model: torch.nn.Module, engine: TensorRTEngine | None) -> Callable[..., torch.Tensor]:
    """``denoise_step`` consuming the KV stack the prefix seam returns."""
    original = model.denoise_step  # bound to the PyTorch implementation before the swap
    memo: dict[str, Any] = {}

    def denoise_step(
        self: torch.nn.Module,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if engine is None:
            # One cache per prefix pass; every Euler step of a chunk shares it.
            if memo.get("stack") is not past_key_values:
                memo["stack"] = past_key_values
                memo["cache"] = cache_from_stack(past_key_values)
            return original(prefix_pad_masks, memo["cache"], x_t, timestep)
        velocity = engine(
            x_t=x_t.to(torch.float32),
            timestep=timestep.reshape(1).to(torch.float32),
            prefix_pad_masks=prefix_pad_masks.to(torch.bool),
            kv_stack=past_key_values.to(torch.bfloat16),
        )[EXPERT_OUTPUT]
        return velocity.clone().to(x_t.dtype)

    return denoise_step


class InstalledEngines:
    """Handle over the engines installed by :func:`install_engines`; ``remove()`` restores PyTorch."""

    def __init__(self) -> None:
        self.engines: dict[str, TensorRTEngine] = {}
        self._restore: list[Callable[[], None]] = []

    def rebind(self, obj: Any, name: str, value: Any) -> None:
        # An instance attribute shadows the class method (nn.Module.__call__
        # and upstream's explicit ``.forward(`` / ``.denoise_step(`` calls both
        # go through attribute lookup); deleting it puts the PyTorch one back.
        setattr(obj, name, value)
        self._restore.append(lambda: obj.__dict__.pop(name, None))

    def remove(self) -> None:
        for restore in reversed(self._restore):
            restore()
        self._restore.clear()
        for engine in self.engines.values():
            engine.close()
        self.engines.clear()


def install_engines(
    policy, engine_dir: str | Path, *, components: Iterable[str] | None = None
) -> InstalledEngines:
    """Load the plugins an engine directory needs and swap its engines into *policy*.

    ``components`` restricts the swap (``("expert",)`` keeps the PyTorch LLM);
    by default every engine present in the directory is installed. An engine
    the directory lacks is skipped with a log line when the selection was
    implicit and is an error when it was asked for. Both seams are always
    installed — the KV stack is their shared contract — with the missing side
    running PyTorch.
    """
    engine_dir = Path(engine_dir)
    model = model_of(policy)
    require_eager(model)
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
            engine.validate_binding_names(EXPERT_INPUTS, {EXPERT_OUTPUT})
        installed.engines[name] = engine
        logger.info("%s: serving %s", name, path)
    if not installed.engines:
        raise FileNotFoundError(f"no FoldQuant engine found in {engine_dir}")
    vwe = model.vlm_with_expert
    installed.rebind(vwe, "forward", _prefix_forward(vwe, installed.engines.get("llm")))
    installed.rebind(
        model, "denoise_step", types.MethodType(_denoise_step(model, installed.engines.get("expert")), model)
    )
    return installed


class PrefixCapture:
    """Record the KV stack every prefix pass returns, under PyTorch or the engines.

    Upstream calls ``vlm_with_expert.forward(`` explicitly, so a forward hook
    never fires; this wraps whatever ``forward`` is currently bound (the
    PyTorch one or the FoldQuant seam) and stacks a PyTorch cache into the
    engine layout so the two passes compare tensor to tensor.
    """

    def __init__(self, policy) -> None:
        model = model_of(policy)
        require_eager(model)
        self._vwe = model.vlm_with_expert
        self.stacks: list[torch.Tensor] = []
        self._had_override = False
        self._previous: Any = None

    def __enter__(self) -> PrefixCapture:
        vwe = self._vwe
        self._had_override = "forward" in vwe.__dict__
        self._previous = vwe.__dict__.get("forward")
        inner = self._previous if self._had_override else types.MethodType(type(vwe).forward, vwe)

        def forward(**kwargs: Any) -> Any:
            out = inner(**kwargs)
            embeds = kwargs.get("inputs_embeds")
            if embeds is not None and embeds[1] is None:
                kv = out[1]
                self.stacks.append((kv if isinstance(kv, torch.Tensor) else stack_cache(kv)).detach().clone())
            return out

        vwe.forward = forward  # type: ignore[method-assign]
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._had_override:
            self._vwe.forward = self._previous  # type: ignore[method-assign]
        else:
            self._vwe.__dict__.pop("forward", None)
