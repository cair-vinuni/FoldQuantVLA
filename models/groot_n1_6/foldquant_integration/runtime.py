# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Serve FoldQuant engines from inside the upstream ``Gr00tPolicy``.

Upstream N1.6 deploys one TensorRT engine (the DiT) by rebinding
``policy.model.action_head.model.forward`` to the engine call
(``standalone_inference_script.replace_dit_with_tensorrt``). This module does
the same for the two modules FoldQuant quantizes:

* the LLM engine takes the place of ``backbone.model.language_model.forward``
  (``inputs_embeds`` in, ``hidden_states`` out); the Eagle wrapper around it
  keeps splicing the vision features into the embeddings in PyTorch, and the
  backbone keeps reading ``hidden_states[-1]`` as before;
* the DiT engine takes the place of ``action_head.model.forward``; the flow-
  matching loop, the state / action encoders and the action decoder stay in
  PyTorch.

Everything else (processor, ViT, collation, action decoding) is untouched,
so the policy's public behaviour is exactly upstream's with two modules
swapped underneath.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from foldquant.runtime.engine import TensorRTEngine
from foldquant.runtime.plugins import load_plugins
import torch

from ._upstream import COMPONENTS, ENGINES_RECORD_NAME, MANIFEST_NAME


logger = logging.getLogger("foldquant.groot_n1_6.runtime")

LLM_INPUTS = {"inputs_embeds", "attention_mask"}
LLM_OUTPUT = "hidden_states"
DIT_INPUTS = {"sa_embs", "vl_embs", "timestep", "image_mask", "backbone_attention_mask"}
DIT_OUTPUT = "output"


def plugin_libs_of(engine_dir: Path) -> List[str]:
    """Plugin libraries an engine directory needs, from its build record or export manifest."""
    for name in (ENGINES_RECORD_NAME, MANIFEST_NAME):
        path = engine_dir / name
        if path.is_file():
            return list(json.loads(path.read_text()).get("plugin_libs", []))
    raise FileNotFoundError(f"{engine_dir} holds neither {ENGINES_RECORD_NAME} nor {MANIFEST_NAME}")


def llm_module(policy) -> torch.nn.Module:
    """The Qwen3 causal LM inside the Eagle backbone (16 kept decoder layers)."""
    return policy.model.backbone.model.language_model


def dit_module(policy) -> torch.nn.Module:
    """The action head's ``AlternateVLDiT``."""
    return policy.model.action_head.model


def _llm_forward(engine: TensorRTEngine) -> Callable[..., Any]:
    from transformers.modeling_outputs import CausalLMOutputWithPast

    def forward(
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **_unused: Any,
    ) -> Any:
        if inputs_embeds is None:
            raise RuntimeError(
                "the FoldQuant LLM engine takes inputs_embeds; the Eagle wrapper always passes them"
            )
        embeds = inputs_embeds.to(torch.bfloat16)
        if attention_mask is None:
            attention_mask = torch.ones(embeds.shape[:2], device=embeds.device)
        outputs = engine(inputs_embeds=embeds, attention_mask=attention_mask.to(torch.int64))
        out = outputs[LLM_OUTPUT]
        # The engine reuses its output buffer across calls: hand back a copy.
        return CausalLMOutputWithPast(logits=None, hidden_states=(out.clone(),))

    return forward


def _dit_forward(engine: TensorRTEngine) -> Callable[..., Any]:
    def forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if return_all_hidden_states:
            raise RuntimeError("the FoldQuant DiT engine returns the final hidden state only")
        if timestep is None:
            raise RuntimeError("the FoldQuant DiT engine needs the discretised timestep")
        vl = encoder_hidden_states.to(torch.bfloat16)
        # True = attend. A plain (non-alternating) DiT passes no masks; attend everything.
        ones = torch.ones(vl.shape[:2], dtype=torch.bool, device=vl.device)
        out = engine(
            sa_embs=hidden_states.to(torch.bfloat16),
            vl_embs=vl,
            timestep=timestep.to(torch.int64),
            image_mask=ones if image_mask is None else image_mask.to(torch.bool),
            backbone_attention_mask=(
                ones if backbone_attention_mask is None else backbone_attention_mask.to(torch.bool)
            ),
        )[DIT_OUTPUT]
        return out.clone()

    return forward


class InstalledEngines:
    """Handle over the engines installed by :func:`install_engines`; ``remove()`` restores PyTorch."""

    def __init__(self) -> None:
        self.engines: Dict[str, TensorRTEngine] = {}
        self._restore: List[Callable[[], None]] = []

    def _rebind(self, module: torch.nn.Module, forward: Callable[..., Any]) -> None:
        # An instance attribute shadows the class method for nn.Module.__call__;
        # deleting it puts the PyTorch forward back.
        module.forward = forward  # type: ignore[method-assign]
        self._restore.append(lambda: module.__dict__.pop("forward", None))

    def remove(self) -> None:
        for restore in reversed(self._restore):
            restore()
        self._restore.clear()
        for engine in self.engines.values():
            engine.close()
        self.engines.clear()


def install_engines(
    policy, engine_dir: str | Path, *, components: Optional[Iterable[str]] = None
) -> InstalledEngines:
    """Load the plugins an engine directory needs and swap its engines into *policy*.

    ``components`` restricts the swap (``("dit",)`` keeps the PyTorch LLM);
    by default every engine present in the directory is installed. An engine
    the directory lacks is skipped with a log line when the selection was
    implicit and is an error when it was asked for.
    """
    engine_dir = Path(engine_dir)
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
            installed._rebind(llm_module(policy), _llm_forward(engine))
        else:
            engine.validate_binding_names(DIT_INPUTS, {DIT_OUTPUT})
            installed._rebind(dit_module(policy), _dit_forward(engine))
        installed.engines[name] = engine
        logger.info("%s: serving %s", name, path)
    if not installed.engines:
        raise FileNotFoundError(f"no FoldQuant engine found in {engine_dir}")
    return installed
