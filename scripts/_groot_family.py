# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.

"""Shared plumbing for the GR00T LLM tuning scripts (ARC sweep, learned calibration).

The three GR00T integrations expose the same calibration API
(``load_policy`` / ``load_dataset`` / ``sample_observations`` /
``make_forward_loop``) and the same ``_module_paths`` map, so one loader
serves all of them. Run from the family's own virtualenv — the integration is
pinned to its upstream environment — with the family named on the command line;
this module puts ``models/<family>`` on ``sys.path`` so
``foldquant_integration`` resolves to that family's copy.

The Pi0.5 integration drives inference through
``calibration.infer(...)`` rather than ``policy.get_action`` and are not covered
here; the ARC presets in ``results/`` were measured on the GR00T families only.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("groot_n1_5", "groot_n1_6", "groot_n1_7")

logger = logging.getLogger("groot_family")


def integration(family: str) -> Tuple[Any, Any]:
    """``(calibration, export_foldquant)`` modules of ``models/<family>/foldquant_integration``."""
    if family not in FAMILIES:
        raise SystemExit(f"--family must be one of {FAMILIES}; {family!r} has a different calibration API")
    root = REPO / "models" / family
    if not (root / "foldquant_integration").is_dir():
        raise SystemExit(f"{root} has no foldquant_integration/ — is the upstream checkout in place?")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import foldquant_integration.calibration as calibration
    import foldquant_integration.export_foldquant as export_foldquant

    return calibration, export_foldquant


class Loaded:
    """Policy, its decoder, and the sampled observations for one tuning run."""

    def __init__(
        self,
        family: str,
        model_path: str,
        embodiment_tag: Optional[str],
        dataset_path: str,
        *,
        num_calib: int,
        seed: int,
        num_heldout: int = 0,
        video_backend: str = "torchcodec",
    ) -> None:
        from foldquant.llm import resolve_qwen3_decoder

        self.calibration, self.export_foldquant = integration(family)
        self.family = family
        self.seed = seed
        self.policy = self.calibration.load_policy(model_path, embodiment_tag)
        dataset = self.calibration.load_dataset(self.policy, dataset_path, video_backend)
        self.samples, self.observations = self.calibration.sample_observations(
            self.policy, dataset, num_calib, seed=seed
        )
        self.heldout: List[Dict[str, Any]] = []
        if num_heldout:
            excluded = sorted({s.episode for s in self.samples})
            _, self.heldout = self.calibration.sample_observations(
                self.policy, dataset, num_heldout, seed=seed, exclude_episodes=excluded, heldout=True
            )
        modules = self.export_foldquant._module_paths(self.policy)
        self.decoder = resolve_qwen3_decoder(modules["llm"])
        logger.info(
            "%s: %d calibration + %d held-out observations, decoder %s (%d layers)",
            family,
            len(self.observations),
            len(self.heldout),
            type(self.decoder).__name__,
            len(self.decoder.layers),
        )

    def forward_loop(self, observations: Optional[Sequence[Dict[str, Any]]] = None) -> Callable[[Any], None]:
        obs = self.observations if observations is None else observations
        return self.calibration.make_forward_loop(self.policy, obs, seed=self.seed)

    def capture(self, observations: Optional[Sequence[Dict[str, Any]]] = None) -> list:
        """Raw ``(args, kwargs)`` decoder calls of one replay — one per observation."""
        from foldquant.calibrate import capture_llm_snapshots

        return capture_llm_snapshots(self.decoder, self.forward_loop(observations))

    def actions(self, observations: Optional[Sequence[Dict[str, Any]]] = None) -> List[torch.Tensor]:
        """Flat decoded action chunk per observation, seeded so reference and candidate
        integrate from the same flow-matching noise — without that two runs of the
        SAME model drift by ~0.5 max-abs and swamp any knob difference."""
        obs = self.observations if observations is None else observations
        out: List[torch.Tensor] = []
        reset = getattr(self.policy, "reset", None)
        with torch.inference_mode():
            for i, o in enumerate(obs):
                if callable(reset):
                    reset()
                torch.manual_seed(10_000 + i)
                out.append(action_vector(self.policy.get_action(o)))
        return out


def action_vector(result: Any) -> torch.Tensor:
    """Channel-major flat action from a ``get_action`` result (dict, or ``(dict, info)``)."""
    action = result[0] if isinstance(result, tuple) else result
    parts = []
    for k in sorted(action):
        v = action[k]
        if isinstance(v, torch.Tensor):
            t = v
        else:
            arr = np.asarray(v)
            if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
                continue  # task strings / metadata carry no action signal
            t = torch.as_tensor(arr)
        if t.numel():
            parts.append(t.detach().float().reshape(-1).cpu())
    return torch.cat(parts) if parts else torch.empty(0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av, bv = a.detach().double().reshape(-1), b.detach().double().reshape(-1)
    denom = av.norm() * bv.norm()
    return float("nan") if float(denom) == 0.0 else float(torch.dot(av, bv) / denom)


def decoder_output(out: Any) -> torch.Tensor:
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, (tuple, list)):
        return out[0]
    return out
