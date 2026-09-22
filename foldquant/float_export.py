"""Float (unquantized) engine export: the floor arm of every ladder.

A module is traced with ``torch.onnx.export`` under the *runtime's* engine
binding names, so :mod:`build_engines` compiles it and ``install_engines``
serves it exactly like a plugin graph, with ``plugin_libs`` empty. Nothing
downstream distinguishes the arms except the precision of the projections,
which is the property the paper's latency table relies on.

The example inputs are not invented: the module's real call is captured by a
forward pre-hook during one iteration of the same ``forward_loop`` the
quantized exports calibrate on, so shapes, dtypes and every non-tensor kwarg
(``output_hidden_states``, ``return_all_hidden_states``, ...) are the deployed
ones. Each family declares only the mapping from engine binding name to the
module's keyword, the same mapping its ``runtime.py`` applies in reverse.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import torch
import torch.nn as nn

from .export import ExportResult, ForwardLoop

logger = logging.getLogger(__name__)

FLOAT = "float"
"""Scheme key for the unquantized engine. ``none`` keeps the module in PyTorch instead."""


@dataclass(frozen=True)
class Binding:
    """One engine input: its binding name, the module keyword it feeds, and the engine dtype."""

    engine_name: str
    kwarg: str
    dtype: torch.dtype
    #: Dynamic axes as ``{axis: dim_name}``; axes not listed are pinned to the captured size.
    dynamic: Dict[int, str] = field(default_factory=dict)
    #: Builds the example tensor when the captured kwarg is ``None`` (the runtime does the
    #: same at serve time, e.g. an all-True mask). Receives the full captured kwargs.
    default: Optional[Callable[[Dict[str, Any]], torch.Tensor]] = None
    #: The engine binds this input but the module's forward has no keyword for it (the
    #: runtime fills it with a constant). It is kept alive in the trace through an exact
    #: ``out + 0 * sum(input)`` so the exported graph carries every binding the runtime checks.
    passthrough: bool = False
    #: Rewrites the bound tensor before it reaches the module, e.g. to hand a decoder a
    #: prebuilt 4-D additive mask instead of the 2-D padding mask it would otherwise promote
    #: through an implicit ``int64 + bf16`` add, the op whose dynamic dtype the TorchScript
    #: exporter mis-maps to COMPLEX128. Receives ``(tensor, module_kwargs)``.
    transform: Optional[Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]] = None


@dataclass
class CapturedCall:
    args: tuple
    kwargs: Dict[str, Any]


def capture_call(module: nn.Module, forward_loop: ForwardLoop) -> CapturedCall:
    """Run ``forward_loop`` until *module* is called once; return that call's arguments."""
    seen: List[CapturedCall] = []

    class _Stop(Exception):
        pass

    def hook(_m: nn.Module, args: tuple, kwargs: Dict[str, Any]) -> None:
        seen.append(CapturedCall(args, dict(kwargs)))
        raise _Stop()

    handle = module.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            try:
                forward_loop(module)
            except _Stop:
                pass
    finally:
        handle.remove()
    if not seen:
        raise RuntimeError(f"forward_loop never called {type(module).__name__}; cannot capture example inputs.")
    return seen[0]


@contextmanager
def eager_attention(module: nn.Module) -> Iterator[None]:
    """Force ``config._attn_implementation = "eager"`` on every sub-config for the trace.

    flash-attn and SDPA kernels cannot be traced; eager attention is the same
    function. Restored on exit so the PyTorch arm keeps its deployed kernels.
    """
    saved: List[tuple] = []
    for m in module.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            saved.append((cfg, cfg._attn_implementation))
            try:
                cfg._attn_implementation = "eager"
            except Exception:  # some configs guard the setter; the trace then uses whatever it has
                saved.pop()
    try:
        yield
    finally:
        for cfg, impl in saved:
            cfg._attn_implementation = impl


def export_module_float(
    module: nn.Module,
    onnx_path: Path,
    *,
    module_name: str,
    bindings: Sequence[Binding],
    output_name: str,
    forward_loop: ForwardLoop,
    extract: Callable[[Any], torch.Tensor],
    output_dynamic: Optional[Dict[int, str]] = None,
    opset: int = 17,
) -> ExportResult:
    """Trace *module* to ONNX under the engine's binding names.

    Args:
        bindings: engine inputs in binding order; each names the module kwarg it feeds.
        output_name: the engine's output binding.
        extract: maps the module's return value to the tensor the engine emits
            (e.g. ``lambda o: o.hidden_states[-1]`` for a decoder).
        output_dynamic: dynamic axes of the output, ``{axis: dim_name}``.
    """
    call = capture_call(module, forward_loop)
    kw = dict(call.kwargs)
    # Positional args are folded into kwargs by the forward signature's parameter names.
    if call.args:
        import inspect

        names = [p for p in inspect.signature(module.forward).parameters if p not in ("self",)]
        for name, val in zip(names, call.args):
            kw.setdefault(name, val)
    return export_with_example(
        module, onnx_path, module_name=module_name, bindings=bindings, output_name=output_name,
        example_kwargs=kw, extract=extract, output_dynamic=output_dynamic, opset=opset,
    )


def export_with_example(
    module: nn.Module,
    onnx_path: Path,
    *,
    module_name: str,
    bindings: Sequence[Binding],
    output_name: str,
    example_kwargs: Dict[str, Any],
    extract: Callable[[Any], torch.Tensor],
    output_dynamic: Optional[Dict[int, str]] = None,
    opset: int = 17,
    call: Optional[Callable[..., Any]] = None,
) -> ExportResult:
    """Like :func:`export_module_float` but with the module's call given explicitly.

    For modules whose deployed step is not their ``forward`` (a head that runs its Euler loop
    inside a sampling method, with the engine being one step of it), the caller assembles the
    example kwargs itself and may pass ``call`` (a function taking the same kwargs) in place
    of ``module(**kwargs)``.
    """
    kw = dict(example_kwargs)
    fn = call if call is not None else (lambda **k: module(**k))
    for b in bindings:
        if kw.get(b.kwarg) is None and b.default is not None:
            kw[b.kwarg] = b.default(kw)
    missing = [b.kwarg for b in bindings if kw.get(b.kwarg) is None]
    passthrough = {b.kwarg for b in bindings if b.passthrough}
    if missing:
        raise RuntimeError(f"{module_name}: captured call has no {missing}; bindings do not match this module's forward.")

    orig_dtype = {b.kwarg: kw[b.kwarg].dtype for b in bindings}
    fixed = {k: v for k, v in kw.items() if k not in {b.kwarg for b in bindings}}
    # A passthrough kwarg is not a module keyword: never forward it.
    fixed = {k: v for k, v in fixed.items() if k not in passthrough}
    # Tensors we do not bind must be constants of the trace; keep them, but warn: a
    # tensor the engine cannot receive is a tensor the deployed call cannot vary.
    for k, v in fixed.items():
        if isinstance(v, torch.Tensor):
            logger.warning("%s: kwarg %r is a tensor but has no engine binding; baked as a constant.", module_name, k)

    class Traced(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = module

        def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
            call_kw = dict(fixed)
            keep = []
            for b, t in zip(bindings, tensors):
                if b.passthrough:
                    keep.append(t)
                else:
                    # Floating inputs take the ENGINE dtype (what the deployed graph receives), not the
                    # dtype the PyTorch call happened to carry: the policy runs its action head under
                    # autocast, so a captured fp32 tensor is downcast at run time but not under export.
                    tgt = b.dtype if orig_dtype[b.kwarg].is_floating_point else orig_dtype[b.kwarg]
                    v = t.to(tgt)
                    call_kw[b.kwarg] = b.transform(v, call_kw) if b.transform else v
            out = extract(fn(**call_kw))
            for t in keep:  # exact: 0 * finite = 0; keeps the binding in the traced graph
                out = out + (t.to(out.dtype).sum() * 0).to(out.dtype)
            return out

    example = tuple(kw[b.kwarg].detach().to(b.dtype) for b in bindings)
    dynamic_axes: Dict[str, Dict[int, str]] = {b.engine_name: dict(b.dynamic) for b in bindings if b.dynamic}
    if output_dynamic:
        dynamic_axes[output_name] = dict(output_dynamic)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    was_training = module.training
    module.eval()
    try:
        with torch.inference_mode(), eager_attention(module):
            torch.onnx.export(
                Traced(),
                example,
                str(onnx_path),
                input_names=[b.engine_name for b in bindings],
                output_names=[output_name],
                dynamic_axes=dynamic_axes or None,
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )
    finally:
        module.train(was_training)
    n_fixed = sanitize_onnx(onnx_path)
    logger.info(
        "Emitted float %s graph at %s (%s)%s",
        module_name, onnx_path, ", ".join(b.engine_name for b in bindings),
        f"; rewrote {n_fixed} TensorRT-unsupported Cast target(s) to FLOAT" if n_fixed else "",
    )
    return ExportResult(module_name, FLOAT, onnx_path, [])


# Cast targets TensorRT's ONNX parser accepts. A traced mask or RoPE helper occasionally
# lands on DOUBLE or COMPLEX (e.g. ``attention_mask[:, None, None, :]`` -> Cast -> Add -> Equal
# on transformers 5.x); the values are 0/1 or finite reals, so FLOAT is exact for the
# comparison that follows and keeps the graph loadable.
def sanitize_onnx(onnx_path: Path) -> int:
    """Rewrite Cast targets TensorRT cannot parse to FLOAT, in place. Returns the count."""
    import onnx
    from onnx import TensorProto

    ok = {TensorProto.FLOAT, TensorProto.UINT8, TensorProto.INT8, TensorProto.INT32, TensorProto.INT64,
          TensorProto.BOOL, TensorProto.FLOAT16, TensorProto.BFLOAT16}
    model = onnx.load(str(onnx_path), load_external_data=False)
    n = 0
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i not in ok:
                logger.warning("%s: Cast %s -> %s is not parseable by TensorRT; rewriting to FLOAT",
                               onnx_path.name, node.name, TensorProto.DataType.Name(attr.i))
                attr.i = TensorProto.FLOAT
                n += 1
    # CumSum over a bool / int8 / uint8 tensor (a traced ``mask.cumsum()`` for positions): TensorRT's
    # CumulativeLayer takes float/half/bf16/int32/int64 only -> insert Cast(INT64) on that input.
    cum_ok = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.INT32, TensorProto.INT64}
    cums = [nd for nd in model.graph.node if nd.op_type == "CumSum"]
    if cums:
        from onnx import shape_inference
        try:
            inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
            dt = {vi.name: vi.type.tensor_type.elem_type for vi in list(inferred.graph.value_info) + list(inferred.graph.input)}
            dt.update({t.name: t.data_type for t in model.graph.initializer})
        except Exception as e:  # inference is best-effort; fall back to casting every CumSum input
            logger.warning("%s: shape inference failed (%s); casting every CumSum input to INT64", onnx_path.name, e)
            dt = {}
        new_nodes = []
        for nd in model.graph.node:
            if nd.op_type == "CumSum" and dt.get(nd.input[0], 0) not in cum_ok:
                cast_out = nd.input[0] + "_i64_for_cumsum"
                new_nodes.append(onnx.helper.make_node("Cast", [nd.input[0]], [cast_out], to=TensorProto.INT64,
                                                       name=nd.name + "_cast_i64"))
                nd.input[0] = cast_out
                n += 1
                logger.warning("%s: CumSum %s input is %s; inserted Cast to INT64", onnx_path.name, nd.name,
                               TensorProto.DataType.Name(dt.get(nd.input[0].replace("_i64_for_cumsum", ""), 0)))
            new_nodes.append(nd)
        if len(new_nodes) != len(model.graph.node):
            del model.graph.node[:]
            model.graph.node.extend(new_nodes)
    if n:
        onnx.save(model, str(onnx_path))
    return n


def causal_additive_mask_4d(padding_mask: torch.Tensor, kw: Dict[str, Any], *, ref: str = "inputs_embeds") -> torch.Tensor:
    """``[B, S]`` 0/1 padding mask -> ``[B, 1, S, S]`` additive causal mask in the model dtype.

    Built with static dtypes only, so the trace carries no implicit promotion. HF decoders
    (transformers 4.x ``_update_causal_mask`` and 5.x ``create_causal_mask``) use a 4-D mask
    as given. ``ref`` names the kwarg whose dtype/device the mask must match.
    """
    x = kw[ref]
    dtype, device = x.dtype, x.device
    b, s = padding_mask.shape
    neg = torch.finfo(dtype).min
    causal = torch.ones(s, s, dtype=torch.bool, device=device).tril()
    keep = causal[None, None] & padding_mask.to(torch.bool)[:, None, None, :]
    return torch.where(keep, torch.zeros((), dtype=dtype, device=device), torch.full((), neg, dtype=dtype, device=device))
