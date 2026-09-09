"""Does the cascade emulation reproduce the deployed W4A4 LLM? Compare kv_stack, PyTorch-emulated vs engine.

Usage: python -m foldquant_integration.validate_emulation --checkpoint ... --dataset-path ... \
           --engine-dir exports/w4a4_cascade/engines [--num 8]
"""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import torch
from foldquant.export import export_llm, install_llm_emulation
from foldquant.runtime.engine import TensorRTEngine
from foldquant.runtime.plugins import load_plugins
from . import calibration
from .runtime import llm_module, model_of, plugin_libs_of, stack_cache

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--engine-dir", required=True); ap.add_argument("--num", type=int, default=8); ap.add_argument("--num-calib", type=int, default=128)
    a = ap.parse_args()
    deployed = calibration.load_policy(a.checkpoint, device="cuda", compile=False)
    model, llm = model_of(deployed), llm_module(deployed)
    ds = calibration.load_dataset(deployed, a.dataset_path) if "dataset_path" in calibration.load_dataset.__code__.co_varnames else calibration.load_dataset(a.dataset_path)
    samples, obs = calibration.sample_observations(deployed, ds, a.num_calib, seed=0)
    loop = calibration.make_forward_loop(deployed, obs, seed=0)
    eng_dir = Path(a.engine_dir); manifest = json.loads((eng_dir / "foldquant_export.json").read_text())
    scheme = manifest["schemes"]["llm"]
    # 1) engine kv_stack on held-out prefixes
    load_plugins(plugin_libs_of(eng_dir)); engine = TensorRTEngine(eng_dir / "llm_bf16.engine")
    _, held = calibration.sample_observations(deployed, ds, a.num, seed=42)
    prefixes = []
    vwe = model.vlm_with_expert; orig = vwe.forward
    def spy(*args, **kw):
        if kw.get("inputs_embeds") is not None and kw["inputs_embeds"][1] is None:
            prefixes.append({k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kw.items()})
        return orig(*args, **kw)
    vwe.forward = spy
    with torch.inference_mode():
        for o in held: deployed.select_action(o) if hasattr(deployed, "select_action") else deployed(o)
    vwe.__dict__.pop("forward", None)
    def eng_stack(p):
        am = p["attention_mask"]; am = am[None] if am.dim() == 2 else am
        return engine(prefix_embs=p["inputs_embeds"][0][0].to(torch.bfloat16), attention_mask=am.to(torch.bool),
                      position_ids=p["position_ids"].to(torch.int64))["kv_stack"].clone()
    with torch.inference_mode():
        eng = [eng_stack(p) for p in prefixes]
    # 2) emulated PyTorch kv_stack: re-derive the fold from the same calibration, install, run the prefix
    with tempfile.TemporaryDirectory() as td:
        res = export_llm(llm, Path(td) / "llm.onnx", scheme=scheme, forward_loop=loop, params=(manifest.get("params") or {}).get("llm") or None)
    handle = install_llm_emulation(llm, res)
    try:
        def emu_stack(p):
            am = p["attention_mask"]; am4 = None
            out = llm(inputs_embeds=p["inputs_embeds"][0], attention_mask=am if am.dim() != 3 else am[:, None], position_ids=p["position_ids"], use_cache=True)
            return stack_cache(out.past_key_values)
        with torch.inference_mode():
            emu = [emu_stack(p) for p in prefixes]
    finally:
        handle.remove()
    # 3) compare per layer, K and V
    cos = torch.nn.functional.cosine_similarity
    L = eng[0].shape[0]; rows = []
    for li in range(L):
        for kv, name in ((0, "K"), (1, "V")):
            c = torch.stack([cos(e[li, kv].float().flatten(), m[li, kv].float().flatten(), dim=0) for e, m in zip(eng, emu)])
            rows.append((li, name, c.mean().item(), c.min().item()))
    worst = min(rows, key=lambda r: r[3])
    print(f"layers={L} n={len(eng)} scheme={scheme}")
    print(f"mean cosine over all layers/K,V: {sum(r[2] for r in rows)/len(rows):.5f}   worst layer {worst[0]} {worst[1]} min {worst[3]:.5f}")
    for r in rows[:6] + rows[-2:]: print(f"  layer {r[0]:2d} {r[1]}  mean {r[2]:.5f}  min {r[3]:.5f}")
    (eng_dir / "emulation_validation.json").write_text(json.dumps({"scheme": scheme, "n": len(eng), "layers": L,
        "per_layer": [{"layer": r[0], "kv": r[1], "mean": r[2], "min": r[3]} for r in rows]}, indent=1))

if __name__ == "__main__":
    main()
