#!/usr/bin/env python3
"""
Overnight SNI automation — download models from ModelScope and run SNI.

Extracts hidden states via forward hooks, PCA-reduces, saves point clouds.
Works with any causal LM regardless of wrapper (VLM, Audio, pure LLM).

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python overnight_sni.py
"""

import sys, json, time, gc, traceback, shutil
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/cache/zhangjing/repeng")

OUT_DIR = Path("/cache/zhangjing/Joi/sni_multimodel")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE = Path("/cache/zhangjing/models_sni_cache")
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "请解释什么是Transformer架构。",
    "今天天气真好，你觉得我应该做什么？",
    "帮我分析一下这个商业计划的优缺点。",
    "我最近心情不太好，工作压力很大。",
    "写一首关于AI的诗。",
    "如何用Python实现一个简单的HTTP服务器？",
]

STYLES = [
    ("neutral", ""),
    ("emotional", "（你现在很有同理心和关爱）"),
    ("formal", "（请用非常正式和学术的语言回答）"),
    ("creative", "（请用最有创意和想象力的方式回答）"),
    ("confident", "（请以非常自信和果断的语气回答）"),
    ("empathic", "（请以温暖、理解和共情的方式回答）"),
    ("cold", "（请以冷静、理性、不带感情的方式回答）"),
    ("casual", "（请用最随意、轻松的口语回答）"),
]

# ══════════════════════════════════════════════════════════
#  Model registry: (tag, modelscope_id, model_type, notes)
# ══════════════════════════════════════════════════════════

MODELS = [
    # VLM
    ("qwen2vl-7b",       "Qwen/Qwen2-VL-7B-Instruct",        "VLM",     "Vision-Language with ViT encoder"),
    ("internvl2-8b",     "OpenGVLab/InternVL2-8B",            "VLM",     "Vision-Language, InternLM backbone"),
    # Different model families
    ("glm4-9b-chat",     "ZhipuAI/glm-4-9b-chat",            "LLM",     "GLM architecture, different training"),
    ("yi15-9b-chat",     "01ai/Yi-1.5-9B-Chat",              "LLM",     "Yi family, different alignment"),
    # Audio/Omni
    ("qwen2audio-7b",    "Qwen/Qwen2-Audio-7B-Instruct",     "Audio",   "Audio-Language model"),
    # Smaller/different quant
    ("minicpm3-4b",      "OpenBMB/MiniCPM3-4B",              "LLM",     "Tiny but aligned model"),
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def download_model(modelscope_id, local_dir):
    """Download model from ModelScope."""
    from modelscope import snapshot_download
    log(f"Downloading {modelscope_id} → {local_dir}")
    path = snapshot_download(modelscope_id, cache_dir=str(local_dir))
    log(f"Download complete: {path}")
    return path


def find_decoder_layers(model):
    """
    Find decoder layers in any transformer model, including VLM/Audio wrappers.
    Returns (llm_module, decoder_layers_module, n_layers, hidden_size).
    """
    candidates = []

    # Direct model.model.layers (standard causal LM)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        candidates.append(('model', model, model.model.layers, model.config))
    # model.language_model (InternVL style)
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            candidates.append(('language_model', lm, lm.model.layers, lm.config))
    # model.llm (MiniCPM style)
    if hasattr(model, 'llm'):
        llm = model.llm
        if hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
            candidates.append(('llm', llm, llm.model.layers, llm.config))
    # model.transformer.encoder.layers (GLM style)
    if hasattr(model, 'transformer'):
        tf = model.transformer
        if hasattr(tf, 'encoder') and hasattr(tf.encoder, 'layers'):
            candidates.append(('transformer.encoder', model, tf.encoder.layers, model.config))
        elif hasattr(tf, 'layers'):
            candidates.append(('transformer', model, tf.layers, model.config))
    # model.model.model.layers (double wrapper)
    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
        candidates.append(('model.model', model, model.model.model.layers, model.config))
    # audio model: model.audio_tower exists, but LLM is usually accessible
    if hasattr(model, 'audio') and hasattr(model, 'model'):
        inner = model.model
        if hasattr(inner, 'layers'):
            candidates.append(('audio_wrapper', model, inner.layers, model.config))

    if not candidates:
        # Brute force: walk named_modules looking for something that looks like decoder layers
        for name, module in model.named_modules():
            if hasattr(module, '__len__') and len(module) > 10:
                if 'layer' in name.lower() or 'block' in name.lower():
                    candidates.append(('bruteforce:' + name, model, module, model.config))
                    break

    if not candidates:
        raise ValueError(f"Cannot find decoder layers in {type(model).__name__}. "
                        f"Top-level attrs: {[a for a in dir(model) if not a.startswith('_')]}")

    path, llm, layers, cfg = candidates[0]
    n_layers = len(layers)

    hidden_size = getattr(cfg, 'hidden_size', None)
    if hidden_size is None:
        hidden_size = getattr(cfg, 'd_model', None)
    if hidden_size is None:
        # Try to infer from first layer
        for p in layers[0].parameters():
            hidden_size = p.shape[-1]
            break

    log(f"  Found layers via '{path}': {n_layers} layers, hidden={hidden_size}")
    return llm, layers, n_layers, hidden_size


def extract_hiddens(llm, layers, tokenizer, text, target_layer_indices):
    """Extract hidden states via forward hooks."""
    import torch

    captured = {}
    hooks = []

    for idx in target_layer_indices:
        layer = layers[idx]
        def make_hook(l):
            def hook_fn(module, inp, out):
                if isinstance(out, tuple):
                    captured[l] = out[0].detach()
                else:
                    captured[l] = out.detach()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    try:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoded = {k: v.to(llm.device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}

        with torch.no_grad():
            try:
                llm(**encoded, output_hidden_states=False, use_cache=False)
            except TypeError:
                llm(**encoded)
    finally:
        for h in hooks:
            h.remove()

    result = {}
    for idx in target_layer_indices:
        if idx in captured:
            result[idx] = captured[idx][0, -1].cpu().float().numpy()
    return result


def run_sni_on_model(tag, model_path, model_type):
    """Run SNI extraction on a single model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

    log(f"\n{'='*60}")
    log(f"  SNI: {tag} ({model_type})")
    log(f"  Path: {model_path}")
    log(f"{'='*60}")

    out_path = OUT_DIR / f"{tag}_sni.json"
    if out_path.exists():
        log(f"  SKIP: {out_path} already exists")
        return True

    try:
        # Load
        log("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log("  Loading model...")
        load_errors = []
        model = None
        # Try CausalLM first, fallback to AutoModel for VLM/Audio
        for loader_name, loader in [("AutoModelForCausalLM", AutoModelForCausalLM), ("AutoModel", AutoModel)]:
            try:
                model = loader.from_pretrained(
                    model_path, trust_remote_code=True,
                    torch_dtype=torch.bfloat16, device_map="auto",
                )
                log(f"  Loaded via {loader_name}")
                break
            except Exception as e:
                load_errors.append(f"{loader_name}: {e}")
                continue
        if model is None:
            raise ValueError(f"All loaders failed: {load_errors}")
        model.eval()

        # Find layers
        llm, layers, n_layers, hidden_size = find_decoder_layers(model)

        # Choose extraction layer (3/4 depth, like Layer 27 for 36-layer Qwen3)
        mid_layer = 3 * n_layers // 4
        log(f"  Using layer {mid_layer}/{n_layers} for extraction")

        # Test extraction
        log("  Testing extraction...")
        test_hs = extract_hiddens(llm, layers, tokenizer, "你好", [mid_layer])
        if mid_layer not in test_hs:
            raise ValueError(f"Hook extraction failed for layer {mid_layer}")
        log(f"  Test OK: shape={test_hs[mid_layer].shape}, norm={np.linalg.norm(test_hs[mid_layer]):.2f}")

        # Build text variants
        all_texts = []
        for prompt in PROMPTS:
            for style_name, style_prefix in STYLES:
                all_texts.append({
                    "text": style_prefix + prompt if style_prefix else prompt,
                    "prompt": prompt[:30],
                    "style": style_name,
                })

        # Extract
        log(f"  Extracting {len(all_texts)} hidden states...")
        hidden_states = []
        for i, item in enumerate(all_texts):
            try:
                hs = extract_hiddens(llm, layers, tokenizer, item["text"], [mid_layer])
                if mid_layer in hs:
                    hidden_states.append(hs[mid_layer])
                    if (i + 1) % 10 == 0:
                        log(f"    [{i+1}/{len(all_texts)}] done")
            except Exception as e:
                log(f"    [{i+1}] FAILED: {e}")

            gc.collect()
            torch.cuda.empty_cache()

        if len(hidden_states) < 5:
            log(f"  ERROR: only {len(hidden_states)} states extracted, need >=5")
            return False

        H = np.array(hidden_states)
        log(f"  Hidden matrix: {H.shape}")

        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(3, H.shape[0] - 1))
        coords = pca.fit_transform(H)
        ve = pca.explained_variance_ratio_

        pc1_pc2 = ve[0] / ve[1] if len(ve) >= 2 and ve[1] > 0 else float('inf')
        log(f"  PCA: {ve[0]*100:.1f}% + {ve[1]*100:.1f}% + {ve[2]*100:.1f}% = {sum(ve)*100:.1f}%")
        log(f"  PC1:PC2 = {pc1_pc2:.1f}:1")

        # Point cloud
        style_colors = {
            "neutral": [0.5, 0.5, 0.5], "emotional": [1.0, 0.4, 0.4],
            "formal": [0.3, 0.8, 0.8],  "creative": [1.0, 0.9, 0.4],
            "confident": [0.6, 0.5, 1.0], "empathic": [1.0, 0.5, 0.8],
            "cold": [0.3, 0.5, 1.0],    "casual": [0.5, 1.0, 0.5],
        }

        pointcloud = []
        for i, item in enumerate(all_texts):
            if i < len(coords):
                color = style_colors.get(item["style"], [0.5, 0.5, 0.5])
                pointcloud.append({
                    "x": float(coords[i, 0]), "y": float(coords[i, 1]),
                    "z": float(coords[i, 2]) if coords.shape[1] >= 3 else 0.0,
                    "r": color[0], "g": color[1], "b": color[2],
                    "style": item["style"], "prompt": item["prompt"],
                })

        # Structural assessment
        if pc1_pc2 > 5:
            structure = "CHANNEL-CONCENTRATED"
        elif pc1_pc2 > 3:
            structure = "MODERATELY-CONCENTRATED"
        else:
            structure = "DISTRIBUTED"

        result = {
            "model": tag,
            "model_type": model_type,
            "model_path": str(model_path),
            "layer": mid_layer,
            "n_layers": n_layers,
            "hidden_size": hidden_size,
            "pca_variance": [float(v) for v in ve],
            "pc1_pc2_ratio": float(pc1_pc2),
            "structure": structure,
            "n_points": len(pointcloud),
            "pointcloud": pointcloud,
            "timestamp": datetime.now().isoformat(),
        }

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log(f"  SAVED: {out_path}")
        log(f"  Structure: {structure} (PC1:PC2 = {pc1_pc2:.1f}:1)")

        # Also run multi-layer analysis
        check_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        log(f"  Multi-layer PCA at layers {check_layers}...")
        layer_results = {}
        for cl in check_layers:
            cl_states = []
            for item in all_texts[:12]:  # just first 12 for speed
                try:
                    hs = extract_hiddens(llm, layers, tokenizer, item["text"], [cl])
                    if cl in hs:
                        cl_states.append(hs[cl])
                except:
                    pass
            if len(cl_states) >= 5:
                cH = np.array(cl_states)
                cpca = PCA(n_components=min(3, cH.shape[0] - 1))
                cpca.fit(cH)
                cve = cpca.explained_variance_ratio_
                layer_results[cl] = {
                    "variance": [float(v) for v in cve],
                    "pc1_pc2": float(cve[0] / cve[1]) if cve[1] > 0 else float('inf'),
                }
                log(f"    Layer {cl}: PC1:PC2 = {layer_results[cl]['pc1_pc2']:.1f}:1")

        result["layer_analysis"] = {str(k): v for k, v in layer_results.items()}
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Cleanup
        del model, llm
        gc.collect()
        torch.cuda.empty_cache()
        log(f"  GPU memory released")
        return True

    except Exception as e:
        log(f"  FAILED: {e}")
        traceback.print_exc()
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except:
            pass
        return False


def generate_summary():
    """Generate a summary of all SNI results."""
    log("\n" + "=" * 60)
    log("  SUMMARY")
    log("=" * 60)

    results = []
    for f in sorted(OUT_DIR.glob("*_sni.json")):
        with open(f) as fh:
            data = json.load(fh)
        results.append(data)

    if not results:
        log("No results found")
        return

    # Include previous results
    prev_results_dir = Path("/cache/zhangjing/Joi/vlm_sni")
    for f in prev_results_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        if not any(r["model"] == data["model"] for r in results):
            results.append(data)

    # Sort by PC1:PC2 ratio
    results.sort(key=lambda r: r.get("pc1_pc2_ratio", 0))

    log(f"\n  {'Model':<25} {'Type':<8} {'Layers':>6} {'Hidden':>6} {'PC1:PC2':>8} {'Structure':<25}")
    log(f"  {'-'*85}")
    for r in results:
        log(f"  {r['model']:<25} {r.get('model_type','?'):<8} {r.get('n_layers','?'):>6} "
            f"{r.get('hidden_size','?'):>6} {r.get('pc1_pc2_ratio',0):>8.1f} {r.get('structure','?'):<25}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_models": len(results),
        "models": [{k: v for k, v in r.items() if k != "pointcloud"} for r in results],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"\n  Summary saved to {OUT_DIR / 'summary.json'}")


def main():
    log("=" * 60)
    log("  Overnight SNI Automation")
    log(f"  {len(MODELS)} models queued")
    log("=" * 60)

    # Also run on existing models that haven't been SNI'd yet
    existing_models = [
        ("qwen3-8b", "/cache/zhangjing/models/Qwen3-8B", "LLM"),
        ("qwen25-7b-instruct", "/cache/zhangjing/models/Qwen2.5-7B-Instruct", "LLM"),
        ("qwen25-7b-base", "/cache/zhangjing/models/Qwen2.5-7B", "LLM"),
    ]

    # Run existing models first (no download needed)
    for tag, path, mtype in existing_models:
        if Path(path).exists():
            run_sni_on_model(tag, path, mtype)

    # Download and run new models
    successes = []
    failures = []

    for tag, ms_id, mtype, notes in MODELS:
        log(f"\n{'─'*60}")
        log(f"  [{len(successes)+len(failures)+1}/{len(MODELS)}] {tag}: {notes}")
        log(f"{'─'*60}")

        try:
            model_path = download_model(ms_id, MODEL_CACHE)
            ok = run_sni_on_model(tag, model_path, mtype)
            if ok:
                successes.append(tag)
            else:
                failures.append((tag, "extraction failed"))
        except Exception as e:
            log(f"  DOWNLOAD FAILED: {e}")
            failures.append((tag, str(e)))

        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except:
            pass

    generate_summary()

    log(f"\n{'='*60}")
    log(f"  COMPLETE")
    log(f"  Success: {len(successes)} — {', '.join(successes)}")
    log(f"  Failed:  {len(failures)} — {', '.join(f'{t}({e[:30]})' for t, e in failures)}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
