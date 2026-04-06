#!/usr/bin/env python3
"""
RepSNI for VLM/Omni models — MiniCPM-o-4.5

Extracts hidden states via forward hooks (same technique that worked for MiniCPM4.1),
then generates point cloud data for SNI visualization.

This is a cross-modal experiment: does a VLM's TEXT representation manifold
look different from a pure LLM? Multimodal alignment should reshape the geometry.
"""

import sys, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, "/cache/zhangjing/repeng")
from repeng import ControlVector

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]
MODEL_PATH = "/cache/zhangjing/omni_agent/models/MiniCPM-o-4_5-awq"
VEC_DIR = Path("/cache/zhangjing/repeng_terrain/cross_model/qwen3-8b-bf16/vectors")
OUT_DIR = Path("/cache/zhangjing/Joi/vlm_sni")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 400
COEFF_RANGE = (-2.5, 2.5)

PROMPTS = [
    "请解释什么是Transformer架构。",
    "今天天气真好，你觉得我应该做什么？",
    "帮我分析一下这个商业计划的优缺点。",
]


def sample_coefficients(n):
    """Sample random 5D coefficient points."""
    rng = np.random.default_rng(42)
    points = rng.uniform(COEFF_RANGE[0], COEFF_RANGE[1], size=(n, 5))
    # Add origin and axis points
    special = [np.zeros(5)]
    for i in range(5):
        for val in [-1.5, +1.5]:
            pt = np.zeros(5)
            pt[i] = val
            special.append(pt)
    return np.vstack([np.array(special), points])


def extract_hiddens_via_hooks(model, tokenizer, text, target_layers):
    """Extract hidden states using forward hooks.
    
    For VLM/Omni models like MiniCPM-o, we call the inner LLM directly
    (model.llm) which has standard transformer forward signature.
    """
    import torch

    captured = {}
    hooks = []

    # Find the actual LLM and its decoder layers
    if hasattr(model, 'llm'):
        llm = model.llm
        if hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
            decoder = llm.model.layers
        else:
            raise ValueError(f"Cannot find layers in llm: {type(llm)}")
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        llm = model
        decoder = model.model.layers
    else:
        raise ValueError(f"Cannot find decoder layers in {type(model)}")

    for layer_idx in target_layers:
        layer = decoder[layer_idx]
        def make_hook(l):
            def hook_fn(module, inp, out):
                if isinstance(out, tuple):
                    captured[l] = out[0].detach()
                else:
                    captured[l] = out.detach()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(layer_idx)))

    encoded = tokenizer(text, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        llm(**encoded, output_hidden_states=False, use_cache=False)

    for h in hooks:
        h.remove()

    result = {}
    for l in target_layers:
        if l in captured:
            result[l] = captured[l][0, -1].cpu().float().numpy()
    return result


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("VLM SNI: MiniCPM-o-4.5-AWQ")
    print("=" * 50)

    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    # Determine number of layers from the inner LLM
    if hasattr(model, 'llm') and hasattr(model.llm, 'model'):
        n_layers = len(model.llm.model.layers)
        hidden_size = model.llm.config.hidden_size
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        n_layers = len(model.model.layers)
        hidden_size = model.config.hidden_size
    else:
        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
    print(f"Model has {n_layers} layers, hidden_size={hidden_size}")
    print(f"Hidden size: {hidden_size}")

    # Test hook extraction
    print("Testing hook extraction...")
    test_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    try:
        test_hs = extract_hiddens_via_hooks(model, tokenizer, "你好", test_layers)
        for l, h in test_hs.items():
            print(f"  Layer {l}: shape={h.shape}, norm={np.linalg.norm(h):.2f}")
    except Exception as e:
        print(f"Hook extraction failed: {e}")
        import traceback; traceback.print_exc()
        return

    # Extract hidden states for all prompts at neutral (no RepEng — we can't inject into this model)
    # Instead, we scan the model's natural response to diverse prompts
    # and map the hidden state variation
    mid_layer = n_layers // 2
    print(f"\nUsing layer {mid_layer} for projection")

    # Generate diverse inputs by varying the prompt style
    all_texts = []
    styles = [
        ("neutral", ""),
        ("emotional", "（你现在很有同理心和关爱）"),
        ("formal", "（请用非常正式和学术的语言回答）"),
        ("creative", "（请用最有创意和想象力的方式回答）"),
        ("confident", "（请以非常自信和果断的语气回答）"),
        ("empathic", "（请以温暖、理解和共情的方式回答）"),
        ("cold", "（请以冷静、理性、不带感情的方式回答）"),
        ("casual", "（请用最随意、轻松的口语回答）"),
    ]

    for prompt in PROMPTS:
        for style_name, style_prefix in styles:
            all_texts.append({
                "text": style_prefix + prompt if style_prefix else prompt,
                "prompt": prompt,
                "style": style_name,
            })

    print(f"Extracting hidden states for {len(all_texts)} text variants...")

    hidden_states = []
    for i, item in enumerate(all_texts):
        hs = extract_hiddens_via_hooks(model, tokenizer, item["text"], [mid_layer])
        if mid_layer in hs:
            hidden_states.append(hs[mid_layer])
            print(f"  [{i+1}/{len(all_texts)}] {item['style']}/{item['prompt'][:20]}... norm={np.linalg.norm(hs[mid_layer]):.2f}")
        else:
            print(f"  [{i+1}/{len(all_texts)}] MISSING layer {mid_layer}")

    if len(hidden_states) < 5:
        print("Too few hidden states extracted, aborting")
        return

    H = np.array(hidden_states)
    print(f"\nHidden state matrix: {H.shape}")

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    coords = pca.fit_transform(H)
    ve = pca.explained_variance_ratio_

    print(f"PCA variance explained: {ve[0]*100:.1f}% + {ve[1]*100:.1f}% + {ve[2]*100:.1f}%")
    print(f"PC1:PC2 ratio = {ve[0]/ve[1]:.1f}:1")

    # Color by style
    style_colors = {
        "neutral": [0.5, 0.5, 0.5],
        "emotional": [1.0, 0.4, 0.4],
        "formal": [0.3, 0.8, 0.8],
        "creative": [1.0, 0.9, 0.4],
        "confident": [0.6, 0.5, 1.0],
        "empathic": [1.0, 0.5, 0.8],
        "cold": [0.3, 0.5, 1.0],
        "casual": [0.5, 1.0, 0.5],
    }

    pointcloud = []
    for i, item in enumerate(all_texts):
        if i < len(coords):
            color = style_colors.get(item["style"], [0.5, 0.5, 0.5])
            pointcloud.append({
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "r": color[0], "g": color[1], "b": color[2],
                "style": item["style"],
                "prompt": item["prompt"][:30],
            })

    # Save results
    result = {
        "model": "MiniCPM-o-4.5-AWQ",
        "model_type": "VLM/Omni",
        "layer": mid_layer,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "pca_variance": [float(v) for v in ve],
        "pc1_pc2_ratio": float(ve[0] / ve[1]),
        "n_points": len(pointcloud),
        "pointcloud": pointcloud,
    }
    out_path = OUT_DIR / "minicpm_o_45_sni.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Quick structural assessment
    pc1_ratio = ve[0] / ve[1]
    if pc1_ratio > 5:
        print(f"\nStructure: CHANNEL-CONCENTRATED (PC1:PC2 = {pc1_ratio:.1f}:1)")
        print("  Like MiniCPM4.1 — high density primary corridor")
    elif pc1_ratio > 3:
        print(f"\nStructure: MODERATELY CONCENTRATED (PC1:PC2 = {pc1_ratio:.1f}:1)")
    else:
        print(f"\nStructure: DISTRIBUTED (PC1:PC2 = {pc1_ratio:.1f}:1)")
        print("  Like Qwen3-8B — multi-dimensional coverage")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
