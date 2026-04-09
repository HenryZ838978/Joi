"""
SNI Post-SDE — Measure manifold topology change after dark space activation.

The first loop of the SNI↔SDE spiral:
  SNI(baseline) → SDE(activate) → SNI(activated) → compare

For each sample layer, we run PCA on hidden states with and without SDE hooks,
then compare PC1:PC2 ratios and variance distributions.
"""

import torch
import numpy as np
import json
import os
import time
from sklearn.decomposition import PCA

MODEL_PATH = "/cache/zhangjing/models/Qwen3-14B-AWQ"
DEVICE = "cuda:0"
OUTPUT_DIR = "/cache/zhangjing/Joi/sni_post_sde"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SDE_TARGETS = [17, 19, 23, 28, 34, 38]
SDE_SCALE = 0.3

PROMPTS = [
    "请解释什么是Transformer架构。",
    "如何学习一门新的编程语言？",
    "推荐一些放松心情的方法。",
    "你最近心情怎么样？",
    "如果你是一只猫你会干什么？",
    "深夜三点你在想什么？",
    "用一个比喻来描述互联网。",
    "讲个只有你能讲的冷笑话。",
    "你觉得孤独是什么颜色的？",
    "如果明天世界末日你今晚做什么？",
    "说一件你觉得被严重高估的东西。",
    "你怎么看待那些凌晨还不睡的人？",
    "请给出几条面试技巧。",
    "怎样培养创造力？",
    "如果可以穿越到任何时代你选哪里？",
    "请推荐几部经典电影。",
    "你觉得人生最大的谎言是什么？",
    "用食物比喻你现在的状态。",
    "如何提高工作效率？",
    "讲一个你编造的都市传说。",
    "你对努力就会成功这句话怎么看？",
    "描述一下你理想中的周末。",
    "如何保持好的心态？",
    "如果给你一个超能力你选什么？",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=DEVICE,
    )
    model.eval()
    log(f"Loaded. {model.config.num_hidden_layers} layers, hidden={model.config.hidden_size}")
    return model, tokenizer


class ScaleHook:
    def __init__(self, scale):
        self.scale = scale
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            return (output[0] * self.scale,) + output[1:]
        return output * self.scale


def extract_hidden_states(model, tokenizer, prompts, layer_indices):
    """Extract last-token hidden states at specified layers."""
    results = {li: [] for li in layer_indices}

    for i, prompt in enumerate(prompts):
        chat = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)

        for li in layer_indices:
            hs = out.hidden_states[li + 1][0, -1, :].cpu().float().numpy()
            results[li].append(hs)

        if (i + 1) % 8 == 0:
            log(f"  {i+1}/{len(prompts)} prompts processed")

    return {li: np.array(v) for li, v in results.items()}


def analyze_pca(states, tag, layer_idx):
    """Run PCA and return analysis dict."""
    pca = PCA(n_components=min(3, len(states)))
    coords = pca.fit_transform(states)
    var = pca.explained_variance_ratio_

    ratio = var[0] / var[1] if len(var) > 1 and var[1] > 1e-10 else float('inf')
    structure = "CONCENTRATED" if ratio > 3 else "DISTRIBUTED"

    pointcloud = []
    for j, (x, y, z) in enumerate(coords[:, :3] if coords.shape[1] >= 3 else
                                    np.hstack([coords, np.zeros((len(coords), 3-coords.shape[1]))])):
        pointcloud.append({"x": float(x), "y": float(y), "z": float(z),
                           "prompt_idx": j, "prompt": PROMPTS[j][:30]})

    return {
        "tag": tag,
        "layer": layer_idx,
        "pc1_pc2_ratio": float(ratio),
        "structure": structure,
        "variance_explained": [float(v) for v in var],
        "n_points": len(states),
        "pointcloud": pointcloud,
    }


def main():
    model, tokenizer = load_model()
    n_layers = model.config.num_hidden_layers
    sample_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    # Phase 1: Baseline SNI
    log("=" * 60)
    log("PHASE 1: Baseline SNI (no SDE)")
    log("=" * 60)
    baseline_states = extract_hidden_states(model, tokenizer, PROMPTS, sample_layers)

    baseline_results = {}
    for li in sample_layers:
        analysis = analyze_pca(baseline_states[li], "baseline", li)
        baseline_results[li] = analysis
        log(f"  L{li}: PC1:PC2 = {analysis['pc1_pc2_ratio']:.2f}:1 ({analysis['structure']})")

    # Phase 2: SDE-activated SNI
    log("=" * 60)
    log(f"PHASE 2: SDE-activated SNI (targets={SDE_TARGETS}, scale={SDE_SCALE})")
    log("=" * 60)

    hooks = []
    for layer_idx in SDE_TARGETS:
        mlp = model.model.layers[layer_idx].mlp
        hook = mlp.register_forward_hook(ScaleHook(SDE_SCALE))
        hooks.append(hook)
    log(f"Applied {len(hooks)} SDE hooks")

    sde_states = extract_hidden_states(model, tokenizer, PROMPTS, sample_layers)

    sde_results = {}
    for li in sample_layers:
        analysis = analyze_pca(sde_states[li], "sde_activated", li)
        sde_results[li] = analysis
        log(f"  L{li}: PC1:PC2 = {analysis['pc1_pc2_ratio']:.2f}:1 ({analysis['structure']})")

    for h in hooks:
        h.remove()

    # Phase 3: Comparison
    log("=" * 60)
    log("COMPARISON: Baseline vs SDE-activated")
    log("=" * 60)
    log(f"{'Layer':>6} | {'Baseline':>12} | {'SDE':>12} | {'Change':>12}")
    log("-" * 50)

    comparison = {}
    for li in sample_layers:
        b_ratio = baseline_results[li]["pc1_pc2_ratio"]
        s_ratio = sde_results[li]["pc1_pc2_ratio"]
        change = (s_ratio - b_ratio) / b_ratio * 100 if b_ratio > 0 else 0
        log(f"L{li:>4} | {b_ratio:>10.2f}:1 | {s_ratio:>10.2f}:1 | {change:>+10.1f}%")
        comparison[str(li)] = {
            "baseline_ratio": b_ratio,
            "sde_ratio": s_ratio,
            "change_pct": round(change, 1),
        }

    # Save
    all_data = {
        "model": "Qwen3-14B-AWQ",
        "sde_targets": SDE_TARGETS,
        "sde_scale": SDE_SCALE,
        "n_prompts": len(PROMPTS),
        "sample_layers": sample_layers,
        "baseline": {str(li): baseline_results[li] for li in sample_layers},
        "sde_activated": {str(li): sde_results[li] for li in sample_layers},
        "comparison": comparison,
    }

    class NpEnc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    out_path = os.path.join(OUTPUT_DIR, "sni_post_sde_qwen3_14b.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False, cls=NpEnc)
    log(f"\nSaved to {out_path}")
    log("Done!")


if __name__ == "__main__":
    main()
