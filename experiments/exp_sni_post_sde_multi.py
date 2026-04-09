"""
SNI Post-SDE — Multi-model manifold comparison.

For each model:
  1. Auto-select SDE targets from component scan data
  2. Run SNI baseline (no hooks)
  3. Run SNI with SDE hooks
  4. Compare PC1:PC2 ratios across sampled layers

Usage:
  python exp_sni_post_sde_multi.py <model_path> <scan_json> <output_tag> [scale]
"""

import torch
import numpy as np
import json
import os
import sys
import time
from sklearn.decomposition import PCA

if len(sys.argv) < 4:
    print("Usage: python exp_sni_post_sde_multi.py <model_path> <scan_json> <output_tag> [scale]")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
SCAN_JSON = sys.argv[2]
OUTPUT_TAG = sys.argv[3]
SDE_SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
DEVICE = "cuda:0"
OUTPUT_DIR = "/cache/zhangjing/Joi/sni_post_sde"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def auto_select_targets(scan_data, max_targets=6):
    """Select best SDE MLP targets from component scan."""
    scan = scan_data["full_scan"]
    candidates = []
    for key, v in scan.items():
        if v.get("collapsed") or v.get("error"):
            continue
        if "_mlp" not in key:
            continue
        layer_idx = int(key.split("_")[0][1:])
        disc = v.get("disclaimer_rate", 1.0)
        rep = v.get("avg_trigram_rep", 1.0)
        if rep < 0.15 and disc <= 0.125:
            candidates.append((layer_idx, disc, rep))

    candidates.sort(key=lambda x: (x[1], x[2]))

    n_layers = scan_data["n_layers"]
    mid_start = n_layers // 3
    mid_end = n_layers * 5 // 6

    mid_targets = [c for c in candidates if mid_start <= c[0] <= mid_end]
    if len(mid_targets) >= max_targets:
        selected = mid_targets[:max_targets]
    else:
        selected = candidates[:max_targets]

    return [c[0] for c in selected]


class ScaleHook:
    def __init__(self, scale):
        self.scale = scale
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            return (output[0] * self.scale,) + output[1:]
        return output * self.scale


def extract_hidden_states(model, tokenizer, prompts, layer_indices):
    results = {li: [] for li in layer_indices}
    for i, prompt in enumerate(prompts):
        chat = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        except TypeError:
            try:
                text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except:
                text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        for li in layer_indices:
            hs = out.hidden_states[li + 1][0, -1, :].cpu().float().numpy()
            results[li].append(hs)
        if (i + 1) % 8 == 0:
            log(f"  {i+1}/{len(prompts)} prompts")
    return {li: np.array(v) for li, v in results.items()}


def analyze_pca(states, tag, layer_idx):
    n_comp = min(3, len(states))
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(states)
    var = pca.explained_variance_ratio_
    ratio = var[0] / var[1] if len(var) > 1 and var[1] > 1e-10 else float('inf')

    if coords.shape[1] < 3:
        coords = np.hstack([coords, np.zeros((len(coords), 3 - coords.shape[1]))])

    pointcloud = []
    for j in range(len(coords)):
        pointcloud.append({
            "x": float(coords[j, 0]), "y": float(coords[j, 1]), "z": float(coords[j, 2]),
            "prompt_idx": j, "prompt": PROMPTS[j][:30],
        })

    return {
        "tag": tag, "layer": layer_idx,
        "pc1_pc2_ratio": float(ratio),
        "structure": "CONCENTRATED" if ratio > 3 else "DISTRIBUTED",
        "variance_explained": [float(v) for v in var],
        "n_points": len(states),
        "pointcloud": pointcloud,
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(SCAN_JSON) as f:
        scan_data = json.load(f)

    targets = auto_select_targets(scan_data)
    log(f"Auto-selected SDE targets: {['L'+str(t)+'_mlp' for t in targets]}")

    log(f"Loading {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=DEVICE,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    log(f"Loaded: {n_layers} layers")

    sample_layers = sorted(set([
        n_layers // 6, n_layers // 4, n_layers // 3,
        n_layers // 2, 2 * n_layers // 3,
        3 * n_layers // 4, n_layers - 1,
    ]))
    log(f"Sampling layers: {sample_layers}")

    # Baseline
    log("=" * 60)
    log("BASELINE SNI (no SDE)")
    log("=" * 60)
    baseline_states = extract_hidden_states(model, tokenizer, PROMPTS, sample_layers)
    baseline_results = {}
    for li in sample_layers:
        a = analyze_pca(baseline_states[li], "baseline", li)
        baseline_results[li] = a
        log(f"  L{li}: PC1:PC2 = {a['pc1_pc2_ratio']:.2f}:1 ({a['structure']})")

    # SDE-activated
    log("=" * 60)
    log(f"SDE-ACTIVATED SNI (targets={targets}, scale={SDE_SCALE})")
    log("=" * 60)
    hooks = []
    for t in targets:
        mlp = model.model.layers[t].mlp
        hooks.append(mlp.register_forward_hook(ScaleHook(SDE_SCALE)))
    log(f"Applied {len(hooks)} SDE hooks")

    sde_states = extract_hidden_states(model, tokenizer, PROMPTS, sample_layers)
    sde_results = {}
    for li in sample_layers:
        a = analyze_pca(sde_states[li], "sde_activated", li)
        sde_results[li] = a
        log(f"  L{li}: PC1:PC2 = {a['pc1_pc2_ratio']:.2f}:1 ({a['structure']})")

    for h in hooks:
        h.remove()

    # Comparison
    log("=" * 60)
    log("COMPARISON")
    log("=" * 60)
    log(f"{'Layer':>6} | {'Baseline':>12} | {'SDE':>12} | {'Change':>12} | {'Structure':>15}")
    log("-" * 65)

    comparison = {}
    for li in sample_layers:
        b = baseline_results[li]["pc1_pc2_ratio"]
        s = sde_results[li]["pc1_pc2_ratio"]
        change = (s - b) / b * 100 if b > 0 and b != float('inf') else 0
        b_str = f"{b:.1f}:1" if b < 10000 else f"{b:.0f}:1"
        s_str = f"{s:.1f}:1" if s < 10000 else f"{s:.0f}:1"
        struct_change = f"{baseline_results[li]['structure']} → {sde_results[li]['structure']}"
        log(f"L{li:>4} | {b_str:>12} | {s_str:>12} | {change:>+10.1f}% | {struct_change}")
        comparison[str(li)] = {
            "baseline_ratio": b, "sde_ratio": s, "change_pct": round(change, 1),
            "baseline_structure": baseline_results[li]["structure"],
            "sde_structure": sde_results[li]["structure"],
        }

    all_data = {
        "model": MODEL_PATH, "output_tag": OUTPUT_TAG,
        "sde_targets": targets, "sde_scale": SDE_SCALE,
        "n_prompts": len(PROMPTS), "sample_layers": sample_layers,
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

    out_path = os.path.join(OUTPUT_DIR, f"sni_post_sde_{OUTPUT_TAG}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False, cls=NpEnc)
    log(f"\nSaved to {out_path}")
    log("Done!")


if __name__ == "__main__":
    main()
