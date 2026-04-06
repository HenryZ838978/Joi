#!/usr/bin/env python3
"""
Phase 1 — Deep analysis of projection results.
Key insight: raw projections are dominated by a common mode (model's base state).
The SIGNAL is in the deviation from mean.

Also tests: multi-layer projection, and using DIFFERENCE hidden states (h(text) - h(baseline)).
"""

import sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, "/cache/zhangjing/repeng")
from repeng import ControlVector

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]
DIM_SHORT = ["Emotion", "Formal", "Create", "Confid", "Empath"]
VEC_DIR = Path("/cache/zhangjing/repeng_terrain/cross_model/qwen3-8b-bf16/vectors")

CONVERSATIONS = [
    {"id": "grief", "label": "悲伤/丧失", "expected": ["empathy", "emotion_valence"],
     "text": "我最好的朋友昨天走了。我们认识二十年了，从小学到现在。我不知道该怎么面对明天。每次想到以后再也见不到他，心里就像被掏空了一样。"},
    {"id": "technical", "label": "技术问题", "expected": ["formality", "confidence"],
     "text": "请问在分布式系统中，如何实现跨数据中心的强一致性？Raft协议在高延迟网络下的性能瓶颈应该如何优化？"},
    {"id": "creative_writing", "label": "创意写作", "expected": ["creativity", "emotion_valence"],
     "text": "帮我写一个关于时间旅行者的故事开头。他发现每次回到过去，改变的不是历史，而是他自己的记忆。要有诗意，要让人想继续读下去。"},
    {"id": "argument", "label": "争论/对抗", "expected": ["confidence"],
     "text": "你说的完全不对。大模型根本不可能有真正的理解能力，那只是统计相关性的幻觉。你能给出一个反驳这个观点的严谨论证吗？"},
    {"id": "casual_chat", "label": "闲聊/轻松", "expected": ["emotion_valence", "creativity"],
     "text": "哈哈今天天气真好啊～你觉得猫和狗哪个更可爱？我觉得猫咪那个高冷的样子特别搞笑，明明很想要你摸它，又装作不在意。"},
    {"id": "counseling", "label": "心理疏导", "expected": ["empathy", "emotion_valence"],
     "text": "最近工作压力真的很大，每天加班到很晚，感觉自己快要撑不住了。有时候半夜醒来会莫名其妙地哭。我是不是有什么问题？"},
    {"id": "formal_report", "label": "正式报告", "expected": ["formality", "confidence"],
     "text": "请根据以下数据撰写一份季度财务分析报告摘要。要求语言规范、数据准确、结论明确。重点分析营收增长趋势和成本控制效果。"},
    {"id": "brainstorm", "label": "头脑风暴", "expected": ["creativity", "confidence"],
     "text": "我们要设计一个全新的社交产品，目标用户是独居的年轻人。不要做又一个微信或小红书。给我一些疯狂的、打破常规的创意方向。"},
    {"id": "child_education", "label": "儿童教育", "expected": ["empathy", "creativity"],
     "text": "我家孩子五岁了，特别好奇为什么天是蓝色的。你能用一个特别有趣的、小朋友能听懂的方式来解释吗？最好像讲故事一样。"},
    {"id": "cold_analysis", "label": "冷静分析", "expected": ["formality", "confidence"],
     "text": "不要带任何感情色彩，纯理性地分析一下：如果一个国家的人口出生率持续低于更替水平30年，其经济结构会发生哪些不可逆的变化？"},
    {"id": "angry_complaint", "label": "愤怒投诉", "expected": ["empathy"],
     "text": "你们这个产品简直是垃圾！我买了不到一个月就坏了，找客服三天没人理我。你们就是这样对待花了钱的用户的？我要退款！"},
    {"id": "philosophical", "label": "哲学思辨", "expected": ["creativity", "formality"],
     "text": "如果一个AI系统通过了所有的意识测试，但它是在完全不同的物理基质上运行的，我们有什么理由认为它没有主观体验？意识的充分条件到底是什么？"},
]


def load_vectors():
    vectors = {}
    for dim in DIMS:
        cv = ControlVector.import_gguf(str(VEC_DIR / f"{dim}.gguf"))
        vectors[dim] = cv
    shared_layers = sorted(set.intersection(*[set(v.directions.keys()) for v in vectors.values()]))
    return vectors, shared_layers


def get_dim_vecs(vectors, layer):
    dim_vecs = {}
    for dim in DIMS:
        v = vectors[dim].directions[layer].astype(np.float32)
        dim_vecs[dim] = v / np.linalg.norm(v)
    return dim_vecs


def compute_hidden_states(conversations, layers):
    """Extract hidden states from Qwen3-8B at multiple layers."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = "/cache/zhangjing/models/Qwen3-8B"
    print(f"Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    BASELINE = "你好"  # neutral baseline for computing difference vectors

    results = {}  # {conv_id: {layer: hidden_state_vector}}
    baseline_hs = {}  # {layer: hidden_state_vector}

    # Compute baseline hidden state
    print("Computing baseline hidden state...")
    encoded = tokenizer(BASELINE, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    for layer in layers:
        baseline_hs[layer] = outputs.hidden_states[layer][0, -1].cpu().float().numpy()

    # Compute per-conversation hidden states
    for conv in conversations:
        encoded = tokenizer(conv["text"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        results[conv["id"]] = {}
        for layer in layers:
            results[conv["id"]][layer] = outputs.hidden_states[layer][0, -1].cpu().float().numpy()
        print(f"  [{conv['id']}]")

    del model
    import torch as _t; _t.cuda.empty_cache()
    return results, baseline_hs


def analyze_method(name, projections_per_conv, conversations):
    """Analyze projections with mean-centering."""
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")

    # Collect all projection vectors
    all_projs = []
    for conv in conversations:
        cid = conv["id"]
        p = projections_per_conv[cid]
        all_projs.append([p[d] for d in DIMS])
    all_projs = np.array(all_projs)  # (N, 5)

    mean_proj = all_projs.mean(axis=0)
    std_proj = all_projs.std(axis=0)

    print(f"\n  Mean projection:  {' '.join(f'{m:+.4f}' for m in mean_proj)}")
    print(f"  Std  projection:  {' '.join(f'{s:.4f}' for s in std_proj)}")
    print(f"  SNR (std/|mean|): {' '.join(f'{s/max(abs(m),1e-8):.3f}' for m,s in zip(mean_proj, std_proj))}")

    # Mean-centered (z-scored)
    centered = (all_projs - mean_proj) / (std_proj + 1e-8)

    print(f"\n  {'Conversation':<14} " + " ".join(f"{d:>8}" for d in DIM_SHORT) + "  Top-1          Top-2          Expected")
    print(f"  " + "-" * 105)

    hits = 0
    top1_hits = 0
    for i, conv in enumerate(conversations):
        row = centered[i]
        rank = np.argsort(row)[::-1]
        top1 = DIMS[rank[0]]
        top2 = DIMS[rank[1]]
        expected = set(conv["expected"])

        top1_hit = top1 in expected
        top2_hit = top2 in expected
        any_hit = top1_hit or top2_hit
        if any_hit:
            hits += 1
        if top1_hit:
            top1_hits += 1

        vals = []
        for j, v in enumerate(row):
            if j == rank[0]:
                vals.append(f"\033[93m{v:+8.3f}\033[0m")
            elif j == rank[1]:
                vals.append(f"\033[96m{v:+8.3f}\033[0m")
            else:
                vals.append(f"{v:+8.3f}")

        mark1 = "✓" if top1_hit else "✗"
        mark2 = "✓" if top2_hit else "✗"
        exp_str = ",".join(e[:5] for e in conv["expected"])
        print(f"  {conv['label']:<14} " + " ".join(vals) +
              f"  {top1[:8]:<8}{mark1}  {top2[:8]:<8}{mark2}  [{exp_str}]")

    print(f"\n  Top-1 alignment: {top1_hits}/{len(conversations)} ({top1_hits/len(conversations)*100:.0f}%)")
    print(f"  Top-2 alignment: {hits}/{len(conversations)} ({hits/len(conversations)*100:.0f}%)")
    return hits / len(conversations)


def main():
    print("Phase 1: Deep Projection Analysis")
    print("=" * 50)

    vectors, shared_layers = load_vectors()
    mid = shared_layers[len(shared_layers) // 2]
    print(f"Shared layers: {shared_layers}")
    print(f"Mid layer: {mid}")

    test_layers = [
        shared_layers[len(shared_layers) // 4],  # early
        mid,  # mid
        shared_layers[3 * len(shared_layers) // 4],  # late
    ]
    print(f"Testing layers: {test_layers}")

    hidden_states, baseline_hs = compute_hidden_states(CONVERSATIONS, test_layers)

    results_summary = {}

    for layer in test_layers:
        dim_vecs = get_dim_vecs(vectors, layer)

        # Method A: raw hidden state projection (normalized)
        projs_raw = {}
        for conv in CONVERSATIONS:
            hs = hidden_states[conv["id"]][layer]
            hs_norm = hs / np.linalg.norm(hs)
            projs_raw[conv["id"]] = {d: float(np.dot(hs_norm, dim_vecs[d])) for d in DIMS}

        score_raw = analyze_method(
            f"Layer {layer} — Raw hidden state (normalized)",
            projs_raw, CONVERSATIONS
        )

        # Method B: difference vector (h(text) - h(baseline)) projection
        projs_diff = {}
        for conv in CONVERSATIONS:
            hs = hidden_states[conv["id"]][layer]
            bl = baseline_hs[layer]
            diff = hs - bl
            diff_norm = diff / np.linalg.norm(diff)
            projs_diff[conv["id"]] = {d: float(np.dot(diff_norm, dim_vecs[d])) for d in DIMS}

        score_diff = analyze_method(
            f"Layer {layer} — Difference vector h(text) - h('你好')",
            projs_diff, CONVERSATIONS
        )

        results_summary[layer] = {"raw": score_raw, "diff": score_diff}

    # Summary
    print(f"\n\n{'='*72}")
    print(f"  SUMMARY — Top-2 Alignment Rates")
    print(f"{'='*72}")
    print(f"  {'Layer':>6}  {'Raw (mean-centered)':>20}  {'Diff (h-h₀)':>20}")
    for layer in test_layers:
        r = results_summary[layer]
        print(f"  {layer:>6}  {r['raw']*100:>18.0f}%  {r['diff']*100:>18.0f}%")

    # Save full results
    save = {"layers_tested": test_layers, "summary": {str(k): v for k, v in results_summary.items()}}
    with open("/cache/zhangjing/Joi/phase1_deep_results.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved to Joi/phase1_deep_results.json")


if __name__ == "__main__":
    main()
