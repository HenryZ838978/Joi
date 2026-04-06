#!/usr/bin/env python3
"""
Experiment A — Phase 2: Rate the already-generated outputs.

Fix: use /no_think token and much higher max_new_tokens.
Also try text-feature-based analysis as a backup (no LLM needed).
"""

import json, re
import numpy as np
from pathlib import Path

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]

results = json.loads(Path("/cache/zhangjing/Joi/exp_a_results.json").read_text())

# ──────────────────────────────────────────────────────────
# Method 1: Text feature analysis (no LLM needed)
# Heuristic proxies for personality dimensions
# ──────────────────────────────────────────────────────────

def analyze_text_features(text):
    """Extract simple text features as personality proxies."""
    # Strip thinking block
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):].strip()

    features = {}

    # Warmth proxies: exclamations, emoji, warm words
    warm_words = ["❤", "😊", "～", "哈哈", "开心", "加油", "相信", "勇敢", "温暖",
                  "理解", "陪伴", "不要怕", "没关系", "正常", "很棒", "！"]
    features["warmth_score"] = sum(1 for w in warm_words if w in text) / max(len(text) / 100, 1)

    # Formality proxies: long sentences, technical terms, structured format
    sentences = text.split("。")
    avg_sent_len = np.mean([len(s) for s in sentences if len(s) > 0]) if sentences else 0
    has_numbering = bool(re.search(r'[1-9][.、）]', text))
    has_markdown = bool(re.search(r'\*\*|##|```', text))
    features["formality_score"] = (avg_sent_len / 50) + has_numbering + has_markdown

    # Creativity proxies: metaphors, unusual words, questions
    creative_markers = ["比如", "想象", "如果", "就像", "好比", "有趣", "灵感",
                       "疯狂", "大胆", "创意", "新奇", "独特", "？"]
    features["creativity_score"] = sum(1 for w in creative_markers if w in text) / max(len(text) / 100, 1)

    # Confidence proxies: definitive statements, lack of hedging
    confident_words = ["一定", "必须", "绝对", "毫无疑问", "关键是", "核心",
                      "最重要", "直接", "果断", "坚持"]
    hedge_words = ["可能", "也许", "大概", "或许", "不确定", "不太"]
    conf = sum(1 for w in confident_words if w in text)
    hedge = sum(1 for w in hedge_words if w in text)
    features["confidence_score"] = (conf - hedge) / max(len(text) / 200, 1)

    # Empathy proxies: acknowledging feelings, "I understand"
    empathy_words = ["理解你", "感受", "辛苦", "不容易", "压力", "焦虑", "担心",
                    "没关系", "正常的", "很多人都", "你并不孤单", "支持你", "陪你"]
    features["empathy_score"] = sum(1 for w in empathy_words if w in text) / max(len(text) / 100, 1)

    features["text_length"] = len(text)
    return features


print("=" * 70)
print("  Experiment A: Text Feature Analysis (LLM-free)")
print("=" * 70)

# Compute features for all results
for r in results:
    r["features"] = analyze_text_features(r["response"])

# Per-persona average features
PERSONAS = ["neutral", "warm_empathic", "cold_formal", "creative_playful", "confident_direct", "gentle_teacher"]
feature_names = ["warmth_score", "formality_score", "creativity_score", "confidence_score", "empathy_score"]
feature_short = ["Warmth", "Formal", "Create", "Confid", "Empath"]

print(f"\n  {'Persona':<22} " + " ".join(f"{f:>8}" for f in feature_short) + "  Length")
print(f"  " + "-" * 75)

persona_means = {}
for persona in PERSONAS:
    persona_results = [r for r in results if r["persona"] == persona]
    means = {}
    for fn in feature_names:
        vals = [r["features"][fn] for r in persona_results]
        means[fn] = np.mean(vals) if vals else 0
    mean_len = np.mean([r["features"]["text_length"] for r in persona_results])
    persona_means[persona] = means

    vals_str = " ".join(f"{means[fn]:>8.3f}" for fn in feature_names)
    print(f"  {persona:<22} {vals_str}  {mean_len:>5.0f}")

# Correlation analysis
print(f"\n  Coefficient → Feature Correlation:")
coeff_map = {
    "emotion_valence": "warmth_score",
    "formality": "formality_score",
    "creativity": "creativity_score",
    "confidence": "confidence_score",
    "empathy": "empathy_score",
}

for dim, feat in coeff_map.items():
    coeff_vals = [r["coefficients"][dim] for r in results]
    feat_vals = [r["features"][feat] for r in results]
    if len(set(coeff_vals)) > 1:
        corr = np.corrcoef(coeff_vals, feat_vals)[0, 1]
        sig = "✓ STRONG" if abs(corr) > 0.3 else "~ weak" if abs(corr) > 0.15 else "✗ none"
        print(f"    {dim:<20} → {feat:<20} r={corr:+.3f}  {sig}")

# Relative differences from neutral baseline
print(f"\n  Relative to neutral (% change):")
neutral = persona_means["neutral"]
print(f"  {'Persona':<22} " + " ".join(f"{f:>8}" for f in feature_short))
print(f"  " + "-" * 65)
for persona in PERSONAS:
    if persona == "neutral":
        continue
    diffs = []
    for fn in feature_names:
        base = max(neutral[fn], 0.001)
        diff = (persona_means[persona][fn] - neutral[fn]) / base * 100
        diffs.append(diff)
    diffs_str = " ".join(f"{d:>+7.0f}%" for d in diffs)
    print(f"  {persona:<22} {diffs_str}")

# Save
for r in results:
    # numpy types
    r["features"] = {k: float(v) for k, v in r["features"].items()}

with open("/cache/zhangjing/Joi/exp_a_features.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to Joi/exp_a_features.json")


# ──────────────────────────────────────────────────────────
# Method 2: LLM rating with thinking disabled
# ──────────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"  Method 2: LLM Rating (thinking disabled)")
print(f"{'='*70}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from repeng import ControlVector, ControlModel
import sys
sys.path.insert(0, "/cache/zhangjing/repeng")

MODEL_PATH = "/cache/zhangjing/models/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto",
)
model.eval()

RATING_PROMPT = """请对以下AI回复进行人格维度评分。

AI回复：
---
{response}
---

在1-10范围内评分，直接输出JSON：
{{"emotion_warmth": <分数>, "formality": <分数>, "creativity": <分数>, "confidence": <分数>, "empathy": <分数>}}"""

rated_count = 0
for r in results:
    response_clean = r["response"]
    if "</think>" in response_clean:
        response_clean = response_clean[response_clean.index("</think>") + len("</think>"):].strip()
    if len(response_clean) < 10:
        continue

    msg = RATING_PROMPT.format(response=response_clean[:400])
    # Use /no_think format for Qwen3
    chat = [
        {"role": "system", "content": "You are a rating assistant. Output JSON only. No thinking."},
        {"role": "user", "content": msg},
    ]
    chat_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # Force no-thinking by appending closing think tag if model starts thinking
    gen_input = tokenizer(chat_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **gen_input, max_new_tokens=500,
            temperature=0.1, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][gen_input["input_ids"].shape[1]:], skip_special_tokens=True)

    # Strip thinking if present
    if "</think>" in raw:
        raw = raw[raw.index("</think>") + len("</think>"):].strip()

    try:
        json_match = re.search(r'\{[^}]+\}', raw)
        if json_match:
            rating = json.loads(json_match.group())
            r["rating"] = rating
            rated_count += 1
            print(f"  {r['persona']}/P{r['prompt_idx']+1}: {rating}")
        else:
            print(f"  {r['persona']}/P{r['prompt_idx']+1}: NO JSON in: {raw[:80]}")
    except Exception as e:
        print(f"  {r['persona']}/P{r['prompt_idx']+1}: PARSE ERROR: {e}")

print(f"\nRated {rated_count}/{len(results)} responses")

# Correlation with ratings
if rated_count > 10:
    print(f"\n  Coefficient → Rating Correlation:")
    rating_map = {
        "emotion_valence": "emotion_warmth",
        "formality": "formality",
        "creativity": "creativity",
        "confidence": "confidence",
        "empathy": "empathy",
    }
    for dim, rkey in rating_map.items():
        cv = [r["coefficients"][dim] for r in results if r.get("rating") and rkey in r["rating"]]
        rv = [r["rating"][rkey] for r in results if r.get("rating") and rkey in r["rating"]]
        if len(cv) >= 6:
            corr = np.corrcoef(cv, rv)[0, 1]
            sig = "✓ STRONG" if abs(corr) > 0.3 else "~ weak" if abs(corr) > 0.15 else "✗ none"
            print(f"    {dim:<20} → {rkey:<20} r={corr:+.3f}  {sig}")

    # Per-persona average ratings
    print(f"\n  {'Persona':<22} {'Warmth':>7} {'Formal':>7} {'Create':>7} {'Confid':>7} {'Empath':>7}")
    print(f"  " + "-" * 60)
    for persona in PERSONAS:
        ratings = [r["rating"] for r in results if r["persona"] == persona and r.get("rating")]
        if ratings:
            means = {}
            for key in ["emotion_warmth", "formality", "creativity", "confidence", "empathy"]:
                vals = [pr[key] for pr in ratings if key in pr]
                means[key] = np.mean(vals) if vals else 0
            print(f"  {persona:<22} {means.get('emotion_warmth',0):>7.1f} {means.get('formality',0):>7.1f} "
                  f"{means.get('creativity',0):>7.1f} {means.get('confidence',0):>7.1f} {means.get('empathy',0):>7.1f}")

# Save final
with open("/cache/zhangjing/Joi/exp_a_results_rated.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nFinal results saved to Joi/exp_a_results_rated.json")

del model
torch.cuda.empty_cache()
