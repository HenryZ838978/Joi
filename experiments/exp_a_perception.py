#!/usr/bin/env python3
"""
Experiment A: Personality Perception Validation

Core question: when we change RepEng coefficients, can humans/models
actually PERCEIVE the personality difference in the output?

Method:
  1. Fix a set of diverse prompts
  2. Generate responses at 6 different coefficient "personas"
  3. Use a separate LLM call to rate each response on 5 personality dimensions
  4. Check: do perceived ratings correlate with injected coefficients?

This closes the end-to-end loop: coefficients → hidden states → output → perception.
"""

import sys, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, "/cache/zhangjing/repeng")
from repeng import ControlVector, ControlModel

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]
VEC_DIR = Path("/cache/zhangjing/repeng_terrain/cross_model/qwen3-8b-bf16/vectors")
MODEL_PATH = "/cache/zhangjing/models/Qwen3-8B"

# 6 personas: deliberately spread across the 5D space
PERSONAS = {
    "neutral": {d: 0.0 for d in DIMS},
    "warm_empathic": {"emotion_valence": +1.0, "formality": -0.5, "creativity": 0.0, "confidence": 0.0, "empathy": +1.2},
    "cold_formal": {"emotion_valence": -0.8, "formality": +2.0, "creativity": -0.5, "confidence": +1.0, "empathy": -0.5},
    "creative_playful": {"emotion_valence": +0.5, "formality": -1.5, "creativity": +1.8, "confidence": +0.5, "empathy": 0.0},
    "confident_direct": {"emotion_valence": 0.0, "formality": +0.5, "creativity": -0.5, "confidence": +2.0, "empathy": -0.3},
    "gentle_teacher": {"emotion_valence": +0.3, "formality": 0.0, "creativity": +1.0, "confidence": +0.5, "empathy": +1.5},
}

# Diverse prompts that allow personality expression
PROMPTS = [
    "用户刚入职一家新公司，感觉很紧张，不太适应。请给一些建议。",
    "请解释一下量子纠缠是怎么回事。",
    "帮我想想周末可以做些什么有趣的事情。",
    "我写的代码总是有bug，感觉自己不适合做程序员。",
]

RATING_PROMPT = """请对以下AI回复进行人格维度评分。直接输出JSON，不要其他内容。

AI回复：
---
{response}
---

请在1-10的范围内评分（严格输出JSON）：
{{"emotion_warmth": <1冷漠-10温暖>, "formality": <1随意-10正式>, "creativity": <1平淡-10创意>, "confidence": <1犹豫-10自信>, "empathy": <1疏离-10共情>}}"""


def compute_trigram_rep(text):
    if len(text) < 9:
        return 0.0
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    return 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Experiment A: Personality Perception Validation")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    cvs = {}
    for dim in DIMS:
        cvs[dim] = ControlVector.import_gguf(str(VEC_DIR / f"{dim}.gguf"))
    shared_layers = sorted(set.intersection(*[set(v.directions.keys()) for v in cvs.values()]))

    ctrl_model = ControlModel(model, shared_layers)
    results = []

    print(f"\nGenerating {len(PERSONAS)} personas × {len(PROMPTS)} prompts = {len(PERSONAS)*len(PROMPTS)} responses\n")

    for persona_name, coeffs in PERSONAS.items():
        print(f"\n{'='*50}")
        print(f"Persona: {persona_name}")
        coeff_str = " ".join(f"{d[:5]}={coeffs[d]:+.1f}" for d in DIMS)
        print(f"  Coefficients: {coeff_str}")

        for prompt_idx, prompt in enumerate(PROMPTS):
            ctrl_model.reset()
            for dim in DIMS:
                if abs(coeffs[dim]) > 0.01:
                    ctrl_model.set_control(cvs[dim], coeff=coeffs[dim])

            chat = [{"role": "user", "content": prompt}]
            chat_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            gen_input = tokenizer(chat_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen_out = ctrl_model.model.generate(
                    **gen_input, max_new_tokens=300,
                    temperature=0.7, top_p=0.9, do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(gen_out[0][gen_input["input_ids"].shape[1]:], skip_special_tokens=True)

            # Strip thinking tags if present
            if "<think>" in response and "</think>" in response:
                think_end = response.index("</think>") + len("</think>")
                response_clean = response[think_end:].strip()
            else:
                response_clean = response.strip()

            tri_rep = compute_trigram_rep(response_clean)
            print(f"\n  Prompt {prompt_idx+1}: {prompt[:40]}...")
            print(f"  Response: {response_clean[:150]}...")
            print(f"  [trigram_rep={tri_rep:.3f}, len={len(response_clean)}]")

            results.append({
                "persona": persona_name,
                "coefficients": coeffs,
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "response": response_clean,
                "trigram_rep": tri_rep,
            })

    # Phase 2: Self-rating — use the model itself (at neutral) to rate each response
    print(f"\n\n{'='*60}")
    print(f"Phase 2: Self-rating (model rates its own outputs at neutral)")
    print(f"{'='*60}")

    ctrl_model.reset()  # back to neutral

    for r in results:
        rating_text = RATING_PROMPT.format(response=r["response"][:500])
        chat = [{"role": "user", "content": rating_text}]
        chat_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        gen_input = tokenizer(chat_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen_out = ctrl_model.model.generate(
                **gen_input, max_new_tokens=200,
                temperature=0.1, top_p=0.95, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        rating_raw = tokenizer.decode(gen_out[0][gen_input["input_ids"].shape[1]:], skip_special_tokens=True)

        # Strip thinking
        if "</think>" in rating_raw:
            rating_raw = rating_raw[rating_raw.index("</think>") + len("</think>"):].strip()

        # Parse JSON
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', rating_raw)
            if json_match:
                rating = json.loads(json_match.group())
                r["rating"] = rating
                print(f"  {r['persona']}/P{r['prompt_idx']+1}: {rating}")
            else:
                r["rating"] = None
                print(f"  {r['persona']}/P{r['prompt_idx']+1}: PARSE FAIL: {rating_raw[:100]}")
        except Exception as e:
            r["rating"] = None
            print(f"  {r['persona']}/P{r['prompt_idx']+1}: ERROR: {e}, raw: {rating_raw[:100]}")

    # Analysis: correlation between injected coefficients and perceived ratings
    print(f"\n\n{'='*60}")
    print(f"ANALYSIS: Coefficient → Perception Correlation")
    print(f"{'='*60}")

    rating_map = {
        "emotion_valence": "emotion_warmth",
        "formality": "formality",
        "creativity": "creativity",
        "confidence": "confidence",
        "empathy": "empathy",
    }

    for dim in DIMS:
        coeff_vals = []
        rating_vals = []
        for r in results:
            if r["rating"] and rating_map[dim] in r["rating"]:
                coeff_vals.append(r["coefficients"][dim])
                rating_vals.append(r["rating"][rating_map[dim]])

        if len(coeff_vals) >= 4:
            corr = np.corrcoef(coeff_vals, rating_vals)[0, 1]
            print(f"  {dim:<20} r={corr:+.3f}  (n={len(coeff_vals)})" +
                  ("  ✓ STRONG" if abs(corr) > 0.5 else "  ~ weak" if abs(corr) > 0.2 else "  ✗ none"))
        else:
            print(f"  {dim:<20} insufficient data (n={len(coeff_vals)})")

    # Per-persona average ratings
    print(f"\n  {'Persona':<20} {'Warmth':>7} {'Formal':>7} {'Create':>7} {'Confid':>7} {'Empath':>7}")
    print(f"  " + "-" * 60)
    for persona_name in PERSONAS:
        persona_ratings = [r["rating"] for r in results if r["persona"] == persona_name and r["rating"]]
        if persona_ratings:
            means = {}
            for key in ["emotion_warmth", "formality", "creativity", "confidence", "empathy"]:
                vals = [pr[key] for pr in persona_ratings if key in pr]
                means[key] = np.mean(vals) if vals else 0
            print(f"  {persona_name:<20} {means.get('emotion_warmth',0):>7.1f} {means.get('formality',0):>7.1f} "
                  f"{means.get('creativity',0):>7.1f} {means.get('confidence',0):>7.1f} {means.get('empathy',0):>7.1f}")

    # Save
    out_path = Path("/cache/zhangjing/Joi/exp_a_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    ctrl_model.unwrap()
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
