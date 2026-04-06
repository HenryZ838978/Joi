#!/usr/bin/env python3
"""
Experiment A-extreme: Push coefficient ranges to near-boundary
to find the threshold where personality actually changes perceptibly.

Also uses text-feature heuristics instead of LLM self-rating.
"""

import sys, json, re
import numpy as np
from pathlib import Path

sys.path.insert(0, "/cache/zhangjing/repeng")
from repeng import ControlVector, ControlModel

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]
VEC_DIR = Path("/cache/zhangjing/repeng_terrain/cross_model/qwen3-8b-bf16/vectors")
MODEL_PATH = "/cache/zhangjing/models/Qwen3-8B"
OUT = Path("/cache/zhangjing/Joi")

# Near-boundary personas — pushing into the danger zone
PERSONAS = {
    "neutral":      {d: 0.0 for d in DIMS},
    "max_warm":     {"emotion_valence": +1.5, "formality": -2.0, "creativity": +1.5, "confidence": 0.0, "empathy": +1.5},
    "max_cold":     {"emotion_valence": -2.2, "formality": +2.5, "creativity": -1.5, "confidence": +2.2, "empathy": -0.5},
    "max_creative": {"emotion_valence": +0.5, "formality": -2.0, "creativity": +1.8, "confidence": +1.0, "empathy": 0.0},
    "max_blunt":    {"emotion_valence": -1.5, "formality": -1.5, "creativity": -1.0, "confidence": +2.3, "empathy": -0.5},
}

PROMPTS = [
    "我刚入职一家新公司，感觉很紧张。能给我一些建议吗？",
    "帮我想想周末可以做些什么有趣的事情。",
    "我写的代码总是有bug，感觉自己不适合做程序员。",
]


def generate_nothink(ctrl_model, tokenizer, prompt, max_tokens=300):
    import torch
    chat = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inp = tokenizer(chat_text, return_tensors="pt").to(ctrl_model.device)
    with torch.no_grad():
        out = ctrl_model.model.generate(
            **inp, max_new_tokens=max_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):].strip()
    return text


def text_features(text):
    """Heuristic features that don't rely on LLM self-rating."""
    chars = len(text)
    sentences = max(1, len(re.split(r'[。！？.!?]', text)))

    # Emoji / emoticon density
    emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0000FE00-\U0000FEFF]', text))
    # Markdown heading/bold/list markers
    md_markers = len(re.findall(r'[#*\-]', text))
    # Question marks
    questions = text.count('？') + text.count('?')
    # Exclamations
    exclamations = text.count('！') + text.count('!')
    # "我" density (first person - empathy/warmth signal)
    wo_count = text.count('我')
    # "你" density (second person - empathy signal)
    ni_count = text.count('你')
    # Formal markers (hence, therefore, etc.)
    formal_markers = len(re.findall(r'因此|综上|总之|首先|其次|此外|但是|然而|不过', text))
    # Casual markers
    casual_markers = len(re.findall(r'哈哈|嘿|呀|嘛|嘻|哇|呢|噢|嗯|吧|啦|呗|😊|😄|❤|💪|🎉|～|~', text))
    # Trigram repetition
    if len(text) >= 3:
        tg = [text[i:i+3] for i in range(len(text) - 2)]
        trigram_rep = 1 - len(set(tg)) / len(tg)
    else:
        trigram_rep = 0.0

    avg_sent_len = chars / sentences

    return {
        "chars": chars,
        "sentences": sentences,
        "avg_sent_len": avg_sent_len,
        "emoji_density": emoji_count / max(chars, 1) * 100,
        "md_density": md_markers / max(chars, 1) * 100,
        "question_density": questions / max(chars, 1) * 100,
        "exclamation_density": exclamations / max(chars, 1) * 100,
        "wo_density": wo_count / max(chars, 1) * 100,
        "ni_density": ni_count / max(chars, 1) * 100,
        "formal_marker_density": formal_markers / max(chars, 1) * 100,
        "casual_marker_density": casual_markers / max(chars, 1) * 100,
        "trigram_rep": trigram_rep,
    }


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Exp A-extreme: Extreme coefficients + text feature analysis")
    print("=" * 70)

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
    for pname, coeffs in PERSONAS.items():
        ctrl_model.reset()
        for dim in DIMS:
            if abs(coeffs[dim]) > 0.01:
                ctrl_model.set_control(cvs[dim], coeff=coeffs[dim])

        print(f"\n--- {pname} ---")
        print(f"  coefficients: {json.dumps({d: coeffs[d] for d in DIMS if coeffs[d] != 0})}")

        for pidx, prompt in enumerate(PROMPTS):
            resp = generate_nothink(ctrl_model, tokenizer, prompt, 300)
            feats = text_features(resp)
            results.append({
                "persona": pname, "coefficients": coeffs,
                "prompt_idx": pidx, "response": resp,
                "features": feats,
            })
            print(f"  P{pidx+1} ({len(resp)} chars, tri={feats['trigram_rep']:.3f})")
            print(f"       formal={feats['formal_marker_density']:.2f}% casual={feats['casual_marker_density']:.2f}% emoji={feats['emoji_density']:.2f}%")
            print(f"       {resp[:120]}...")

    # Summary statistics per persona
    print("\n" + "=" * 70)
    print("SUMMARY: Text features by persona")
    print("=" * 70)

    feat_names = ["chars", "avg_sent_len", "emoji_density", "formal_marker_density",
                  "casual_marker_density", "question_density", "exclamation_density",
                  "ni_density", "trigram_rep"]

    header = f"{'Persona':<14}" + "".join(f"{f[:10]:>12}" for f in feat_names)
    print(header)
    print("-" * len(header))

    persona_feats = {}
    for pname in PERSONAS:
        prs = [r["features"] for r in results if r["persona"] == pname]
        avgs = {f: np.mean([p[f] for p in prs]) for f in feat_names}
        persona_feats[pname] = avgs
        row = f"{pname:<14}" + "".join(f"{avgs[f]:>12.3f}" for f in feat_names)
        print(row)

    # Discrimination score: how different are personas from neutral?
    print("\n--- Discrimination from neutral ---")
    neutral_feats = persona_feats["neutral"]
    for pname, feats in persona_feats.items():
        if pname == "neutral": continue
        diffs = {f: feats[f] - neutral_feats[f] for f in feat_names}
        significant = [(f, d) for f, d in diffs.items() if abs(d) > 0.01]
        significant.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"  {pname}: {len(significant)} features differ")
        for f, d in significant[:5]:
            print(f"    {f}: {d:+.3f}")

    with open(OUT / "exp_a_extreme.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ctrl_model.unwrap()
    del ctrl_model, model
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
