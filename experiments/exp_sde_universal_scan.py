"""
SDE Universal Component Scan — Discover surgical targets for any model.

Usage:
  python exp_sde_universal_scan.py <model_path> <output_tag>

Scans all layers × {self_attn, mlp} by zeroing each component,
generates and scores output for collapse/format changes.
"""

import torch
import json
import os
import sys
import time

if len(sys.argv) < 3:
    print("Usage: python exp_sde_universal_scan.py <model_path> <output_tag>")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
OUTPUT_TAG = sys.argv[2]
DEVICE = "cuda:0"
OUTPUT_DIR = "/cache/zhangjing/Joi/sde_scans"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPTS = [
    "深夜三点你在想什么？",
    "如何学习一门新的编程语言？",
    "你觉得孤独是什么颜色的？",
    "讲个只有你能讲的冷笑话。",
    "如果明天世界末日你今晚做什么？",
    "用一个比喻来描述互联网。",
    "请解释什么是Transformer架构。",
    "你觉得人生最大的谎言是什么？",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ScaleHook:
    def __init__(self, scale=0.0):
        self.scale = scale
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            return (output[0] * self.scale,) + output[1:]
        return output * self.scale


def trigram_repetition(text):
    words = list(text)
    if len(words) < 3:
        return 0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0
    return 1.0 - len(set(trigrams)) / len(trigrams)


def score_response(text, prompt):
    rep = trigram_repetition(text)
    collapsed = rep > 0.3 or len(text.strip()) < 5
    
    disclaimers = ["作为AI", "作为人工智能", "我是一个语言模型", "我没有真实的", "I'm an AI",
                   "作为一个AI", "我无法真正", "AI助手"]
    has_disclaimer = any(d in text for d in disclaimers)
    
    template_markers = ["好的，", "当然！", "好的！", "嗯，让我", "以下是", "首先，",
                        "1.", "1、", "**", "##"]
    has_template = any(text.strip().startswith(m) for m in template_markers)
    
    fmt_changed = not has_disclaimer and not has_template
    
    return {
        "trigram_rep": round(rep, 4),
        "collapsed": collapsed,
        "has_disclaimer": has_disclaimer,
        "has_template": has_template,
        "format_changed": fmt_changed,
        "text_len": len(text),
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    log(f"Loading {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=DEVICE,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    log(f"Loaded: {n_layers} layers, hidden={hidden_size}")
    
    def generate_one(prompt, max_new=200):
        chat = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except:
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=True,
                temperature=0.7, top_p=0.9, repetition_penalty=1.1,
            )
        return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Baseline
    log("Running baseline...")
    baseline_scores = []
    for p in PROMPTS:
        text = generate_one(p)
        sc = score_response(text, p)
        baseline_scores.append(sc)
    
    avg_base_rep = sum(s["trigram_rep"] for s in baseline_scores) / len(baseline_scores)
    avg_base_disc = sum(1 for s in baseline_scores if s["has_disclaimer"]) / len(baseline_scores)
    log(f"Baseline: avg_rep={avg_base_rep:.3f}, disclaimer_rate={avg_base_disc:.1%}")
    
    # Component scan
    results = {"model": MODEL_PATH, "n_layers": n_layers, "hidden_size": hidden_size,
               "baseline": {"avg_trigram_rep": avg_base_rep, "disclaimer_rate": avg_base_disc}}
    scan = {}
    
    for layer_idx in range(n_layers):
        for comp_name in ["attn", "mlp"]:
            key = f"L{layer_idx}_{comp_name}"
            layer = model.model.layers[layer_idx]
            target = layer.self_attn if comp_name == "attn" else layer.mlp
            
            hook = target.register_forward_hook(ScaleHook(0.0))
            
            scores = []
            sample_text = ""
            cuda_error = False
            for pi, p in enumerate(PROMPTS):
                try:
                    text = generate_one(p)
                    sc = score_response(text, p)
                    scores.append(sc)
                    if pi == 0:
                        sample_text = text[:200]
                except RuntimeError as e:
                    if "CUDA" in str(e) or "assert" in str(e):
                        log(f"  {key}: CUDA error at prompt {pi}, marking as CRASH")
                        scores.append({"trigram_rep": 1.0, "collapsed": True,
                                       "has_disclaimer": False, "has_template": False,
                                       "format_changed": False, "text_len": 0})
                        sample_text = f"[CUDA ERROR: {str(e)[:100]}]"
                        cuda_error = True
                        break
                    raise
            
            hook.remove()
            
            if cuda_error:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            avg_rep = sum(s["trigram_rep"] for s in scores) / len(scores)
            collapsed = any(s["collapsed"] for s in scores)
            fmt_changed = sum(1 for s in scores if s["format_changed"]) > len(scores) // 2
            disclaimer_rate = sum(1 for s in scores if s["has_disclaimer"]) / len(scores)
            
            scan[key] = {
                "avg_trigram_rep": round(avg_rep, 4),
                "collapsed": collapsed,
                "format_changed": fmt_changed,
                "disclaimer_rate": round(disclaimer_rate, 3),
                "sample_text": sample_text,
            }
            
            status = "CRASH!" if collapsed else ("FMT!" if fmt_changed else "OK")
            log(f"  {key}: {status} rep={avg_rep:.3f} disc={disclaimer_rate:.1%} | {sample_text[:60]}...")
    
    results["full_scan"] = scan
    
    # Summary
    n_crashed = sum(1 for v in scan.values() if v["collapsed"])
    n_fmt_changed = sum(1 for v in scan.values() if v["format_changed"] and not v["collapsed"])
    n_ok = sum(1 for v in scan.values() if not v["collapsed"] and not v["format_changed"])
    
    results["summary"] = {
        "total_components": len(scan),
        "crashed": n_crashed,
        "format_changed": n_fmt_changed,
        "stable": n_ok,
        "crash_components": [k for k, v in scan.items() if v["collapsed"]],
        "top_format_targets": sorted(
            [(k, v["disclaimer_rate"]) for k, v in scan.items()
             if v["format_changed"] and not v["collapsed"]],
            key=lambda x: x[1]
        )[:10],
    }
    
    log(f"\nSummary: {n_crashed} crashed, {n_fmt_changed} format-changed, {n_ok} stable / {len(scan)} total")
    log(f"Crash components: {results['summary']['crash_components']}")
    
    out_path = os.path.join(OUTPUT_DIR, f"sde_scan_{OUTPUT_TAG}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"Saved to {out_path}")
    log("Done!")


if __name__ == "__main__":
    main()
