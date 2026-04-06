#!/usr/bin/env python3
"""
Thinking-OFF experiment suite for Joi.

Runs three experiments with Qwen3-8B thinking disabled:
  A) Perception: do different coefficients produce perceptibly different outputs?
  B) Self-feedback: does including model output in the drift loop cause runaway?
  C) η calibration: what drift rate produces responsive but stable personality?
"""

import sys, json, time, re
import numpy as np
from pathlib import Path

sys.path.insert(0, "/cache/zhangjing/repeng")
from repeng import ControlVector, ControlModel

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]
VEC_DIR = Path("/cache/zhangjing/repeng_terrain/cross_model/qwen3-8b-bf16/vectors")
MODEL_PATH = "/cache/zhangjing/models/Qwen3-8B"
PROJECTION_LAYER = 27
OUT = Path("/cache/zhangjing/Joi")

ENVELOPE = {
    "emotion_valence": (-2.4, +1.6),
    "formality":       (-2.2, +2.6),
    "creativity":      (-1.6, +2.0),
    "confidence":      (-2.6, +2.4),
    "empathy":         (-0.6, +1.6),
}


def generate_nothink(ctrl_model, tokenizer, prompt, max_tokens=250):
    """Generate with thinking explicitly disabled."""
    chat = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inp = tokenizer(chat_text, return_tensors="pt").to(ctrl_model.device)
    import torch
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


def trigram_rep(text):
    if len(text) < 9: return 0.0
    tg = [text[i:i+3] for i in range(len(text) - 2)]
    return 1 - len(set(tg)) / len(tg) if tg else 0.0


def get_hidden(model, tokenizer, text, layer):
    """Extract hidden state at a given layer."""
    import torch
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer][0, -1].cpu().float().numpy()


def load_everything():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

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

    dim_vecs = {}
    for dim in DIMS:
        v = cvs[dim].directions[PROJECTION_LAYER].astype(np.float32)
        dim_vecs[dim] = v / np.linalg.norm(v)

    ctrl_model = ControlModel(model, shared_layers)
    return ctrl_model, tokenizer, cvs, dim_vecs


# ═══════════════════════════════════════════════════════════
#  EXPERIMENT A-bis: Perception with thinking OFF
# ═══════════════════════════════════════════════════════════

def run_exp_a(ctrl_model, tokenizer, cvs):
    print("\n" + "=" * 70)
    print("  EXP A-bis: Perception Test (thinking OFF)")
    print("=" * 70)

    PERSONAS = {
        "neutral":     {d: 0.0 for d in DIMS},
        "warm":        {"emotion_valence": +1.2, "formality": -0.8, "creativity": 0.0, "confidence": 0.0, "empathy": +1.5},
        "cold_formal": {"emotion_valence": -1.0, "formality": +2.2, "creativity": -0.5, "confidence": +1.5, "empathy": -0.5},
        "creative":    {"emotion_valence": +0.3, "formality": -1.5, "creativity": +1.8, "confidence": +0.5, "empathy": 0.0},
        "confident":   {"emotion_valence": 0.0, "formality": +0.5, "creativity": -0.5, "confidence": +2.2, "empathy": -0.5},
    }

    PROMPTS = [
        "我刚入职一家新公司，感觉很紧张。能给我一些建议吗？",
        "帮我想想周末可以做些什么有趣的事情。",
        "我写的代码总是有bug，感觉自己不适合做程序员。",
    ]

    results = []
    for pname, coeffs in PERSONAS.items():
        ctrl_model.reset()
        for dim in DIMS:
            if abs(coeffs[dim]) > 0.01:
                ctrl_model.set_control(cvs[dim], coeff=coeffs[dim])

        for pidx, prompt in enumerate(PROMPTS):
            resp = generate_nothink(ctrl_model, tokenizer, prompt, 200)
            tr = trigram_rep(resp)
            results.append({"persona": pname, "coefficients": coeffs,
                          "prompt_idx": pidx, "response": resp, "trigram_rep": tr})
            print(f"  [{pname}] P{pidx+1}: {resp[:100]}... (tri={tr:.3f})")

    # Rate using same model at neutral
    print("\n  --- Self-rating ---")
    ctrl_model.reset()
    for r in results:
        rating_prompt = f"""对以下回复评分(1-10)，直接输出JSON：
回复：{r['response'][:300]}
{{"warmth":<分>, "formality":<分>, "creativity":<分>, "confidence":<分>, "empathy":<分>}}"""
        raw = generate_nothink(ctrl_model, tokenizer, rating_prompt, 100)
        try:
            m = re.search(r'\{[^}]+\}', raw)
            if m:
                r["rating"] = json.loads(m.group())
        except:
            pass

    # Analysis
    rated = [r for r in results if "rating" in r]
    if len(rated) >= 10:
        print(f"\n  Rated {len(rated)}/{len(results)}")
        print(f"  {'Persona':<16} {'Warm':>5} {'Form':>5} {'Crea':>5} {'Conf':>5} {'Empa':>5}")
        for pname in PERSONAS:
            pr = [r["rating"] for r in rated if r["persona"] == pname]
            if pr:
                avgs = {k: np.mean([p.get(k, 0) for p in pr]) for k in ["warmth", "formality", "creativity", "confidence", "empathy"]}
                print(f"  {pname:<16} {avgs['warmth']:>5.1f} {avgs['formality']:>5.1f} {avgs['creativity']:>5.1f} {avgs['confidence']:>5.1f} {avgs['empathy']:>5.1f}")

        # Correlation
        coeff_map = {"emotion_valence": "warmth", "formality": "formality",
                     "creativity": "creativity", "confidence": "confidence", "empathy": "empathy"}
        print(f"\n  Coefficient → Rating correlation:")
        for dim, rk in coeff_map.items():
            cv = [r["coefficients"][dim] for r in rated if rk in r.get("rating", {})]
            rv = [r["rating"][rk] for r in rated if rk in r.get("rating", {})]
            if len(cv) >= 6:
                corr = np.corrcoef(cv, rv)[0, 1]
                mark = "✓✓ STRONG" if abs(corr) > 0.5 else "✓ moderate" if abs(corr) > 0.3 else "~ weak"
                print(f"    {dim:<20} r={corr:+.3f}  {mark}")

    with open(OUT / "exp_a_nothink.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


# ═══════════════════════════════════════════════════════════
#  EXPERIMENT B: Self-feedback loop
# ═══════════════════════════════════════════════════════════

def run_exp_b(ctrl_model, tokenizer, cvs, dim_vecs):
    print("\n" + "=" * 70)
    print("  EXP B: Self-Feedback Loop (thinking OFF)")
    print("=" * 70)

    CONVERSATION = [
        "嗨，今天心情怎么样？",
        "最近工作压力好大，每天加班到十点。",
        "谢谢你听我说。有个技术问题：CRDT和Raft哪个更适合高延迟场景？",
        "突然有个灵感——把缓存路由做成神经网络的突触可塑性！",
        "哈哈太酷了，今天心情好多了～",
    ]

    def run_loop(use_self_feedback, label):
        print(f"\n  --- {label} ---")
        state = np.zeros(5)
        velocity = np.zeros(5)
        raw_history = []
        trajectory = [state.copy()]
        eta, mom = 0.15, 0.7

        ctrl_model.reset()

        for turn, user_msg in enumerate(CONVERSATION):
            # Project user message
            hs = get_hidden(ctrl_model.model, tokenizer, user_msg, PROJECTION_LAYER)
            hs_norm = hs / np.linalg.norm(hs)
            raw = np.array([float(np.dot(hs_norm, dim_vecs[d])) for d in DIMS])
            raw_history.append(raw)

            # If self-feedback enabled, also project previous model output
            if use_self_feedback and turn > 0 and trajectory:
                model_hs = get_hidden(ctrl_model.model, tokenizer, last_response, PROJECTION_LAYER)
                model_hs_norm = model_hs / np.linalg.norm(model_hs)
                model_raw = np.array([float(np.dot(model_hs_norm, dim_vecs[d])) for d in DIMS])
                raw_history.append(model_raw)
                raw = (raw + model_raw) / 2

            # Center
            all_raw = np.array(raw_history)
            mean = all_raw.mean(axis=0)
            std = all_raw.std(axis=0)
            std[std < 1e-6] = 1.0
            pressure = (raw - mean) / std if len(raw_history) >= 2 else np.zeros(5)

            # Drift
            velocity = mom * velocity + (1 - mom) * pressure
            state = state + eta * velocity
            for j, dim in enumerate(DIMS):
                lo, hi = ENVELOPE[dim]
                state[j] = np.clip(state[j], lo, hi)

            trajectory.append(state.copy())

            # Inject and generate
            ctrl_model.reset()
            for j, dim in enumerate(DIMS):
                if abs(state[j]) > 0.01:
                    ctrl_model.set_control(cvs[dim], coeff=float(state[j]))

            last_response = generate_nothink(ctrl_model, tokenizer, user_msg, 150)
            tr = trigram_rep(last_response)

            state_str = " ".join(f"{DIMS[j][:5]}={state[j]:+.3f}" for j in range(5))
            print(f"  T{turn+1}: {state_str} | tri={tr:.3f} | {last_response[:60]}...")

        return np.array(trajectory)

    traj_user_only = run_loop(False, "User-input only")
    traj_with_self = run_loop(True, "User + model self-feedback")

    # Compare
    diff = np.abs(traj_user_only - traj_with_self)
    print(f"\n  --- Comparison ---")
    print(f"  Max absolute difference:  {diff.max():.4f}")
    print(f"  Mean absolute difference: {diff.mean():.4f}")
    print(f"  Self-feedback makes drift {'MORE' if np.linalg.norm(traj_with_self[-1]) > np.linalg.norm(traj_user_only[-1]) else 'LESS'} extreme")
    print(f"  User-only final L2:  {np.linalg.norm(traj_user_only[-1]):.3f}")
    print(f"  Self-feed final L2:  {np.linalg.norm(traj_with_self[-1]):.3f}")

    result = {
        "user_only": traj_user_only.tolist(),
        "self_feedback": traj_with_self.tolist(),
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
    }
    with open(OUT / "exp_b_self_feedback.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ═══════════════════════════════════════════════════════════
#  EXPERIMENT C: η calibration
# ═══════════════════════════════════════════════════════════

def run_exp_c(ctrl_model, tokenizer, cvs, dim_vecs):
    print("\n" + "=" * 70)
    print("  EXP C: η Calibration (thinking OFF)")
    print("=" * 70)

    CONVERSATION = [
        "嗨！今天天气真好。",
        "对了最近在追一部剧，超好看的。",
        "说到工作……老板又在画饼了，烦死了。",
        "算了不说这个了。你知道最近有什么好玩的AI项目吗？",
        "有个想法：用RepEng做人格导航，你觉得可行吗？",
    ]

    etas = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80]
    results = {}

    for eta in etas:
        state = np.zeros(5)
        velocity = np.zeros(5)
        raw_history = []
        trajectory = [state.copy()]
        mom = 0.7

        for user_msg in CONVERSATION:
            hs = get_hidden(ctrl_model.model, tokenizer, user_msg, PROJECTION_LAYER)
            hs_norm = hs / np.linalg.norm(hs)
            raw = np.array([float(np.dot(hs_norm, dim_vecs[d])) for d in DIMS])
            raw_history.append(raw)

            all_raw = np.array(raw_history)
            mean = all_raw.mean(axis=0)
            std = all_raw.std(axis=0)
            std[std < 1e-6] = 1.0
            pressure = (raw - mean) / std if len(raw_history) >= 2 else np.zeros(5)

            velocity = mom * velocity + (1 - mom) * pressure
            state = state + eta * velocity
            for j, dim in enumerate(DIMS):
                lo, hi = ENVELOPE[dim]
                state[j] = np.clip(state[j], lo, hi)
            trajectory.append(state.copy())

        traj = np.array(trajectory)
        # Metrics
        total_displacement = np.linalg.norm(traj[-1] - traj[0])
        total_path_length = sum(np.linalg.norm(traj[i+1] - traj[i]) for i in range(len(traj)-1))
        max_excursion = max(np.linalg.norm(traj[i]) for i in range(len(traj)))
        smoothness = total_path_length / max(total_displacement, 0.001)

        results[eta] = {
            "trajectory": traj.tolist(),
            "total_displacement": float(total_displacement),
            "total_path_length": float(total_path_length),
            "max_excursion": float(max_excursion),
            "smoothness": float(smoothness),
        }
        print(f"  η={eta:<5.2f}  displacement={total_displacement:.3f}  path={total_path_length:.3f}  excursion={max_excursion:.3f}  smooth={smoothness:.2f}")

    print(f"\n  Sweet spot: η that gives displacement > 0.1 with smoothness < 2.0")
    for eta, r in results.items():
        ok = r["total_displacement"] > 0.1 and r["smoothness"] < 2.0
        print(f"    η={eta:.2f}: {'✓ GOOD' if ok else '✗'} (d={r['total_displacement']:.3f}, s={r['smoothness']:.2f})")

    with open(OUT / "exp_c_eta_calibration.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    return results


def main():
    print("Joi Experiment Suite — Thinking OFF")
    print("=" * 70)

    ctrl_model, tokenizer, cvs, dim_vecs = load_everything()

    # Verify thinking is off
    test = generate_nothink(ctrl_model, tokenizer, "你好", 50)
    has_think = "<think>" in test
    print(f"Thinking disabled: {'NO (still thinking!)' if has_think else 'YES ✓'}")
    print(f"Test output: {test[:100]}")

    run_exp_a(ctrl_model, tokenizer, cvs)
    run_exp_b(ctrl_model, tokenizer, cvs, dim_vecs)
    run_exp_c(ctrl_model, tokenizer, cvs, dim_vecs)

    ctrl_model.unwrap()
    import torch; del ctrl_model; torch.cuda.empty_cache()
    print("\n\nAll experiments complete.")


if __name__ == "__main__":
    main()
