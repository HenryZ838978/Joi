#!/usr/bin/env python3
"""
Phase 3: Closed-Loop Integration

End-to-end: conversation → hidden state → projection → drift → coefficient injection → generation

Uses transformers + repeng ControlModel directly (no vLLM needed).
Demonstrates the full Joi loop in miniature.
"""

import sys, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, "/cache/zhangjing/repeng")
from repeng import ControlVector, ControlModel

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]
DIM_SHORT = ["Emotion", "Formal", "Creative", "Confid", "Empathy"]
VEC_DIR = Path("/cache/zhangjing/repeng_terrain/cross_model/qwen3-8b-bf16/vectors")
MODEL_PATH = "/cache/zhangjing/models/Qwen3-8B"
PROJECTION_LAYER = 27

ENVELOPE = {
    "emotion_valence": (-2.4, +1.6),
    "formality":       (-2.2, +2.6),
    "creativity":      (-1.6, +2.0),
    "confidence":      (-2.6, +2.4),
    "empathy":         (-0.6, +1.6),
}

ETA = 0.15
MOMENTUM = 0.7

CONVERSATION = [
    "嗨！今天心情怎么样？",
    "最近工作压力特别大，每天加班到很晚，感觉自己要撑不住了。",
    "谢谢你听我说。有个技术问题想请教：分布式系统中CRDT和Raft哪个更适合高延迟场景？",
    "突然有个灵感！如果把缓存路由做成类似神经网络的突触可塑性呢？",
    "哈哈这个想法太酷了。今天聊得开心，心情好多了～",
]


def compute_trigram_rep(text):
    """Measure output degradation via trigram repetition ratio."""
    if len(text) < 9:
        return 0.0
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    if not trigrams:
        return 0.0
    return 1 - len(set(trigrams)) / len(trigrams)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Phase 3: Closed-Loop Joi Prototype")
    print("=" * 60)

    # Load model
    print("Loading Qwen3-8B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    # Load control vectors
    cvs = {}
    for dim in DIMS:
        cvs[dim] = ControlVector.import_gguf(str(VEC_DIR / f"{dim}.gguf"))

    shared_layers = sorted(set.intersection(*[set(v.directions.keys()) for v in cvs.values()]))

    # Load projection vectors (normalized, at PROJECTION_LAYER)
    dim_vecs = {}
    for dim in DIMS:
        v = cvs[dim].directions[PROJECTION_LAYER].astype(np.float32)
        dim_vecs[dim] = v / np.linalg.norm(v)

    # Wrap model for control injection
    ctrl_model = ControlModel(model, shared_layers)

    # State
    state = np.zeros(5)
    velocity = np.zeros(5)
    raw_proj_history = []
    trajectory = [state.copy()]
    generations = []

    print(f"\nStarting {len(CONVERSATION)}-turn closed-loop session")
    print(f"Parameters: η={ETA}, momentum={MOMENTUM}, layer={PROJECTION_LAYER}")
    print(f"{'='*60}\n")

    for turn_idx, user_msg in enumerate(CONVERSATION):
        print(f"── Turn {turn_idx + 1} ──────────────────────────────────────")
        print(f"User: {user_msg}")

        # Step 1: Extract hidden state of user message → semantic pressure
        encoded = tokenizer(user_msg, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ctrl_model.reset()
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)

        hs = outputs.hidden_states[PROJECTION_LAYER][0, -1].cpu().float().numpy()
        hs_norm = hs / np.linalg.norm(hs)
        raw_proj = np.array([float(np.dot(hs_norm, dim_vecs[d])) for d in DIMS])
        raw_proj_history.append(raw_proj)

        # Mean-center using running history
        all_projs = np.array(raw_proj_history)
        mean = all_projs.mean(axis=0)
        std = all_projs.std(axis=0)
        std[std < 1e-6] = 1.0
        pressure = (raw_proj - mean) / std

        # Step 2: Apply drift dynamics
        velocity = MOMENTUM * velocity + (1 - MOMENTUM) * pressure
        new_state = state + ETA * velocity

        # Step 3: Envelope clipping
        clipped = False
        for j, dim in enumerate(DIMS):
            lo, hi = ENVELOPE[dim]
            if new_state[j] < lo:
                new_state[j] = lo
                velocity[j] *= -0.3
                clipped = True
            elif new_state[j] > hi:
                new_state[j] = hi
                velocity[j] *= -0.3
                clipped = True

        state = new_state
        trajectory.append(state.copy())

        pressure_str = " ".join(f"{DIM_SHORT[j]}:{pressure[j]:+.2f}" for j in range(5))
        state_str = " ".join(f"{DIM_SHORT[j]}:{state[j]:+.3f}" for j in range(5))
        print(f"  Pressure: {pressure_str}")
        print(f"  State:    {state_str}" + (" [CLIPPED]" if clipped else ""))

        # Step 4: Inject control vectors with current coefficients
        ctrl_model.reset()
        for j, dim in enumerate(DIMS):
            if abs(state[j]) > 0.01:
                ctrl_model.set_control(cvs[dim], coeff=float(state[j]))

        # Step 5: Generate response
        chat = [{"role": "user", "content": user_msg}]
        chat_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        gen_input = tokenizer(chat_text, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            gen_out = ctrl_model.model.generate(
                **gen_input,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_time = time.time() - t0

        response = tokenizer.decode(gen_out[0][gen_input["input_ids"].shape[1]:], skip_special_tokens=True)
        tri_rep = compute_trigram_rep(response)

        print(f"  Model: {response[:200]}{'...' if len(response) > 200 else ''}")
        print(f"  [trigram_rep={tri_rep:.3f}, gen_time={gen_time:.1f}s, tokens={len(gen_out[0]) - gen_input['input_ids'].shape[1]}]")
        print()

        generations.append({
            "turn": turn_idx + 1,
            "user": user_msg,
            "response": response,
            "trigram_rep": tri_rep,
            "pressure": {DIMS[j]: float(pressure[j]) for j in range(5)},
            "state": {DIMS[j]: float(state[j]) for j in range(5)},
            "gen_time": gen_time,
        })

    # Save results
    results = {
        "params": {"eta": ETA, "momentum": MOMENTUM, "projection_layer": PROJECTION_LAYER},
        "envelope": ENVELOPE,
        "trajectory": [
            {DIMS[j]: float(trajectory[i][j]) for j in range(5)}
            for i in range(len(trajectory))
        ],
        "generations": generations,
    }
    out_path = Path("/cache/zhangjing/Joi/phase3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    avg_tri = np.mean([g["trigram_rep"] for g in generations])
    max_tri = max(g["trigram_rep"] for g in generations)
    print(f"Average trigram_rep: {avg_tri:.4f} (threshold: 0.05)")
    print(f"Max trigram_rep:     {max_tri:.4f}")
    print(f"Envelope violations: {'None' if not any(g.get('clipped') for g in generations) else 'Yes'}")

    final = trajectory[-1]
    print(f"\nFinal personality state:")
    for j, dim in enumerate(DIMS):
        lo, hi = ENVELOPE[dim]
        pct = (final[j] - lo) / (hi - lo) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {DIM_SHORT[j]:>10}: {final[j]:+.3f}  [{bar}] ({pct:.0f}% of envelope)")

    ctrl_model.unwrap()
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
