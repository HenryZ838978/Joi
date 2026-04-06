#!/usr/bin/env python3
"""
Experiment D: Cross-session continuity.

Verifies that saving and restoring a Joi checkpoint produces
identical trajectory continuation. No generation needed — pure
projection + drift math.
"""

import sys, json, yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, "/cache/zhangjing/repeng")
from repeng import ControlVector

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

ETA, MOM = 0.15, 0.7


def get_hidden(model, tokenizer, text, layer):
    import torch
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer][0, -1].cpu().float().numpy()
    del out
    return h


def drift_step(state, velocity, pressure):
    velocity = MOM * velocity + (1 - MOM) * pressure
    state = state + ETA * velocity
    for j, dim in enumerate(DIMS):
        lo, hi = ENVELOPE[dim]
        state[j] = np.clip(state[j], lo, hi)
    return state.copy(), velocity.copy()


def project(model, tokenizer, text, dim_vecs, raw_history):
    hs = get_hidden(model, tokenizer, text, PROJECTION_LAYER)
    hs_norm = hs / np.linalg.norm(hs)
    raw = np.array([float(np.dot(hs_norm, dim_vecs[d])) for d in DIMS])
    raw_history.append(raw)
    if len(raw_history) < 2:
        return np.zeros(5)
    all_raw = np.array(raw_history)
    mean = all_raw.mean(axis=0)
    std = all_raw.std(axis=0)
    std[std < 1e-6] = 1.0
    return (raw - mean) / std


def save_checkpoint(filepath, state, velocity, raw_history, trajectory, turn):
    checkpoint = {
        "version": "0.1",
        "model": "Qwen3-8B",
        "turn": int(turn),
        "eta": float(ETA),
        "momentum": float(MOM),
        "state": {DIMS[i]: float(state[i]) for i in range(5)},
        "velocity": {DIMS[i]: float(velocity[i]) for i in range(5)},
        "raw_history": [r.tolist() for r in raw_history],
        "trajectory": [t.tolist() for t in trajectory],
    }
    with open(filepath, "w") as f:
        yaml.dump(checkpoint, f, default_flow_style=False, allow_unicode=True)
    return checkpoint


def load_checkpoint(filepath):
    with open(filepath) as f:
        cp = yaml.safe_load(f)
    state = np.array([cp["state"][d] for d in DIMS])
    velocity = np.array([cp["velocity"][d] for d in DIMS])
    raw_history = [np.array(r) for r in cp["raw_history"]]
    trajectory = [np.array(t) for t in cp["trajectory"]]
    return state, velocity, raw_history, trajectory, cp["turn"]


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import gc

    print("Exp D: Cross-Session Continuity (projection-only, no generation)")
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
    dim_vecs = {}
    for dim in DIMS:
        v = cvs[dim].directions[PROJECTION_LAYER].astype(np.float32)
        dim_vecs[dim] = v / np.linalg.norm(v)

    ALL_TURNS = [
        "嗨，今天心情怎么样？",
        "最近工作压力好大，每天加班到十点。",
        "有时候觉得自己什么都做不好。",
        "谢谢你听我说这些。",
        "对了有个技术问题，CRDT和Raft哪个好？",
        "突然有个灵感——把数据库做成大脑突触！",
        "哈哈太酷了，开心了好多。",
        "说真的，你觉得我该不该跳槽？",
        "好吧，再想想。给我讲个笑话？",
        "谢谢你，今天聊得很开心～",
    ]

    SPLIT = 5
    checkpoint_path = OUT / "exp_d_checkpoint.yaml"

    # ── Run 1: Full continuous (ground truth) ─────────────
    print("\n--- GROUND TRUTH: 10 turns continuous ---")
    state = np.zeros(5)
    vel = np.zeros(5)
    raw_hist = []
    traj = [state.copy()]

    for i, msg in enumerate(ALL_TURNS):
        pressure = project(model, tokenizer, msg, dim_vecs, raw_hist)
        state, vel = drift_step(state, vel, pressure)
        traj.append(state.copy())
        s_str = " ".join(f"{d[:4]}={state[j]:+.4f}" for j, d in enumerate(DIMS))
        print(f"  T{i+1:>2}: {s_str}")
        gc.collect()
        torch.cuda.empty_cache()

    traj_gt = np.array(traj)

    # ── Run 2: Split at SPLIT, save & restore ─────────────
    print(f"\n--- SPLIT RUN: T1-{SPLIT} → SAVE → T{SPLIT+1}-10 ---")
    state = np.zeros(5)
    vel = np.zeros(5)
    raw_hist = []
    traj = [state.copy()]

    for i, msg in enumerate(ALL_TURNS[:SPLIT]):
        pressure = project(model, tokenizer, msg, dim_vecs, raw_hist)
        state, vel = drift_step(state, vel, pressure)
        traj.append(state.copy())
        s_str = " ".join(f"{d[:4]}={state[j]:+.4f}" for j, d in enumerate(DIMS))
        print(f"  T{i+1:>2}: {s_str}")
        gc.collect(); torch.cuda.empty_cache()

    save_checkpoint(checkpoint_path, state, vel, raw_hist, traj, SPLIT)
    print(f"  [SAVED to {checkpoint_path}]")

    # Clear all state
    del state, vel, raw_hist, traj
    print("  [STATE CLEARED]")

    # Restore
    state, vel, raw_hist, traj, turn_offset = load_checkpoint(checkpoint_path)
    print(f"  [RESTORED from checkpoint, turn={turn_offset}]")

    for i, msg in enumerate(ALL_TURNS[SPLIT:]):
        pressure = project(model, tokenizer, msg, dim_vecs, raw_hist)
        state, vel = drift_step(state, vel, pressure)
        traj.append(state.copy())
        s_str = " ".join(f"{d[:4]}={state[j]:+.4f}" for j, d in enumerate(DIMS))
        print(f"  T{turn_offset+i+1:>2}: {s_str}")
        gc.collect(); torch.cuda.empty_cache()

    traj_split = np.array(traj)

    # ── Compare ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    min_len = min(len(traj_gt), len(traj_split))
    diff = np.abs(traj_gt[:min_len] - traj_split[:min_len])

    print(f"\n  Points: GT={len(traj_gt)}, Split={len(traj_split)}")
    print(f"  Max difference:  {diff.max():.2e}")
    print(f"  Mean difference: {diff.mean():.2e}")

    print(f"\n  {'Turn':<6} {'Diff L2':>10} {'Note':>10}")
    for i in range(min_len):
        d = np.linalg.norm(traj_gt[i] - traj_split[i])
        note = "← SEAM" if i == SPLIT else ""
        print(f"  T{i:<5} {d:>10.2e} {note:>10}")

    perfect = diff.max() < 1e-10
    print(f"\n  Result: {'✓ PERFECT CONTINUITY' if perfect else '✗ DIVERGED'}")
    print(f"  (max diff = {diff.max():.2e})")

    # Save
    result = {
        "ground_truth": traj_gt.tolist(),
        "split_restored": traj_split.tolist(),
        "max_diff": float(diff.max()),
        "is_perfect": bool(perfect),
    }
    with open(OUT / "exp_d_continuity.json", "w") as f:
        json.dump(result, f, indent=2)

    # Show the checkpoint for reference
    print(f"\n  Checkpoint saved at: {checkpoint_path}")
    with open(checkpoint_path) as f:
        print(f"  Checkpoint content (first 20 lines):")
        for i, line in enumerate(f):
            if i >= 20: break
            print(f"    {line.rstrip()}")

    del model
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
