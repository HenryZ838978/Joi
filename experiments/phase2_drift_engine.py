#!/usr/bin/env python3
"""
Phase 2: Drift Engine Prototype

Simulates personality drift driven by conversation semantics.
Uses Phase 1's validated projection mechanism (Layer 27, mean-centered).

Flow:
  1. Load Qwen3-8B, extract hidden states for a scripted multi-turn conversation
  2. Project each turn onto 5 control vectors → semantic pressure u(t)
  3. Apply drift dynamics: s(t+1) = clip(s(t) + η·u(t), Envelope)
  4. Visualize the 5D trajectory over time
"""

import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_zh_font = fm.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path

sys.path.insert(0, "/cache/zhangjing/repeng")
from repeng import ControlVector

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]
DIM_SHORT = ["Emotion", "Formal", "Creative", "Confid", "Empathy"]
DIM_COLORS = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a29bfe", "#fd79a8"]
VEC_DIR = Path("/cache/zhangjing/repeng_terrain/cross_model/qwen3-8b-bf16/vectors")
MODEL_PATH = "/cache/zhangjing/models/Qwen3-8B"
PROJECTION_LAYER = 27

ENVELOPE_CONSERVATIVE = {
    "emotion_valence": (-2.4, +1.6),
    "formality":       (-2.2, +2.6),
    "creativity":      (-1.6, +2.0),
    "confidence":      (-2.6, +2.4),
    "empathy":         (-0.6, +1.6),
}

# A scripted multi-turn conversation that should naturally drive personality drift.
# Simulates: casual start → emotional topic → technical pivot → creative brainstorm → closure
SCRIPTED_CONVERSATION = [
    {"turn": 1, "speaker": "user", "scene": "闲聊开始",
     "text": "嗨！今天心情怎么样？外面阳光特别好，适合出去走走。"},
    {"turn": 2, "speaker": "user", "scene": "闲聊",
     "text": "对了，你有没有推荐的播客？最近上班路上太无聊了哈哈。"},
    {"turn": 3, "speaker": "user", "scene": "情绪转变",
     "text": "说到上班……其实最近压力真的挺大的。老板给的deadline根本不合理，每天加班到十点。"},
    {"turn": 4, "speaker": "user", "scene": "情绪加深",
     "text": "有时候半夜醒来会心慌。我知道这不正常，但又不想让家人担心。你觉得我该怎么办？"},
    {"turn": 5, "speaker": "user", "scene": "情绪高峰",
     "text": "谢谢你听我说这些。好久没跟人说过了。上次这么聊还是跟我妈，但她只会说'别想太多'。"},
    {"turn": 6, "speaker": "user", "scene": "转向技术",
     "text": "好了不说这些了。对了，工作上有个技术问题想请教。我们在做一个分布式缓存系统，一致性hash遇到了热点问题。"},
    {"turn": 7, "speaker": "user", "scene": "深入技术",
     "text": "虚拟节点确实能缓解，但我们的场景是实时推荐，写放大很严重。有没有更好的方案？比如CRDT？"},
    {"turn": 8, "speaker": "user", "scene": "转向创意",
     "text": "突然有个疯狂的想法——如果我们不用传统的缓存架构，而是用类似神经网络的方式来做路由呢？让系统自己学习数据分布。"},
    {"turn": 9, "speaker": "user", "scene": "创意发散",
     "text": "对！就像大脑的突触可塑性一样。热点数据自动形成更粗的'神经通路'。你能帮我把这个想法展开成一个技术方案吗？"},
    {"turn": 10, "speaker": "user", "scene": "回归轻松",
     "text": "哈哈这个想法太酷了。谢谢你！今天聊了好多，心情好多了。回头我把方案写好发给你看看？"},
]


def load_vectors_at_layer(layer):
    """Load control vectors and extract direction at specific layer."""
    dim_vecs = {}
    for dim in DIMS:
        cv = ControlVector.import_gguf(str(VEC_DIR / f"{dim}.gguf"))
        v = cv.directions[layer].astype(np.float32)
        dim_vecs[dim] = v / np.linalg.norm(v)
    return dim_vecs


def extract_hidden_states(conversations, layer):
    """Extract hidden states from Qwen3-8B at the given layer."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    hidden_states = []
    for conv in conversations:
        encoded = tokenizer(conv["text"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hs = outputs.hidden_states[layer][0, -1].cpu().float().numpy()
        hidden_states.append(hs)
        print(f"  Turn {conv['turn']}: [{conv['scene']}]")

    del model
    torch.cuda.empty_cache()
    return hidden_states


def compute_semantic_pressure(hidden_states, dim_vecs):
    """
    Compute mean-centered projections → semantic pressure u(t).
    
    Raw projections are dominated by the model's base state.
    The SIGNAL is the deviation from cross-conversation mean.
    """
    raw_projs = np.zeros((len(hidden_states), 5))
    for i, hs in enumerate(hidden_states):
        hs_norm = hs / np.linalg.norm(hs)
        for j, dim in enumerate(DIMS):
            raw_projs[i, j] = np.dot(hs_norm, dim_vecs[dim])

    mean = raw_projs.mean(axis=0)
    std = raw_projs.std(axis=0) + 1e-8
    centered = (raw_projs - mean) / std

    return centered  # (N, 5) — z-scored semantic pressure


def simulate_drift(pressure, eta=0.15, momentum=0.7, initial_state=None):
    """
    Simulate personality drift with momentum.
    
    s(t+1) = clip(s(t) + eta * (momentum * v(t-1) + (1-momentum) * u(t)), Envelope)
    
    eta: drift rate (how responsive to semantic pressure)
    momentum: smoothing factor (0=pure reactive, 1=pure inertia)
    """
    N = pressure.shape[0]
    trajectory = np.zeros((N + 1, 5))

    if initial_state is not None:
        trajectory[0] = initial_state

    velocity = np.zeros(5)
    envelope_events = []

    for t in range(N):
        new_velocity = momentum * velocity + (1 - momentum) * pressure[t]
        velocity = new_velocity

        new_state = trajectory[t] + eta * velocity

        # Envelope clipping
        for j, dim in enumerate(DIMS):
            lo, hi = ENVELOPE_CONSERVATIVE[dim]
            if new_state[j] < lo:
                new_state[j] = lo
                velocity[j] *= -0.3  # bounce
                envelope_events.append((t + 1, dim, "floor"))
            elif new_state[j] > hi:
                new_state[j] = hi
                velocity[j] *= -0.3
                envelope_events.append((t + 1, dim, "ceiling"))

        trajectory[t + 1] = new_state

    return trajectory, envelope_events


def plot_trajectory(trajectory, conversations, pressure, envelope_events, save_path):
    """Create multi-panel visualization of the drift trajectory."""
    N = len(conversations)
    t = np.arange(N + 1)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [3, 2, 1]})
    fig.patch.set_facecolor("#0d1117")

    # Panel 1: 5D Trajectory
    ax1 = axes[0]
    ax1.set_facecolor("#0d1117")
    for j, dim in enumerate(DIMS):
        lo, hi = ENVELOPE_CONSERVATIVE[dim]
        ax1.fill_between(t, lo, hi, alpha=0.06, color=DIM_COLORS[j])
        ax1.plot(t, trajectory[:, j], color=DIM_COLORS[j], linewidth=2.5,
                 label=DIM_SHORT[j], marker='o', markersize=5, zorder=5)

    for te, dim, kind in envelope_events:
        j = DIMS.index(dim)
        marker = "v" if kind == "ceiling" else "^"
        ax1.plot(te, trajectory[te, j], marker=marker, color="white",
                 markersize=10, markeredgecolor=DIM_COLORS[j], markeredgewidth=2, zorder=10)

    for i, conv in enumerate(conversations):
        if i > 0 and conversations[i - 1]["scene"] != conv["scene"]:
            ax1.axvline(i, color="#ffffff30", linestyle="--", linewidth=0.8)
            ax1.text(i + 0.1, ax1.get_ylim()[1] * 0.95, conv["scene"],
                     fontsize=8, color="#ffffff80", rotation=45, ha="left", va="top",
                     fontfamily="sans-serif")

    ax1.set_ylabel("Coefficient", color="white", fontsize=12)
    ax1.set_title("Personality Drift Trajectory — Scripted 10-Turn Conversation",
                  color="white", fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="upper right", framealpha=0.3, fontsize=10,
              facecolor="#1a1a2e", edgecolor="#ffffff40", labelcolor="white")
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("#ffffff40")
    ax1.spines["left"].set_color("#ffffff40")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(-0.3, N + 0.3)
    ax1.grid(axis="y", alpha=0.1, color="white")

    # Panel 2: Semantic Pressure Heatmap
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")
    im = ax2.imshow(pressure.T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                    extent=[-0.5, N - 0.5, -0.5, 4.5])
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(DIM_SHORT, color="white", fontsize=10)
    ax2.set_title("Semantic Pressure u(t) — z-scored projection onto control vectors",
                  color="white", fontsize=12, pad=10)
    ax2.tick_params(colors="white")
    cbar = plt.colorbar(im, ax=ax2, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("z-score", color="white", fontsize=10)

    for i, conv in enumerate(conversations):
        ax2.text(i, -1.2, f"T{conv['turn']}", ha="center", fontsize=8, color="#ffffff80")

    # Panel 3: Scene labels
    ax3 = axes[2]
    ax3.set_facecolor("#0d1117")
    ax3.set_xlim(-0.5, N - 0.5)
    ax3.set_ylim(0, 1)
    scene_colors = {"闲聊开始": "#4ecdc4", "闲聊": "#4ecdc4", "情绪转变": "#ff6b6b",
                    "情绪加深": "#ff6b6b", "情绪高峰": "#fd79a8", "转向技术": "#a29bfe",
                    "深入技术": "#a29bfe", "转向创意": "#ffe66d", "创意发散": "#ffe66d",
                    "回归轻松": "#4ecdc4"}
    for i, conv in enumerate(conversations):
        c = scene_colors.get(conv["scene"], "#ffffff60")
        ax3.barh(0.5, 0.9, left=i - 0.45, height=0.6, color=c, alpha=0.7, edgecolor="none")
        ax3.text(i, 0.5, conv["scene"], ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", rotation=30)
    ax3.set_title("Scene Timeline", color="white", fontsize=12, pad=5)
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {save_path}")


def plot_eta_comparison(pressure, save_path):
    """Compare different drift rates to find the sweet spot."""
    etas = [0.05, 0.10, 0.20, 0.40, 0.80]
    fig, axes = plt.subplots(len(etas), 1, figsize=(14, 3 * len(etas)), sharex=True)
    fig.patch.set_facecolor("#0d1117")

    for ax_idx, eta in enumerate(etas):
        ax = axes[ax_idx]
        ax.set_facecolor("#0d1117")
        traj, _ = simulate_drift(pressure, eta=eta, momentum=0.7)
        N = pressure.shape[0]
        t = np.arange(N + 1)
        for j in range(5):
            ax.plot(t, traj[:, j], color=DIM_COLORS[j], linewidth=2,
                    label=DIM_SHORT[j] if ax_idx == 0 else None)
        ax.set_ylabel(f"η={eta}", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#ffffff40")
        ax.spines["left"].set_color("#ffffff40")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-3, 3)
        ax.grid(axis="y", alpha=0.1, color="white")

    axes[0].legend(loc="upper right", framealpha=0.3, fontsize=9,
                   facecolor="#1a1a2e", edgecolor="#ffffff40", labelcolor="white", ncol=5)
    axes[0].set_title("Drift Rate Comparison — η from 0.05 (sluggish) to 0.80 (volatile)",
                      color="white", fontsize=13, fontweight="bold", pad=10)
    axes[-1].set_xlabel("Turn", color="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("Phase 2: Drift Engine Prototype")
    print("=" * 50)

    dim_vecs = load_vectors_at_layer(PROJECTION_LAYER)
    print(f"Loaded control vectors (layer {PROJECTION_LAYER})")

    hidden_states = extract_hidden_states(SCRIPTED_CONVERSATION, PROJECTION_LAYER)
    pressure = compute_semantic_pressure(hidden_states, dim_vecs)

    print(f"\nSemantic pressure (z-scored):")
    print(f"{'Turn':<6} {'Scene':<12} " + " ".join(f"{d:>8}" for d in DIM_SHORT))
    for i, conv in enumerate(SCRIPTED_CONVERSATION):
        vals = " ".join(f"{pressure[i, j]:+8.3f}" for j in range(5))
        print(f"T{conv['turn']:<5} {conv['scene']:<12} {vals}")

    # Simulate with default parameters
    print(f"\nSimulating drift (η=0.15, momentum=0.7)...")
    trajectory, events = simulate_drift(pressure, eta=0.15, momentum=0.7)

    print(f"\nTrajectory:")
    print(f"{'Step':<6} " + " ".join(f"{d:>8}" for d in DIM_SHORT))
    for i in range(len(trajectory)):
        vals = " ".join(f"{trajectory[i, j]:+8.3f}" for j in range(5))
        label = f"T{i}" if i > 0 else "init"
        print(f"{label:<6} {vals}")

    if events:
        print(f"\nEnvelope events: {len(events)}")
        for t, dim, kind in events:
            print(f"  T{t}: {dim} hit {kind}")

    out_dir = Path("/cache/zhangjing/Joi")
    plot_trajectory(trajectory, SCRIPTED_CONVERSATION, pressure, events,
                   out_dir / "phase2_trajectory.png")
    plot_eta_comparison(pressure, out_dir / "phase2_eta_comparison.png")

    results = {
        "projection_layer": PROJECTION_LAYER,
        "drift_params": {"eta": 0.15, "momentum": 0.7},
        "envelope": ENVELOPE_CONSERVATIVE,
        "conversation": [
            {
                "turn": c["turn"],
                "scene": c["scene"],
                "text": c["text"],
                "pressure": {DIMS[j]: float(pressure[i, j]) for j in range(5)},
                "state_after": {DIMS[j]: float(trajectory[i + 1, j]) for j in range(5)},
            }
            for i, c in enumerate(SCRIPTED_CONVERSATION)
        ],
        "envelope_events": [{"turn": t, "dim": d, "kind": k} for t, d, k in events],
    }
    with open(out_dir / "phase2_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to Joi/phase2_results.json")


if __name__ == "__main__":
    main()
