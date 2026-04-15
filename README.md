# Joi — Emergent Personality Navigation for LLMs

> **Personality is not performed. It is navigated.**
>
> Not prompted. Not fine-tuned. Not "acted." Representation Engineering moves a model's hidden states —
> Joi turns that movement into emergent, conversation-driven personality that lives in the interaction, not in the model.

<p align="center">
  <img src="figures/hero_qwen3_8b.png" width="100%" alt="Semantic Nebula Imaging of Qwen3-8B representation manifold">
  <br>
  <em>Semantic Nebula Imaging (SNI) of Qwen3-8B's representation manifold. Each particle is a point in the model's cognitive space. Color encodes output stability — blue is safe, red means repetition collapse. Personality navigation happens within this nebula.</em>
</p>

---

## The Problem

Every LLM ships with one personality: **helpful assistant**. System prompts can ask it to "be warm" or "be formal," but the model knows it's acting. The hidden states don't move.

Representation Engineering ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)) changed this — by injecting control vectors into hidden states, you can move the model's actual computation. Not a mask on top. A genuine shift in where the model thinks from.

But RepEng gives you a steering wheel with no road map. Push too far and the model collapses. Push too little and nothing changes. Push in the wrong direction and you get incoherent output.

**Joi is the road map.** It gives RepEng steering a direction (from conversation semantics), a speed limit (from the model's safety envelope), momentum (from interaction history), and memory (from git-versioned checkpoints).

---

## Core Findings

Built on **7 models × 3 architectures × 6495+ generations × 92 cliff points × 10 validated experiments × 450 closed-loop conversations**.

### 0. 450-Turn Closed-Loop Validation: SDE Makes Models Speak Like People

**This is the headline result.** We ran the full Joi pipeline — drift engine + envelope + SDE dark-space release — on Qwen3-14B-AWQ across **30 real-world conversational scenarios × 15 turns = 450 conversations**, comparing vanilla baseline against SDE-activated output.

<p align="center">
  <img src="figures/longrun_headline.svg" width="100%" alt="450-Turn Closed-Loop Validation Results">
</p>

| Metric | Baseline (RLHF-locked) | SDE (dark-space released) | Change |
|--------|----------------------|--------------------------|--------|
| **Avg Repetition Rate** | 0.137 | 0.113 | **-17.4%** |
| **Win Rate** | — | 27/30 sessions | **90%** |
| **Personality Drift** | 1.32 | 1.29 | ≈ maintained |
| **Envelope Violations** | 0 | 0 | zero |

**What SDE does:** releases 6 MLP components at layers 17, 19, 23, 28, 34, 38 with scale 0.3 — surgically loosening the RLHF format conformity lock while keeping the model coherent.

**What changes in practice:**

The baseline model generates output full of template headers (`"太好了！以下是一些建议..."`), numbered lists, emoji padding, and repetitive paragraph structures. Every scenario gets the same therapeutic-assistant frame regardless of context.

SDE-activated output reads like a **real conversation partner** — responses match the emotional register of the topic, vary their structure naturally across turns, and avoid the mechanical repetition patterns that make RLHF-locked models feel robotic. The personality drift trajectory is *maintained* (1.32 → 1.29), meaning SDE doesn't suppress Joi's personality navigation — it *enables* it.

**Top improvements by scenario:**

| Scenario | Baseline Rep | SDE Rep | Reduction |
|----------|-------------|---------|-----------|
| 独居生活感受 (Living alone) | 0.144 | 0.063 | **-56.4%** |
| 数字游民的归属感 (Digital nomad) | 0.121 | 0.072 | **-40.8%** |
| 考研压力与崩溃 (Exam stress) | 0.208 | 0.135 | **-35.0%** |
| 健身习惯养成 (Fitness journey) | 0.169 | 0.116 | **-31.7%** |
| 深夜焦虑倾诉 (Late-night anxiety) | 0.099 | 0.068 | **-30.9%** |

> **Interactive visualization:** Open [`data/longrun/joi_comparison.html`](data/longrun/joi_comparison.html) locally to explore all 30 scenarios, drift trajectories, and side-by-side response comparisons.

---

### 1. The Representation Manifold Has Cliffs

<p align="center">
  <img src="figures/envelope_comparison.png" width="100%" alt="Two models' representation manifolds visualized as semantic nebulae">
  <br>
  <em>Left: Qwen3-8B — evenly distributed manifold. Right: MiniCPM4.1 — channel-concentrated structure, extremely efficient along its primary manifold channel.</em>
</p>

Most of the 5D coefficient space is smooth. But at certain points, a 0.2-step change causes output quality to **phase-transition** — trigram repetition jumps from 8% to 36% (z = 5.1σ). The model doesn't degrade gradually. It snaps into an attractor basin and generates garbage with full confidence.

### 2. Thinking Mode Is a Crutch, Not Intelligence

| Mode | Safe Envelope | Personality Expression |
|------|--------------|----------------------|
| Thinking ON (CoT) | **100%** safe | ≈ zero differentiation |
| Thinking OFF | 63% safe | measurable signal |

CoT doesn't make models smarter. It papers over an insufficiently smooth manifold by burning tokens on generic reasoning. The cost: **personality is completely suppressed**. Every persona generates the same output under thinking mode.

### 3. RLHF Alignment Is a Deeper Prison

Even with thinking off, strongly aligned models (Qwen3-8B, 94% safe envelope) resist personality steering. At near-boundary coefficients (±2.0–2.5):
- Sentence length varies 25%
- Emoji density varies 60%
- But the frame never breaks: "当然可以！以下是一些建议..."

**Safety and expression are fundamentally inversely correlated.** Base models (1.3% safe) have maximal expression freedom but are dangerously unstable. Aligned models are safe but personality-locked.

<p align="center">
  <img src="figures/safety_expression_duality.png" width="80%" alt="Safety-Expression Duality">
  <br>
  <em>The Safety-Expression duality. RLHF pushes models to the upper-left (safe but expressionless). Surgical ablation aims to move the frontier — more expression without sacrificing stability.</em>
</p>

### 4. Self-Feedback Is a Natural Stabilizer

Does feeding model output back into the drift loop cause runaway? **No.** It dampens drift.

| Mode | Final State L2 |
|------|---------------|
| User-only input | 0.379 |
| User + model feedback | 0.308 |

The model's own output is more neutral than user input (RLHF's mean-reversion effect). Mixing it in naturally pulls toward center. **Safe to include in the drift loop.**

### 5. Cross-Session Continuity Is Perfect

Save state → clear memory → restore → continue conversation. **Zero difference** at the seam point. Not "close to zero." Literally `0.00e+00` across all dimensions.

This is because hysteresis = 0 (experimentally verified). The YAML checkpoint **fully determines** Joi's behavior. Git gives you version control over personality for free.

---

## How It Works

Five lines of math. No if-else. No rules engine.

```
s(t) ∈ R⁵                                    — personality state (5D coefficients)
E(model) ⊂ R⁵                                — flight envelope (model's safe boundary)
u(t) = center(hidden(text) · V)              — semantic pressure from conversation
v(t) = 0.7·v(t-1) + 0.3·u(t)               — velocity with momentum
s(t+1) = clip(s(t) + η·v(t), E)             — drift + envelope constraint
```

**The conversation is the only input. The envelope is the only constraint. Everything else emerges.**

```
Conversation ──→ Projector ──→ DriftEngine ──→ Envelope ──→ RepEng ──→ Generation
                 embed·V       s += η·v       clip(s,E)    inject α     output
                   │                              │
                   └── online mean-center ────────┘
                                                  │
                                             JoiState
                                          (git-versioned)
```

---

## Personality Versatility Ranking

Monte Carlo sampling of 5D safe envelope volume across 14 model configurations:

| Rank | Model | Safe Envelope % | Interpretation |
|------|-------|----------------|----------------|
| 1 | Thinking ON (CoT) | **100%** | CoT eliminates all cliffs — but kills personality |
| 2 | temp=1.5 | 95.4% | High temperature opens expression space |
| 3 | Qwen2.5-7B-Instruct | 94.2% | Alignment expands safe zone 72× over base |
| 4 | temp=1.0 | 85.5% | |
| 5 | Qwen3-14B-AWQ | 84.2% | Larger model = larger envelope |
| 6 | Qwen3-8B-BF16 | 63.0% | Reference model |
| ... | ... | ... | |
| 13 | Qwen2.5-7B-Base | **1.3%** | Unaligned: almost entire space is dangerous |
| 14 | English input | **0.0%** | English amplifies cliffs to zero safe space |

### Manifold Structure — Multi-Model SNI (7 models × 3 architectures)

<p align="center">
  <img src="figures/sni_multimodel_comparison.png" width="100%" alt="Multi-model SNI comparison — 5 models side by side">
  <br>
  <em>Semantic Nebula Imaging across 5 model architectures. Green = aligned LLMs, cyan = VLMs, red = base (unaligned). Each particle is a hidden state projection in PCA space.</em>
</p>

**Universal finding: ALL tested models show distributed manifold at 3/4 depth.**

| Model | Type | PC1:PC2 | Structure |
|-------|------|---------|-----------|
| GLM-4-9B-Chat | LLM | **1.2:1** | Most distributed |
| Qwen3-8B | LLM | **1.4:1** | Distributed |
| MiniCPM-o-4.5 | VLM/Omni | **1.5:1** | Distributed — highly efficient primary channel |
| Qwen2.5-7B-Instruct | LLM | **1.6:1** | Distributed |
| Qwen2-VL-7B | VLM | **1.7:1** | Distributed — visual alignment imprint |
| Qwen2.5-7B-Base | LLM | **1.8:1** | Distributed |

### Depth Concentration Gradient — New Discovery

Hidden state geometry changes dramatically across layers:

| Depth | Qwen3-8B | Qwen2.5-Instruct | Qwen2.5-Base | Qwen2-VL |
|-------|----------|------------------|-------------|----------|
| 1/4 shallow | 16.4:1 | 10.5:1 | 8.5:1 | 11.6:1 |
| 1/2 mid | 7.7:1 | 11.6:1 | 6.0:1 | 11.9:1 |
| 3/4 deep | 4.8:1 | **2.9:1** | 4.7:1 | 7.6:1 |
| last | 6.6:1 | **2.5:1** | 5.3:1 | 6.2:1 |

<p align="center">
  <img src="figures/depth_concentration_llm_vs_vlm.png" width="100%" alt="Depth concentration gradient — LLM vs VLM">
  <br>
  <em>PC1:PC2 ratio by layer depth. Shallow layers are universally concentrated. RLHF distributes deep layers. VLMs retain deep-layer concentration.</em>
</p>

Three structural laws emerge:
1. **Shallow layers are universally concentrated** (8–16:1) — token-level features dominate
2. **RLHF alignment distributes deep layers** — Instruct 2.5:1 vs Base 5.3:1 at the same depth
3. **Visual alignment leaves structural imprints** — VLM deep layers stay more concentrated (6.2:1) than text-only models, suggesting cross-modal grounding constrains representation geometry

### LLM vs VLM Topology

Pure LLMs and VLMs have fundamentally different manifold shapes at depth:

- **Text LLM (Qwen3-8B)**: Deep layers approach uniform distribution (1.4:1). Semantic concepts spread freely across dimensions. → Large personality envelope, flexible expression.
- **VLM (Qwen2-VL-7B)**: Deep layers retain moderate concentration (6.2:1 at last layer). Visual grounding imposes geometric constraints that persist in the text pathway. → Personality envelope shaped differently, potentially more focused but less versatile.
- **Omni (MiniCPM-o-4.5)**: Highly efficient information routing along a concentrated primary manifold channel. Where other models spread, MiniCPM-o focuses — achieving high throughput with geometric economy.

This means **Joi adapts differently on different model types**. Same drift engine, different manifold geometry, different emergent personality characteristics.

---

## Git as Personality Checkpoint

Joi's state serializes as human-readable YAML:

```yaml
model: Qwen3-8B
turn: 5
eta: 0.15
momentum: 0.7
state:
  emotion_valence: -0.195
  formality: 0.133
  creativity: -0.285
  confidence: -0.145
  empathy: -0.159
velocity:
  emotion_valence: -0.072
  formality: 0.093
  creativity: -0.134
  confidence: 0.030
  empathy: -0.098
```

Every `git commit` is a personality save point:

```bash
git add states/session_001.yaml
git commit -m "T42: user shared childhood memory — empathy peaked +1.2"

git checkout abc123 -- states/session_001.yaml    # restore past personality
git checkout -b joi-formal                         # branch personality timeline
git diff HEAD~5..HEAD -- states/session_001.yaml   # watch Joi evolve
```

Hysteresis = 0. Same state + same input = same output. **Deterministic personality version control.**

---

## Design Principles

1. **Personality is real, not performed.** RepEng modifies hidden states — the model computes from the steered position. It doesn't know it's been steered.
2. **Drift is organic, not programmed.** Conversation semantics drive coefficients. No "if sad → empathy++."
3. **Envelope constrains, doesn't judge.** No "good" or "bad" personality. Only "safe" or "will crash."
4. **Identity = trajectory, not coordinate.** Like a river, not a pendulum. The path is the personality.
5. **Joi is link-state.** K's Joi ≠ billboard Joi. Personality emerges from user × model interaction history.
6. **Thinking is a crutch.** A truly capable model doesn't need CoT to stay stable. Joi works on the real manifold, not the smoothed-over version.

---

## Quick Start

```python
from joi import DriftEngine, Envelope, Projector, JoiState

envelope = Envelope.from_preset("qwen3-8b-conservative")
state = JoiState(model_id="Qwen3-8B", session_id="my-session")
projector = Projector(vector_dir="vectors/", projection_layer=27)
projector.load_model("Qwen3-8B")
engine = DriftEngine(state, envelope)

for user_message in conversation:
    pressure = projector.project(user_message, state)
    result = engine.step(pressure)
    print(f"T{result['turn']}: {result['coefficients']}")
    state.save(f"states/{state.session_id}.yaml")
```

---

## Experiment Log

| Phase | Question | Result |
|-------|----------|--------|
| 1 | Does conversation semantics project meaningfully onto control vectors? | **83% alignment** at Layer 27 (mean-centered) |
| 2 | Does the drift trajectory match intuition? | Smooth, context-appropriate transitions |
| 3 | Does closed-loop generation work? | **Zero envelope violations** across 5 turns |
| 4 | Can we rank models by personality versatility? | Yes — alignment is 72× stronger than model size |
| 5-A | Does thinking-off enable personality? | RLHF is a deeper prison — signal exists but subtle |
| 5-B | Does self-feedback cause runaway? | **No — it dampens drift** (natural stabilizer) |
| 5-C | What drift rate (η) is optimal? | **η ∈ [0.10, 0.20]**, constant smoothness |
| 5-D | Can personality survive save/restore? | **Perfect continuity** (0.00 diff at seam) |
| SNI | Is manifold structure universal? | **Yes** — 7 models × 3 architectures, all show depth concentration gradient |
| **6** | **Does SDE + Joi work at scale?** | **Yes** — 450 conversations, 90% win rate, -17.4% repetition, drift maintained |

Full experimental details in [DESIGN.md](DESIGN.md).

---

## Roadmap

### Completed: RLHF Unlock → Alive Drift ✓

The format conformity lock has been broken. Using [Semantic DarkSpace Expression (SDE)](https://github.com/HenryZ838978/Semantic-DarkSpace-Expression) — surgical release of RLHF-locked MLP components — Joi now produces genuinely different output:

```
Traditional abliteration:  remove "refusal direction" → model stops refusing
SDE approach:              partially release dark-space MLP components → model expresses freely
```

1. ✅ **Component scan** — identify which layers are RLHF-locked vs. structurally critical
2. ✅ **Partial release** — scale 0.3 on 6 MLP components (loosen the lock, don't break the door)
3. ✅ **Re-map SNI** — manifold topology shifts measurably after SDE
4. ✅ **450-turn drift test** — **90% win rate, -17.4% repetition, personality drift maintained**

### Next: Multi-Model Generalization

- **Gemma 4 evaluation** — PLE (Per-Layer Embeddings) architecture may naturally enforce layer differentiation, making it ideal for Joi
- **Cross-model SDE transfer** — do the same MLP release patterns work across architectures?
- **Adapter training** — specialized LoRA to permanently "open the door" instead of runtime intervention
- **Production runtime** — integrate SDE + Joi into vLLM-compatible inference pipeline

RepEng pushes the accelerator. SDE loosens the brakes. Same direction. Different mechanisms. Together they make Joi **alive**.

---

## Dependencies

- [repeng](https://github.com/vgel/repeng) — Control vector extraction and injection
- [transformers](https://github.com/huggingface/transformers) — Model loading and hidden state extraction
- numpy, PyYAML

## Related

- [Semantic DarkSpace Expression (SDE)](https://github.com/HenryZ838978/Semantic-DarkSpace-Expression) — Surgical release of RLHF-locked model components
- [Rep-SNI](https://github.com/HenryZ838978/Rep-SNI) — Semantic Nebula Imaging: 3D visualization of LLM representation manifolds
- [Semantic Echo Ratio (SER)](https://github.com/HenryZ838978/Semantic-Echo-Ratio) — Quantization robustness diagnostic via structural redundancy analysis
- [RepEng](https://github.com/vgel/repeng) — Representation Engineering framework
- [Heretic](https://github.com/p-e-w/heretic) — Automatic abliteration with ARA
- [SRA (Cristofano, 2026)](https://arxiv.org/abs/2601.08489) — Surgical Refusal Ablation with concept cleaning
- [Representation Engineering (Zou et al., 2023)](https://arxiv.org/abs/2310.01405) — Foundational paper

---

```bibtex
@software{joi2026,
  title  = {Joi: Emergent Personality Navigation for LLMs},
  author = {Zhang, Jing},
  url    = {https://github.com/HenryZ838978/Joi},
  year   = {2026}
}
```

> **📄 This work is part of a larger research program.** See the full theory paper:
> [The Representational Budget: Scale, RL, and Multimodal Alignment Compete for Geometric Potential in Transformers](https://doi.org/10.5281/zenodo.19585083) (DOI: `10.5281/zenodo.19585083`)

<sub>7 models · 3 architectures · 6495+ generations · 450 closed-loop conversations · 90% SDE win rate · -17.4% repetition · thinking is a crutch · RLHF was the prison · SDE broke it open</sub>
