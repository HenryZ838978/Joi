#!/usr/bin/env python3
"""
Generate interactive HTML visualization for Phase 2 drift trajectory.
Self-contained HTML — no server needed.
"""
import json
from pathlib import Path

DATA_PATH = Path("/cache/zhangjing/Joi/phase2_results.json")
OUT_PATH = Path("/cache/zhangjing/Joi/phase2_drift_viz.html")

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]
DIM_SHORT = ["Emotion", "Formality", "Creativity", "Confidence", "Empathy"]
DIM_COLORS = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a29bfe", "#fd79a8"]

ENVELOPE = {
    "emotion_valence": [-2.4, 1.6],
    "formality": [-2.2, 2.6],
    "creativity": [-1.6, 2.0],
    "confidence": [-2.6, 2.4],
    "empathy": [-0.6, 1.6],
}

data = json.loads(DATA_PATH.read_text())

turns_js = json.dumps(data["conversation"], ensure_ascii=False)
dims_js = json.dumps(DIMS)
dim_short_js = json.dumps(DIM_SHORT)
dim_colors_js = json.dumps(DIM_COLORS)
envelope_js = json.dumps(ENVELOPE)

html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Joi Phase 2 — Personality Drift Trajectory</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  background: #0d1117; color: #e6edf3; font-family: -apple-system, sans-serif;
  display: flex; flex-direction: column; min-height: 100vh; padding: 24px;
}}
h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 8px; }}
h1 span {{ color: #fd79a8; }}
.subtitle {{ color: #8b949e; font-size: 13px; margin-bottom: 20px; }}
.grid {{ display: grid; grid-template-columns: 1fr 340px; gap: 20px; flex: 1; }}
.chart-panel {{ background: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d; }}
.side-panel {{ display: flex; flex-direction: column; gap: 16px; }}
.info-card {{ background: #161b22; border-radius: 12px; padding: 16px; border: 1px solid #30363d; }}
.info-card h3 {{ font-size: 13px; color: #8b949e; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }}
canvas {{ width: 100%; cursor: crosshair; }}
.turn-list {{ max-height: 320px; overflow-y: auto; }}
.turn-item {{
  padding: 8px 12px; border-radius: 8px; margin-bottom: 6px; cursor: pointer;
  transition: background 0.2s; border-left: 3px solid transparent;
}}
.turn-item:hover {{ background: #1f2937; }}
.turn-item.active {{ background: #1f2937; border-left-color: #fd79a8; }}
.turn-num {{ font-size: 11px; color: #8b949e; }}
.turn-scene {{ font-size: 13px; font-weight: 600; margin: 2px 0; }}
.turn-text {{ font-size: 12px; color: #8b949e; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.dim-bars {{ display: flex; flex-direction: column; gap: 8px; }}
.dim-row {{ display: flex; align-items: center; gap: 8px; }}
.dim-label {{ width: 70px; font-size: 12px; font-weight: 600; text-align: right; }}
.dim-bar-bg {{
  flex: 1; height: 18px; background: #21262d; border-radius: 9px; position: relative; overflow: hidden;
}}
.dim-bar-fill {{ position: absolute; height: 100%; border-radius: 9px; transition: width 0.4s, left 0.4s; opacity: 0.8; }}
.dim-val {{ width: 50px; font-size: 11px; text-align: right; font-family: 'SF Mono', monospace; }}
.pressure-label {{ font-size: 11px; color: #8b949e; }}
.legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 10px; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
.controls {{ display: flex; gap: 12px; align-items: center; margin-top: 12px; }}
.controls button {{
  background: #21262d; border: 1px solid #30363d; color: #e6edf3; padding: 6px 14px;
  border-radius: 6px; cursor: pointer; font-size: 12px;
}}
.controls button:hover {{ background: #30363d; }}
.controls button.playing {{ background: #fd79a8; color: #0d1117; }}
.slider {{ flex: 1; }}
input[type=range] {{
  -webkit-appearance: none; width: 100%; height: 4px; background: #30363d;
  border-radius: 2px; outline: none;
}}
input[type=range]::-webkit-slider-thumb {{
  -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%;
  background: #fd79a8; cursor: pointer;
}}
</style>
</head>
<body>
<h1>Joi <span>Phase 2</span> — Personality Drift Trajectory</h1>
<p class="subtitle">10-turn scripted conversation · η=0.15 · momentum=0.7 · Layer 27 projection · Qwen3-8B</p>

<div class="grid">
  <div class="chart-panel">
    <canvas id="chart" height="400"></canvas>
    <div class="legend" id="legend"></div>
    <div class="controls">
      <button id="playBtn" onclick="togglePlay()">▶ Play</button>
      <div class="slider">
        <input type="range" id="turnSlider" min="0" max="10" value="10" oninput="setTurn(+this.value)">
      </div>
      <span id="turnLabel" style="font-size:12px; min-width:40px;">T10</span>
    </div>
  </div>

  <div class="side-panel">
    <div class="info-card">
      <h3>Current State</h3>
      <div class="dim-bars" id="dimBars"></div>
    </div>
    <div class="info-card">
      <h3>Conversation</h3>
      <div class="turn-list" id="turnList"></div>
    </div>
  </div>
</div>

<script>
const TURNS = {turns_js};
const DIMS = {dims_js};
const DIM_SHORT = {dim_short_js};
const DIM_COLORS = {dim_colors_js};
const ENVELOPE = {envelope_js};

const trajectory = [[0,0,0,0,0]];
TURNS.forEach(t => {{
  const s = DIMS.map(d => t.state_after[d]);
  trajectory.push(s);
}});

let currentTurn = trajectory.length - 1;
let playing = false;
let playTimer = null;

// Build turn list
const turnList = document.getElementById('turnList');
TURNS.forEach((t, i) => {{
  const el = document.createElement('div');
  el.className = 'turn-item';
  el.id = 'turn-' + i;
  el.innerHTML = `<div class="turn-num">Turn ${{t.turn}}</div><div class="turn-scene">${{t.scene}}</div><div class="turn-text">${{t.text}}</div>`;
  el.onclick = () => setTurn(i + 1);
  turnList.appendChild(el);
}});

// Build legend
const legendEl = document.getElementById('legend');
DIM_SHORT.forEach((d, i) => {{
  const lo = ENVELOPE[DIMS[i]][0], hi = ENVELOPE[DIMS[i]][1];
  legendEl.innerHTML += `<div class="legend-item"><div class="legend-dot" style="background:${{DIM_COLORS[i]}}"></div>${{d}} [${{lo}}, ${{hi}}]</div>`;
}});

// Build dim bars
const dimBarsEl = document.getElementById('dimBars');
DIMS.forEach((d, i) => {{
  dimBarsEl.innerHTML += `
    <div class="dim-row">
      <div class="dim-label" style="color:${{DIM_COLORS[i]}}">${{DIM_SHORT[i]}}</div>
      <div class="dim-bar-bg" id="bar-bg-${{i}}"><div class="dim-bar-fill" id="bar-${{i}}" style="background:${{DIM_COLORS[i]}}"></div></div>
      <div class="dim-val" id="val-${{i}}">0.000</div>
    </div>
    <div class="pressure-label" id="pressure-${{i}}" style="text-align:center"></div>`;
}});

function updateBars(turnIdx) {{
  const s = trajectory[turnIdx];
  DIMS.forEach((d, i) => {{
    const lo = ENVELOPE[d][0], hi = ENVELOPE[d][1];
    const range = hi - lo;
    const pct = ((s[i] - lo) / range) * 100;
    const bar = document.getElementById('bar-' + i);
    const zeroPct = ((0 - lo) / range) * 100;
    if (s[i] >= 0) {{
      bar.style.left = zeroPct + '%';
      bar.style.width = Math.max(0, pct - zeroPct) + '%';
    }} else {{
      bar.style.left = pct + '%';
      bar.style.width = Math.max(0, zeroPct - pct) + '%';
    }}
    document.getElementById('val-' + i).textContent = s[i].toFixed(3);
    if (turnIdx > 0) {{
      const p = TURNS[turnIdx - 1].pressure[d];
      const pEl = document.getElementById('pressure-' + i);
      pEl.textContent = `pressure: ${{p >= 0 ? '+' : ''}}${{p.toFixed(2)}}σ`;
      pEl.style.color = p > 0.5 ? DIM_COLORS[i] : p < -0.5 ? '#8b949e' : '#4a5568';
    }}
  }});
}}

// Canvas chart
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
function resizeCanvas() {{
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = 400 * 2;
  ctx.scale(2, 2);
  drawChart();
}}
window.addEventListener('resize', resizeCanvas);

function drawChart() {{
  const W = canvas.offsetWidth, H = 400;
  ctx.clearRect(0, 0, W, H);

  const pad = {{ l: 50, r: 20, t: 20, b: 40 }};
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;

  const yMin = -3, yMax = 3;
  const xToPixel = i => pad.l + (i / (trajectory.length - 1)) * cw;
  const yToPixel = v => pad.t + ((yMax - v) / (yMax - yMin)) * ch;

  // Grid
  ctx.strokeStyle = '#ffffff10';
  ctx.lineWidth = 0.5;
  for (let y = -3; y <= 3; y++) {{
    ctx.beginPath();
    ctx.moveTo(pad.l, yToPixel(y));
    ctx.lineTo(W - pad.r, yToPixel(y));
    ctx.stroke();
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(y.toString(), pad.l - 6, yToPixel(y) + 3);
  }}

  // Zero line
  ctx.strokeStyle = '#ffffff30';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, yToPixel(0));
  ctx.lineTo(W - pad.r, yToPixel(0));
  ctx.stroke();

  // Scene dividers
  const scenes = TURNS.map(t => t.scene);
  for (let i = 1; i < scenes.length; i++) {{
    if (scenes[i] !== scenes[i - 1]) {{
      const x = xToPixel(i);
      ctx.strokeStyle = '#ffffff15';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, H - pad.b);
      ctx.stroke();
      ctx.setLineDash([]);
    }}
  }}

  // Current turn marker
  if (currentTurn > 0 && currentTurn <= TURNS.length) {{
    const x = xToPixel(currentTurn);
    ctx.fillStyle = '#fd79a820';
    ctx.fillRect(x - 2, pad.t, 4, ch);
  }}

  // Envelope bands (faint)
  DIMS.forEach((d, j) => {{
    const lo = ENVELOPE[d][0], hi = ENVELOPE[d][1];
    ctx.fillStyle = DIM_COLORS[j] + '08';
    ctx.fillRect(pad.l, yToPixel(hi), cw, yToPixel(lo) - yToPixel(hi));
  }});

  // Lines
  const maxI = currentTurn;
  DIMS.forEach((d, j) => {{
    ctx.strokeStyle = DIM_COLORS[j];
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= maxI; i++) {{
      const x = xToPixel(i), y = yToPixel(trajectory[i][j]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();

    // Dots
    for (let i = 0; i <= maxI; i++) {{
      const x = xToPixel(i), y = yToPixel(trajectory[i][j]);
      ctx.fillStyle = i === currentTurn ? '#ffffff' : DIM_COLORS[j];
      ctx.beginPath();
      ctx.arc(x, y, i === currentTurn ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
    }}
  }});

  // X labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('init', xToPixel(0), H - pad.b + 14);
  TURNS.forEach((t, i) => {{
    ctx.fillText('T' + t.turn, xToPixel(i + 1), H - pad.b + 14);
  }});
  // Scene labels below
  ctx.font = '9px sans-serif';
  TURNS.forEach((t, i) => {{
    ctx.fillText(t.scene, xToPixel(i + 1), H - pad.b + 26);
  }});
}}

function setTurn(t) {{
  currentTurn = t;
  document.getElementById('turnSlider').value = t;
  document.getElementById('turnLabel').textContent = t === 0 ? 'init' : 'T' + t;
  updateBars(t);
  drawChart();

  // Highlight turn in list
  document.querySelectorAll('.turn-item').forEach((el, i) => {{
    el.classList.toggle('active', i === t - 1);
  }});
}}

function togglePlay() {{
  playing = !playing;
  const btn = document.getElementById('playBtn');
  if (playing) {{
    btn.textContent = '⏸ Pause';
    btn.classList.add('playing');
    currentTurn = 0;
    playTimer = setInterval(() => {{
      if (currentTurn >= trajectory.length - 1) {{
        togglePlay();
        return;
      }}
      setTurn(currentTurn + 1);
    }}, 800);
  }} else {{
    btn.textContent = '▶ Play';
    btn.classList.remove('playing');
    clearInterval(playTimer);
  }}
}}

// Init
setTimeout(resizeCanvas, 100);
setTurn(trajectory.length - 1);
</script>
</body>
</html>"""

OUT_PATH.write_text(html)
print(f"Written: {OUT_PATH}")
