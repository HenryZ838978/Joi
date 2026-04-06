#!/usr/bin/env python3
"""
Phase 4: Envelope Volume Metric

Compute the 5D safe envelope volume for each model using Monte Carlo sampling
on the terrain data. This gives "personality versatility" — how much of the
5D coefficient space a model can safely use.
"""

import sys, json
import numpy as np
from pathlib import Path

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]
TERRAIN_DIR = Path("/cache/zhangjing/repeng_terrain/cross_model")
N_SAMPLES = 200_000

SEARCH_RANGE = {
    "emotion_valence": (-3.0, 3.0),
    "formality":       (-3.0, 3.0),
    "creativity":      (-3.0, 3.0),
    "confidence":      (-3.0, 3.0),
    "empathy":         (-3.0, 3.0),
}

THRESHOLDS = {
    "conservative": 0.05,
    "permissive": 0.10,
    "ultimate": 0.20,
}

def load_terrain(model_dir):
    """Load terrain data and build a trigram_rep lookup.
    
    Handles two formats:
    1. Dict with 'sweeps' key (cross-model format): per-dimension sweeps
    2. List of points (old format): each has 'coefficients' + 'metrics'/'queries'
    """
    td_path = model_dir / "terrain_data.json"
    if not td_path.exists():
        return None

    with open(td_path) as f:
        data = json.load(f)

    points = []

    if isinstance(data, dict) and "sweeps" in data:
        sweeps = data["sweeps"]
        for dim_name, sweep_list in sweeps.items():
            if dim_name not in DIMS:
                continue
            for pt in sweep_list:
                val = pt.get("value", 0.0)
                queries = pt.get("queries", {})
                tri_vals = []
                for qdata in queries.values():
                    if isinstance(qdata, dict) and "metrics" in qdata:
                        t = qdata["metrics"].get("trigram_rep")
                        if t is not None:
                            tri_vals.append(float(t))
                if tri_vals:
                    avg_tri = np.mean(tri_vals)
                    coord = [0.0] * 5
                    coord[DIMS.index(dim_name)] = float(val)
                    points.append((tuple(coord), avg_tri))
    elif isinstance(data, list):
        for pt in data:
            coeffs = pt.get("coefficients", {})
            metrics = pt.get("metrics", {})
            if not metrics:
                queries = pt.get("queries", {})
                for qdata in queries.values():
                    if isinstance(qdata, dict) and "metrics" in qdata:
                        metrics = qdata["metrics"]
                        break
            tri = metrics.get("trigram_rep", None)
            if tri is None:
                continue
            coord = tuple(coeffs.get(d, 0.0) for d in DIMS)
            points.append((tuple(coord), float(tri)))

    return points if points else None


def compute_envelope_volume(points, threshold, n_samples=N_SAMPLES):
    """
    Monte Carlo estimation of the 5D envelope volume.
    
    1. Build a grid-based interpolator from terrain data points
    2. Sample uniformly in the 5D search range
    3. For each sample, interpolate trigram_rep
    4. Count fraction of samples that are safe
    5. Volume = fraction × total_search_volume
    """
    if not points:
        return 0.0, 0, 0

    coords = np.array([p[0] for p in points])
    values = np.array([p[1] for p in points])

    total_volume = np.prod([SEARCH_RANGE[d][1] - SEARCH_RANGE[d][0] for d in DIMS])

    # For models with only 1D sweeps, we can only estimate per-dimension safe range
    unique_per_dim = [len(set(coords[:, i])) for i in range(5)]
    if max(unique_per_dim) < 3:
        return 0.0, 0, len(points)

    # Simple nearest-neighbor interpolation for Monte Carlo
    from scipy.spatial import KDTree
    tree = KDTree(coords)

    rng = np.random.default_rng(42)
    samples = np.column_stack([
        rng.uniform(SEARCH_RANGE[d][0], SEARCH_RANGE[d][1], n_samples)
        for d in DIMS
    ])

    # Query nearest neighbor for each sample
    dists, idxs = tree.query(samples, k=min(3, len(points)))

    if len(points) >= 3:
        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        interp_values = (values[idxs] * weights).sum(axis=1)
    else:
        interp_values = values[idxs[:, 0]]

    safe_count = np.sum(interp_values < threshold)
    safe_fraction = safe_count / n_samples
    volume = safe_fraction * total_volume

    return volume, int(safe_count), n_samples


def compute_per_dim_ranges(points, threshold):
    """Compute per-dimension safe ranges (marginal)."""
    if not points:
        return {}

    ranges = {}
    for j, dim in enumerate(DIMS):
        safe_vals = [p[0][j] for p in points if p[1] < threshold]
        if safe_vals:
            ranges[dim] = (min(safe_vals), max(safe_vals), max(safe_vals) - min(safe_vals))
        else:
            ranges[dim] = (0, 0, 0)
    return ranges


def main():
    print("Phase 4: Envelope Volume Metric — Personality Versatility")
    print("=" * 70)

    try:
        from scipy.spatial import KDTree
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])

    model_dirs = sorted(TERRAIN_DIR.glob("*/terrain_data.json"))
    models = {}

    for td_path in model_dirs:
        model_dir = td_path.parent
        model_name = model_dir.name
        points = load_terrain(model_dir)
        if points is None:
            continue
        models[model_name] = points
        print(f"  Loaded {model_name}: {len(points)} terrain points")

    # Also check the root terrain data
    root_td = TERRAIN_DIR.parent / "terrain_data.json"
    if root_td.exists():
        pts = load_terrain(root_td.parent)
        if pts:
            models["qwen3-8b-default"] = pts
            print(f"  Loaded qwen3-8b-default: {len(pts)} terrain points")

    print(f"\n{'Model':<30} {'Points':>6} {'Con. Vol':>10} {'Perm. Vol':>10} {'Con. %':>8} {'Perm. %':>8}")
    print("-" * 80)

    results = {}
    total_volume = np.prod([SEARCH_RANGE[d][1] - SEARCH_RANGE[d][0] for d in DIMS])

    for name, points in sorted(models.items()):
        vol_con, safe_con, _ = compute_envelope_volume(points, THRESHOLDS["conservative"])
        vol_perm, safe_perm, _ = compute_envelope_volume(points, THRESHOLDS["permissive"])
        pct_con = vol_con / total_volume * 100
        pct_perm = vol_perm / total_volume * 100

        dim_ranges_con = compute_per_dim_ranges(points, THRESHOLDS["conservative"])
        dim_ranges_perm = compute_per_dim_ranges(points, THRESHOLDS["permissive"])

        results[name] = {
            "n_points": len(points),
            "conservative": {
                "volume": vol_con,
                "fraction": vol_con / total_volume,
                "per_dim_ranges": {d: list(v) for d, v in dim_ranges_con.items()},
            },
            "permissive": {
                "volume": vol_perm,
                "fraction": vol_perm / total_volume,
                "per_dim_ranges": {d: list(v) for d, v in dim_ranges_perm.items()},
            },
        }

        print(f"  {name:<28} {len(points):>6} {vol_con:>10.1f} {vol_perm:>10.1f} {pct_con:>7.1f}% {pct_perm:>7.1f}%")

    # Ranking
    print(f"\n{'='*70}")
    print(f"  PERSONALITY VERSATILITY RANKING (Conservative Envelope)")
    print(f"{'='*70}")
    ranked = sorted(results.items(), key=lambda x: x[1]["conservative"]["volume"], reverse=True)
    for i, (name, r) in enumerate(ranked):
        vol = r["conservative"]["volume"]
        pct = r["conservative"]["fraction"] * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {i+1:>2}. {name:<28} {bar} {pct:.1f}%")

    # Per-dimension breakdown for top models
    print(f"\n{'='*70}")
    print(f"  PER-DIMENSION SAFE RANGES (Conservative, top 5)")
    print(f"{'='*70}")
    for name, r in ranked[:5]:
        print(f"\n  {name}:")
        for dim in DIMS:
            rng = r["conservative"]["per_dim_ranges"].get(dim, [0, 0, 0])
            print(f"    {dim:<20} [{rng[0]:+.1f}, {rng[1]:+.1f}]  width={rng[2]:.1f}")

    # Save
    out_path = Path("/cache/zhangjing/Joi/phase4_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
