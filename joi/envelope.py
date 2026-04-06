"""
Flight Envelope — safe operating boundaries for personality coefficients.

Derived from terrain scans of 14 models (6045 generations, 92 cliff points).
The envelope constrains drift to prevent output degradation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]

PRESETS = {
    "qwen3-8b-conservative": {
        "emotion_valence": (-2.4, +1.6),
        "formality":       (-2.2, +2.6),
        "creativity":      (-1.6, +2.0),
        "confidence":      (-2.6, +2.4),
        "empathy":         (-0.6, +1.6),
    },
    "qwen3-8b-permissive": {
        "emotion_valence": (-2.8, +2.0),
        "formality":       (-2.8, +3.0),
        "creativity":      (-2.4, +2.6),
        "confidence":      (-3.0, +2.8),
        "empathy":         (-1.0, +2.0),
    },
}


@dataclass
class Envelope:
    """5D flight envelope with per-dimension bounds and pair constraints."""

    bounds: dict = field(default_factory=lambda: dict(PRESETS["qwen3-8b-conservative"]))
    pair_constraints: list = field(default_factory=list)
    bounce_factor: float = 0.3

    @classmethod
    def from_preset(cls, name: str) -> Envelope:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
        return cls(bounds=dict(PRESETS[name]))

    @classmethod
    def from_terrain(cls, terrain_path: str | Path, threshold: float = 0.05) -> Envelope:
        """Build envelope from terrain data by finding per-dimension safe ranges."""
        path = Path(terrain_path)
        with open(path) as f:
            data = json.load(f)

        bounds = {}
        if isinstance(data, dict) and "sweeps" in data:
            for dim in DIMS:
                sweep = data["sweeps"].get(dim, [])
                safe_vals = []
                for pt in sweep:
                    queries = pt.get("queries", {})
                    tri_vals = []
                    for qdata in queries.values():
                        if isinstance(qdata, dict) and "metrics" in qdata:
                            t = qdata["metrics"].get("trigram_rep")
                            if t is not None:
                                tri_vals.append(float(t))
                    if tri_vals and np.mean(tri_vals) < threshold:
                        safe_vals.append(pt["value"])
                if safe_vals:
                    bounds[dim] = (min(safe_vals), max(safe_vals))
                else:
                    bounds[dim] = (0.0, 0.0)
        else:
            for dim in DIMS:
                bounds[dim] = (-3.0, 3.0)

        return cls(bounds=bounds)

    def clip(self, state: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Clip state to envelope, with velocity bounce on boundary contact.
        
        Returns: (clipped_state, adjusted_velocity, was_clipped)
        """
        clipped = False
        new_state = state.copy()
        new_vel = velocity.copy()

        for i, dim in enumerate(DIMS):
            lo, hi = self.bounds.get(dim, (-3.0, 3.0))
            if new_state[i] < lo:
                new_state[i] = lo
                new_vel[i] *= -self.bounce_factor
                clipped = True
            elif new_state[i] > hi:
                new_state[i] = hi
                new_vel[i] *= -self.bounce_factor
                clipped = True

        return new_state, new_vel, clipped

    def utilization(self, state: np.ndarray) -> dict:
        """How much of each dimension's range is being used (0-1)."""
        util = {}
        for i, dim in enumerate(DIMS):
            lo, hi = self.bounds.get(dim, (-3.0, 3.0))
            if hi > lo:
                util[dim] = (state[i] - lo) / (hi - lo)
            else:
                util[dim] = 0.5
        return util

    def volume(self) -> float:
        """Compute the hyperrectangular volume of the envelope."""
        vol = 1.0
        for dim in DIMS:
            lo, hi = self.bounds.get(dim, (0, 0))
            vol *= max(hi - lo, 0)
        return vol

    def contains(self, state: np.ndarray) -> bool:
        for i, dim in enumerate(DIMS):
            lo, hi = self.bounds.get(dim, (-3.0, 3.0))
            if state[i] < lo or state[i] > hi:
                return False
        return True
