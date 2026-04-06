"""
JoiState — Git-friendly personality state serialization.

The personality state is a YAML file designed for human readability
and git diff friendliness. Each commit to states/ is a personality checkpoint.

Usage:
    state = JoiState.load("states/session_001.yaml")
    state.step(pressure)
    state.save("states/session_001.yaml")
    # then: git add states/ && git commit -m "T42: user shared childhood memory"
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]


@dataclass
class JoiState:
    """Serializable personality state — the 'soul' snapshot."""

    # 5D personality coefficients
    coefficients: dict = field(default_factory=lambda: {d: 0.0 for d in DIMS})

    # Drift velocity (momentum state)
    velocity: dict = field(default_factory=lambda: {d: 0.0 for d in DIMS})

    # Parameters
    eta: float = 0.15
    momentum: float = 0.7

    # Identity metadata
    model_id: str = ""
    session_id: str = ""
    turn_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    # Trajectory summary (compact — full trajectory in separate log)
    trajectory_digest: list = field(default_factory=list)

    # Projection baseline (running mean/std for centering)
    projection_mean: dict = field(default_factory=lambda: {d: 0.0 for d in DIMS})
    projection_std: dict = field(default_factory=lambda: {d: 1.0 for d in DIMS})
    projection_count: int = 0

    def to_dict(self) -> dict:
        return {
            "joi_state_version": "0.1.0",
            "model_id": self.model_id,
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "personality": {
                "coefficients": {d: round(float(self.coefficients[d]), 6) for d in DIMS},
                "velocity": {d: round(float(self.velocity[d]), 6) for d in DIMS},
            },
            "drift_params": {
                "eta": self.eta,
                "momentum": self.momentum,
            },
            "projection_baseline": {
                "mean": {d: round(float(self.projection_mean[d]), 6) for d in DIMS},
                "std": {d: round(float(self.projection_std[d]), 6) for d in DIMS},
                "sample_count": self.projection_count,
            },
            "trajectory_digest": self.trajectory_digest[-20:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> JoiState:
        s = cls()
        s.model_id = d.get("model_id", "")
        s.session_id = d.get("session_id", "")
        s.turn_count = d.get("turn_count", 0)
        s.created_at = d.get("created_at", "")
        s.updated_at = d.get("updated_at", "")

        p = d.get("personality", {})
        s.coefficients = {dim: p.get("coefficients", {}).get(dim, 0.0) for dim in DIMS}
        s.velocity = {dim: p.get("velocity", {}).get(dim, 0.0) for dim in DIMS}

        dp = d.get("drift_params", {})
        s.eta = dp.get("eta", 0.15)
        s.momentum = dp.get("momentum", 0.7)

        pb = d.get("projection_baseline", {})
        s.projection_mean = {dim: pb.get("mean", {}).get(dim, 0.0) for dim in DIMS}
        s.projection_std = {dim: pb.get("std", {}).get(dim, 1.0) for dim in DIMS}
        s.projection_count = pb.get("sample_count", 0)

        s.trajectory_digest = d.get("trajectory_digest", [])
        return s

    def save(self, path: str | Path) -> None:
        """Save state to YAML (preferred) or JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.created_at:
            self.created_at = self.updated_at

        data = self.to_dict()
        if HAS_YAML and path.suffix in (".yaml", ".yml"):
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> JoiState:
        """Load state from YAML or JSON."""
        path = Path(path)
        with open(path) as f:
            if HAS_YAML and path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls.from_dict(data)

    def as_array(self) -> np.ndarray:
        return np.array([self.coefficients[d] for d in DIMS])

    def velocity_array(self) -> np.ndarray:
        return np.array([self.velocity[d] for d in DIMS])

    def set_from_array(self, arr: np.ndarray) -> None:
        for i, d in enumerate(DIMS):
            self.coefficients[d] = float(arr[i])

    def set_velocity_from_array(self, arr: np.ndarray) -> None:
        for i, d in enumerate(DIMS):
            self.velocity[d] = float(arr[i])

    def record_turn(self, pressure: dict, clipped: bool = False) -> None:
        """Append a compact digest entry for this turn."""
        self.turn_count += 1
        entry = {
            "t": self.turn_count,
            "s": {d: round(float(self.coefficients[d]), 3) for d in DIMS},
        }
        if clipped:
            entry["clipped"] = True
        self.trajectory_digest.append(entry)

    def update_projection_baseline(self, raw_projection: np.ndarray) -> None:
        """Online update of projection mean/std (Welford's algorithm)."""
        self.projection_count += 1
        n = self.projection_count
        for i, d in enumerate(DIMS):
            old_mean = self.projection_mean[d]
            new_mean = old_mean + (raw_projection[i] - old_mean) / n
            self.projection_mean[d] = new_mean
            if n > 1:
                old_std = self.projection_std[d]
                new_var = old_std ** 2 * (n - 2) / (n - 1) + (raw_projection[i] - old_mean) * (raw_projection[i] - new_mean) / (n - 1)
                self.projection_std[d] = max(np.sqrt(max(new_var, 0)), 1e-6)

    def __repr__(self) -> str:
        coeffs = " ".join(f"{d[:5]}={self.coefficients[d]:+.3f}" for d in DIMS)
        return f"JoiState(T{self.turn_count} | {coeffs})"
