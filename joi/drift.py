"""
DriftEngine — the core personality dynamics.

    s(t+1) = clip(s(t) + η · v(t), Envelope)
    v(t)   = momentum · v(t-1) + (1 - momentum) · u(t)

Five lines of math. No if-else. No external judge.
Conversation is the only input. Envelope is the only constraint.
"""

from __future__ import annotations

import numpy as np

from .envelope import Envelope
from .state import JoiState, DIMS


class DriftEngine:
    """
    Stateful personality drift engine.
    
    Maintains a JoiState and applies drift dynamics on each step.
    Designed for real-time use in a conversation loop.
    """

    def __init__(self, state: JoiState, envelope: Envelope):
        self.state = state
        self.envelope = envelope

    def step(self, pressure: np.ndarray) -> dict:
        """
        Apply one drift step given semantic pressure u(t).
        
        Returns dict with step details for logging/visualization.
        """
        s = self.state.as_array()
        v = self.state.velocity_array()
        eta = self.state.eta
        mom = self.state.momentum

        # Dynamics
        v_new = mom * v + (1 - mom) * pressure
        s_new = s + eta * v_new

        # Envelope constraint
        s_clipped, v_clipped, was_clipped = self.envelope.clip(s_new, v_new)

        # Update state
        self.state.set_from_array(s_clipped)
        self.state.set_velocity_from_array(v_clipped)
        self.state.record_turn(
            pressure={DIMS[i]: float(pressure[i]) for i in range(5)},
            clipped=was_clipped,
        )

        return {
            "coefficients": {DIMS[i]: float(s_clipped[i]) for i in range(5)},
            "velocity": {DIMS[i]: float(v_clipped[i]) for i in range(5)},
            "pressure": {DIMS[i]: float(pressure[i]) for i in range(5)},
            "clipped": was_clipped,
            "turn": self.state.turn_count,
            "envelope_util": self.envelope.utilization(s_clipped),
        }

    def reset(self, keep_baseline: bool = True) -> None:
        """Reset personality to origin. Optionally keep projection baseline."""
        for d in DIMS:
            self.state.coefficients[d] = 0.0
            self.state.velocity[d] = 0.0
        self.state.turn_count = 0
        self.state.trajectory_digest.clear()
        if not keep_baseline:
            self.state.projection_count = 0
            for d in DIMS:
                self.state.projection_mean[d] = 0.0
                self.state.projection_std[d] = 1.0
