"""
Projector — maps conversation text to semantic pressure in 5D control space.

The core validated mechanism:
  u(t) = center(hidden_state(text) · control_vectors)

Phase 1 showed 83% alignment at Layer 27 after mean-centering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]


class Projector:
    """
    Projects text into 5D personality pressure space via LLM hidden states.
    
    Requires:
      - Control vectors (GGUF files from RepEng training)
      - The same LLM used to train those vectors (for hidden state extraction)
    """

    def __init__(
        self,
        vector_dir: str | Path,
        projection_layer: int = 27,
    ):
        self.vector_dir = Path(vector_dir)
        self.projection_layer = projection_layer
        self.dim_vecs = self._load_vectors()
        self._model = None
        self._tokenizer = None

    def _load_vectors(self) -> dict:
        import sys
        sys.path.insert(0, str(self.vector_dir.parent.parent.parent / "repeng"))
        try:
            from repeng import ControlVector
        except ImportError:
            raise ImportError("repeng not found. Install from: https://github.com/vgel/repeng")

        dim_vecs = {}
        for dim in DIMS:
            path = self.vector_dir / f"{dim}.gguf"
            if not path.exists():
                raise FileNotFoundError(f"Control vector not found: {path}")
            cv = ControlVector.import_gguf(str(path))
            v = cv.directions[self.projection_layer].astype(np.float32)
            dim_vecs[dim] = v / np.linalg.norm(v)
        return dim_vecs

    def load_model(self, model_path: str, device_map: str = "auto") -> None:
        """Load the LLM for hidden state extraction."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map=device_map,
        )
        self._model.eval()

    def project(self, text: str, state: Optional["JoiState"] = None) -> np.ndarray:
        """
        Compute semantic pressure u(t) for a text input.
        
        Returns raw projection if no state (no centering).
        If state is provided, uses its running baseline for online centering.
        """
        if self._model is None:
            raise RuntimeError("Call load_model() first")

        import torch

        encoded = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**encoded, output_hidden_states=True, use_cache=False)

        hs = outputs.hidden_states[self.projection_layer][0, -1].cpu().float().numpy()
        hs_norm = hs / np.linalg.norm(hs)

        raw = np.array([float(np.dot(hs_norm, self.dim_vecs[d])) for d in DIMS])

        if state is not None:
            state.update_projection_baseline(raw)
            if state.projection_count >= 2:
                mean = np.array([state.projection_mean[d] for d in DIMS])
                std = np.array([state.projection_std[d] for d in DIMS])
                std = np.maximum(std, 1e-6)
                return (raw - mean) / std
            else:
                return np.zeros(5)  # not enough data to center yet
        return raw

    def unload_model(self) -> None:
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            torch.cuda.empty_cache()
