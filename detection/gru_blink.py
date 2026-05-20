"""
GRU-based temporal blink detector.
Trained on sequences of EAR values to detect:
  - Normal blinks
  - Slow blinks (drowsy)
  - Micro-sleeps (eyes closed >500ms)

Uses a tiny 2-layer GRU — runs on CPU in <1ms.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


class BlinkGRU(nn.Module):
    """
    Lightweight GRU for real-time blink classification.
    Input:  EAR sequence (T=30 frames, 1 feature)
    Output: 3-class softmax [normal, slow_blink, microsleep]
    """
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])  # last timestep


@dataclass
class BlinkEvent:
    event_type: str   # "normal" | "slow" | "microsleep"
    confidence: float
    duration_ms: float
    ear_nadir: float  # minimum EAR during blink


class TemporalBlinkDetector:
    """
    Wraps BlinkGRU with a rolling EAR window.
    Falls back to threshold if model not found.
    """
    SEQ_LEN = 30
    CLASSES = ["normal", "slow_blink", "microsleep"]

    def __init__(self, model_path: str = "models/blink_gru.pt"):
        self._window = deque(maxlen=self.SEQ_LEN)
        self._model: Optional[BlinkGRU] = None
        self._load_model(model_path)

    def _load_model(self, path: str):
        try:
            self._model = BlinkGRU()
            self._model.load_state_dict(torch.load(path, map_location="cpu"))
            self._model.eval()
            print("[BlinkGRU] Model loaded.")
        except FileNotFoundError:
            print(f"[BlinkGRU] Model not found at {path}. Using threshold fallback.")

    def update(self, ear: float) -> Optional[BlinkEvent]:
        self._window.append(ear)
        if len(self._window) < self.SEQ_LEN:
            return None

        if self._model is not None:
            return self._predict_gru()
        else:
            return self._predict_threshold(ear)

    def _predict_gru(self) -> Optional[BlinkEvent]:
        seq = np.array(self._window, dtype=np.float32).reshape(1, self.SEQ_LEN, 1)
        with torch.no_grad():
            probs = self._model(torch.from_numpy(seq))[0].numpy()
        class_idx = int(probs.argmax())
        confidence = float(probs[class_idx])

        if confidence < 0.6 or self.CLASSES[class_idx] == "normal":
            return None

        return BlinkEvent(
            event_type=self.CLASSES[class_idx],
            confidence=confidence,
            duration_ms=self.SEQ_LEN * (1000 / 30),  # approx at 30fps
            ear_nadir=float(min(self._window))
        )

    def _predict_threshold(self, ear: float) -> Optional[BlinkEvent]:
        """Simple threshold fallback when model not available."""
        nadir = min(self._window)
        closed_frames = sum(1 for e in self._window if e < 0.22)
        if closed_frames > 15:
            return BlinkEvent("microsleep", 0.95, closed_frames * 33.3, nadir)
        elif closed_frames > 5:
            return BlinkEvent("slow_blink", 0.80, closed_frames * 33.3, nadir)
        return None
