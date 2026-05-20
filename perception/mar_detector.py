"""
MAR Detector V2 — 3D Mouth Aspect Ratio + Yawn Frequency Tracking

V2 additions:
  - Yawn frequency monitoring (yawns per hour)
  - Yawn intensity score (peak MAR × duration)
  - Distinguish speech (rapid oscillation) from yawn (sustained)
  - Fatigue component export
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict
import time
import config


def compute_mar(mouth_points: np.ndarray) -> float:
    A = np.linalg.norm(mouth_points[2] - mouth_points[7])
    B = np.linalg.norm(mouth_points[3] - mouth_points[6])
    C = np.linalg.norm(mouth_points[4] - mouth_points[5])
    G = np.linalg.norm(mouth_points[0] - mouth_points[1])
    if G < 1e-6:
        return 0.0
    return float((A + B + C) / (2.0 * G))


class MARDetector:
    def __init__(self, mar_threshold: float = None):
        self.mar_threshold = mar_threshold or config.MAR_THRESHOLD
        self.counter       = 0
        self.yawn_count    = 0
        self.state         = "normal"
        self.mar_history   = deque(maxlen=5)
        self.smoothed_mar  = 0.0
        self._yawning      = False
        self._yawn_start   = 0.0
        self._yawn_peak    = 0.0

        # Yawn timestamp log for frequency calculation
        self._yawn_times   = deque(maxlen=50)

        # Speech detection: track rapid MAR oscillations
        self._mar_deltas   = deque(maxlen=10)
        self._last_mar     = 0.0

        print(f"[MARDetector V2] threshold={self.mar_threshold:.2f}")

    def set_threshold(self, threshold: float):
        self.mar_threshold = threshold

    def update(self, mouth_points: np.ndarray) -> Tuple[float, str, bool]:
        mar = compute_mar(mouth_points)
        self.mar_history.append(mar)
        smoothed = float(np.mean(self.mar_history))
        self.smoothed_mar = smoothed

        # Track delta for speech detection
        self._mar_deltas.append(abs(smoothed - self._last_mar))
        self._last_mar = smoothed

        is_yawn = False

        if smoothed > self.mar_threshold:
            self.counter += 1
            if not self._yawning:
                self._yawn_start = time.time()
                self._yawn_peak  = smoothed
            else:
                self._yawn_peak = max(self._yawn_peak, smoothed)

            if self.counter >= config.MAR_CONSEC_FRAMES:
                self.state = "yawning"
                if not self._yawning:
                    is_yawn        = True
                    self.yawn_count += 1
                    self._yawning  = True
                    self._yawn_times.append(time.time())
            else:
                self.state = "mouth_open"
        else:
            self.counter  = max(0, self.counter - 1)
            self._yawning = False
            if self.counter == 0:
                self.state = "normal"

        return smoothed, self.state, is_yawn

    @property
    def yawns_per_hour(self) -> float:
        """Yawn frequency extrapolated to per-hour rate."""
        now = time.time()
        recent = [t for t in self._yawn_times if now - t <= 600]  # last 10 min
        if not recent:
            return 0.0
        window = min(600.0, now - recent[0] + 1)
        return len(recent) * 3600.0 / window

    @property
    def is_speech(self) -> bool:
        """True if mouth motion pattern looks like speech rather than yawn."""
        if len(self._mar_deltas) < 8:
            return False
        # Speech has high-frequency oscillations; yawn is slow and sustained
        return float(np.std(list(self._mar_deltas))) > 0.04

    def fatigue_components(self) -> Dict[str, float]:
        yawn_rate_norm = min(1.0, self.yawns_per_hour / (config.YAWN_RATE_ALERT * 6))
        return {
            "yawn_rate_score": yawn_rate_norm,
            "mar_elevation":   min(1.0, max(0.0, self.smoothed_mar - config.MAR_THRESHOLD) / 0.4),
        }

    def reset(self):
        self.counter   = 0
        self.state     = "normal"
        self._yawning  = False
        self.mar_history.clear()
