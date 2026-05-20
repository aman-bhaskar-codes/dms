"""
PERCLOS V2 — Weighted, Adaptive Baseline

V2 improvements:
  - Adaptive per-driver closure threshold (calibrated from 75th percentile EAR)
  - Weighted PERCLOS: longer closures count more than threshold crossings
  - Sub-windows: 10s, 30s, 60s PERCLOS for trend detection
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict
import config


class PERCLOSEngine:
    def __init__(self, fps: int = config.TARGET_FPS):
        self.fps          = fps
        self.window_size  = config.PERCLOS_WINDOW_SEC * fps
        self.buffer       = deque(maxlen=self.window_size)   # weighted closure values
        self.binary_buf   = deque(maxlen=self.window_size)   # simple binary closed/open

        # Sub-windows
        self._buf_10s = deque(maxlen=fps * 10)
        self._buf_30s = deque(maxlen=fps * 30)

        self.perclos     = 0.0
        self.perclos_10s = 0.0
        self.perclos_30s = 0.0
        self.state       = "normal"

        # Adaptive baseline
        self._baseline_samples = deque(maxlen=fps * 10)
        self._open_baseline    = config.EAR_OPEN_BASELINE

        print(f"[PERCLOSEngine V2] window={config.PERCLOS_WINDOW_SEC}s  "
              f"alert>{config.PERCLOS_ALERT_THRESH*100:.0f}%")

    def update(self, ear: float) -> Tuple[float, str]:
        # Update adaptive baseline
        if ear > 0.28:
            self._baseline_samples.append(ear)
            if len(self._baseline_samples) >= 30:
                self._open_baseline = float(
                    np.percentile(list(self._baseline_samples), 75))

        closure_thresh = self._open_baseline * config.PERCLOS_CLOSURE_RATIO
        is_closed = ear < closure_thresh

        # Weighted value: deeper/longer closures count more
        if is_closed:
            depth = max(0.0, closure_thresh - ear) / (closure_thresh + 1e-6)
            weighted_val = 1.0 + depth  # 1.0..2.0 range
        else:
            weighted_val = 0.0

        self.buffer.append(weighted_val)
        self.binary_buf.append(1 if is_closed else 0)
        self._buf_10s.append(1 if is_closed else 0)
        self._buf_30s.append(1 if is_closed else 0)

        if len(self.binary_buf) < 30:
            return 0.0, "calibrating"

        # Standard PERCLOS (binary)
        self.perclos = float(np.mean(list(self.binary_buf)))
        # Sub-windows
        if len(self._buf_10s) >= 30:
            self.perclos_10s = float(np.mean(list(self._buf_10s)))
        if len(self._buf_30s) >= 30:
            self.perclos_30s = float(np.mean(list(self._buf_30s)))

        if self.perclos >= config.PERCLOS_ALERT_THRESH:
            self.state = "drowsy"
        elif self.perclos >= config.PERCLOS_WARN_THRESH:
            self.state = "warning"
        else:
            self.state = "normal"

        return self.perclos, self.state

    @property
    def trend(self) -> str:
        """'rising', 'falling', or 'stable' based on 10s vs 60s comparison."""
        if self.perclos_10s > self.perclos + 0.04:
            return "rising"
        elif self.perclos_10s < self.perclos - 0.04:
            return "falling"
        return "stable"

    @property
    def open_baseline(self) -> float:
        return self._open_baseline

    def fatigue_components(self) -> Dict[str, float]:
        return {
            "perclos_60s": min(1.0, self.perclos / config.PERCLOS_ALERT_THRESH),
            "perclos_10s": min(1.0, self.perclos_10s / config.PERCLOS_ALERT_THRESH),
            "trend_rising": 1.0 if self.trend == "rising" else 0.0,
        }

    def reset(self):
        self.buffer.clear()
        self.binary_buf.clear()
        self._buf_10s.clear()
        self._buf_30s.clear()
        self.perclos = 0.0
        self.state   = "normal"
