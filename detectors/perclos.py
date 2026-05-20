"""PERCLOS V2 — weighted closure ratio with adaptive baseline"""
import numpy as np
from collections import deque
from config import settings


class PERCLOSEngine:
    def __init__(self):
        fps = settings.target_fps
        self._window: deque = deque(maxlen=settings.perclos_window_sec * fps)
        self._baseline_ear = settings.ear_open_baseline
        self.perclos = 0.0

    def set_baseline(self, baseline_ear: float):
        self._baseline_ear = baseline_ear

    def update(self, ear: float) -> dict:
        closure_thresh = self._baseline_ear * settings.perclos_closure_ratio
        is_closed = 1.0 if ear < closure_thresh else 0.0
        self._window.append(is_closed)
        self.perclos = float(np.mean(self._window)) if self._window else 0.0

        if self.perclos >= settings.perclos_alert_thresh:
            state = "critical"
        elif self.perclos >= settings.perclos_warn_thresh:
            state = "warning"
        else:
            state = "normal"

        return {"perclos": self.perclos, "state": state}
