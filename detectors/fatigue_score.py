"""
Fatigue Score Engine V2 — 7-signal weighted composite 0-100
V4 adds: LinearRegression 3-minute prediction + structured output
"""
import numpy as np
from collections import deque
from typing import Dict, Optional
from config import settings


class FatigueScoreEngine:
    def __init__(self):
        self._history: deque = deque(maxlen=settings.fatigue_trend_window)
        self.score = 0.0
        self.trend = 0.0       # rising = positive, falling = negative
        self.prediction_3min: Optional[float] = None
        self.level = "normal"  # normal / mild / warning / critical

    def update(self, ear_data: dict, perclos_data: dict,
               mar_data: dict, head_data: dict,
               gaze_data: dict, rppg_data: dict) -> dict:

        # Normalize each signal 0→1 (higher = worse)
        ear = ear_data.get("ear", 0.3)
        ear_score = np.clip(1.0 - (ear / settings.ear_open_baseline), 0, 1)

        perclos = perclos_data.get("perclos", 0.0)
        perclos_score = np.clip(perclos / settings.perclos_alert_thresh, 0, 1)

        blink_score = ear_data.get("drowsy_blink_score", 0.0)

        sway = head_data.get("sway_score", 0.0)
        sway_score = np.clip(sway / 20.0, 0, 1)

        yawn_rate = mar_data.get("yawn_rate", 0)
        yawn_score = np.clip(yawn_rate / settings.yawn_rate_alert, 0, 1)

        gaze_ctr = gaze_data.get("counter", 0)
        gaze_score = np.clip(gaze_ctr / settings.gaze_consec_frames, 0, 1)

        hr = rppg_data.get("hr_bpm", 0.0)
        hr_valid = rppg_data.get("hr_valid", False)
        if hr_valid and hr > 0:
            rppg_score = max(
                np.clip((hr - settings.rppg_hr_stress_thresh) / 30.0, 0, 1),
                np.clip((settings.rppg_hr_low_thresh - hr) / 15.0, 0, 1),
            )
        else:
            rppg_score = 0.0

        signals = {
            "ear": ear_score,
            "perclos": perclos_score,
            "blink_dynamics": blink_score,
            "head_sway": sway_score,
            "yawn": yawn_score,
            "gaze": gaze_score,
            "rppg": rppg_score,
        }

        # Weighted composite
        score = sum(
            signals[k] * settings.FATIGUE_WEIGHTS[k]
            for k in signals
        ) * 100.0
        self.score = float(np.clip(score, 0, 100))
        self._history.append(self.score)

        # Trend (linear regression over last 300 frames)
        if len(self._history) >= 30:
            arr = np.array(self._history)
            x = np.arange(len(arr))
            self.trend = float(np.polyfit(x, arr, 1)[0])  # slope
        else:
            self.trend = 0.0

        # 3-min prediction
        if len(self._history) >= 60 and self.trend != 0:
            frames_3min = 30 * 60 * 3
            self.prediction_3min = float(np.clip(
                self.score + self.trend * frames_3min, 0, 100
            ))
        else:
            self.prediction_3min = None

        if self.score >= settings.fatigue_critical:
            self.level = "critical"
        elif self.score >= settings.fatigue_warn:
            self.level = "warning"
        elif self.score >= settings.fatigue_mild:
            self.level = "mild"
        else:
            self.level = "normal"

        return {
            "score": self.score, "level": self.level,
            "trend": self.trend,
            "prediction_3min": self.prediction_3min,
            "signals": signals,
        }
