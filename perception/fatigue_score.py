"""
Fatigue Score Engine — 7-Signal Composite Index
NEW in V2.

Combines all detector signals into a single 0–100 fatigue score.
Uses configurable weights (see config.FATIGUE_WEIGHTS).

Score interpretation:
  0–24:   NORMAL — alert driver
  25–44:  MILD — early fatigue indicators
  45–69:  WARNING — driver should consider a break
  70–100: CRITICAL — dangerous fatigue, immediate action needed

The score uses a weighted sum of normalized fatigue sub-components:
  1. EAR signal       (25%) — eye closure fraction + blink dynamics
  2. PERCLOS          (20%) — percentage of time eyes closed
  3. Blink dynamics   (15%) — slow-blink ratio + blink rate anomaly
  4. Head sway        (10%) — oscillatory head movement
  5. Yawn rate        (10%) — yawn frequency and intensity
  6. Gaze quality     (10%) — fixation quality + attention scatter
  7. rPPG HR          (10%) — low heart rate signal

A 5-minute trend line predicts score trajectory.
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional
import time
import config
from sklearn.linear_model import LinearRegression


class FatigueScoreEngine:
    def __init__(self):
        self.score          = 0.0
        self.score_smooth   = 0.0
        self.level          = "normal"

        # Rolling history for smoothing and trend
        self._score_history  = deque(maxlen=config.FATIGUE_PREDICTION_WINDOW)
        self._score_smooth_h = deque(maxlen=30)   # 1-second smooth

        # Per-component last values (for display)
        self.components: Dict[str, float] = {}

        # Trend tracking
        self._trend_slope    = 0.0    # positive = worsening
        self._last_update    = time.time()
        self.trend_predictor = FatigueTrendPredictor()

        # Hysteresis: score must cross threshold for N frames
        self._critical_frames = 0
        self._warn_frames     = 0

        print(f"[FatigueScore] Initialized. Weights: {config.FATIGUE_WEIGHTS}")
        print(f"  Critical≥{config.FATIGUE_CRITICAL}  "
              f"Warning≥{config.FATIGUE_WARNING}  "
              f"Mild≥{config.FATIGUE_MILD}")

    def update(
        self,
        ear_components:     Dict[str, float],   # from EARDetector
        perclos_components: Dict[str, float],   # from PERCLOSEngine
        head_components:    Dict[str, float],   # from HeadPoseEstimator
        mar_components:     Dict[str, float],   # from MARDetector
        gaze_components:    Dict[str, float],   # from GazeTracker
        rppg_components:    Dict[str, float],   # from RPPGEstimator
    ) -> Tuple[float, str]:
        """
        Compute composite fatigue score.

        Returns:
            (score_0_to_100, level_string)
        """
        w = config.FATIGUE_WEIGHTS

        # ── EAR component (25%) ───────────────────────────────────────────────
        ear_score = (
            ear_components.get("ear_deficit",       0.0) * 0.6 +
            ear_components.get("consec_fraction",   0.0) * 0.4
        )

        # ── PERCLOS component (20%) ───────────────────────────────────────────
        perclos_score = (
            perclos_components.get("perclos_60s",  0.0) * 0.5 +
            perclos_components.get("perclos_10s",  0.0) * 0.3 +
            perclos_components.get("trend_rising", 0.0) * 0.2
        )

        # ── Blink dynamics component (15%) ────────────────────────────────────
        blink_score = (
            ear_components.get("slow_blink_ratio",   0.0) * 0.6 +
            ear_components.get("blink_rate_anomaly", 0.0) * 0.4
        )

        # ── Head sway component (10%) ─────────────────────────────────────────
        head_score = (
            head_components.get("sway_score",    0.0) * 0.4 +
            head_components.get("nod_fraction",  0.0) * 0.4 +
            head_components.get("jerk_score",    0.0) * 0.2
        )

        # ── Yawn component (10%) ──────────────────────────────────────────────
        yawn_score = (
            mar_components.get("yawn_rate_score", 0.0) * 0.7 +
            mar_components.get("mar_elevation",   0.0) * 0.3
        )

        # ── Gaze component (10%) ──────────────────────────────────────────────
        gaze_score = (
            gaze_components.get("away_fraction",  0.0) * 0.5 +
            gaze_components.get("gaze_entropy",   0.0) * 0.3 +
            gaze_components.get("low_fixation",   0.0) * 0.2
        )

        # ── rPPG component (10%) ──────────────────────────────────────────────
        rppg_score = rppg_components.get("hr_score", 0.0)

        # ── Weighted sum ──────────────────────────────────────────────────────
        raw_score = (
            w["ear"]            * ear_score     * 100 +
            w["perclos"]        * perclos_score * 100 +
            w["blink_dynamics"] * blink_score   * 100 +
            w["head_sway"]      * head_score    * 100 +
            w["yawn"]           * yawn_score    * 100 +
            w["gaze"]           * gaze_score    * 100 +
            w["rppg"]           * rppg_score    * 100
        )
        raw_score = float(np.clip(raw_score, 0.0, 100.0))

        # Save components for display
        self.components = {
            "EAR":      round(ear_score * 100, 1),
            "PERCLOS":  round(perclos_score * 100, 1),
            "BLINK":    round(blink_score * 100, 1),
            "HEAD":     round(head_score * 100, 1),
            "YAWN":     round(yawn_score * 100, 1),
            "GAZE":     round(gaze_score * 100, 1),
            "HR":       round(rppg_score * 100, 1),
        }

        # Temporal smoothing (exponential moving average)
        self.score = raw_score
        self._score_smooth_h.append(raw_score)
        self.score_smooth = float(np.mean(self._score_smooth_h))
        self._score_history.append(self.score_smooth)

        # Trend analysis
        self._update_trend()
        if config.FATIGUE_PREDICTION_ENABLED:
            self.trend_predictor.history.append(self.score_smooth)

        # Level classification with hysteresis
        self.level = self._classify_level(self.score_smooth)

        return self.score_smooth, self.level

    def _update_trend(self):
        """Compute linear trend slope over history window."""
        h = list(self._score_history)
        if len(h) < 30:
            return
        x = np.arange(len(h), dtype=np.float64)
        # Normalize x to prevent numerical issues
        coeffs = np.polyfit(x / len(h), h, 1)
        self._trend_slope = float(coeffs[0])   # positive = worsening

    def _classify_level(self, score: float) -> str:
        if score >= config.FATIGUE_CRITICAL:
            self._critical_frames += 1
            self._warn_frames     =  0
            if self._critical_frames >= 5:   # 5-frame hysteresis
                return "critical"
        elif score >= config.FATIGUE_WARNING:
            self._critical_frames  = max(0, self._critical_frames - 1)
            self._warn_frames     += 1
            if self._warn_frames >= 5:
                return "warning"
        elif score >= config.FATIGUE_MILD:
            self._critical_frames = max(0, self._critical_frames - 2)
            self._warn_frames     = max(0, self._warn_frames - 1)
            return "mild"
        else:
            self._critical_frames = max(0, self._critical_frames - 2)
            self._warn_frames     = max(0, self._warn_frames - 2)
            return "normal"
        return self.level   # No change until hysteresis clears

    @property
    def trend_str(self) -> str:
        if self._trend_slope > 0.5:
            return "↑ rising"
        elif self._trend_slope < -0.5:
            return "↓ falling"
        return "→ stable"

    @property
    def predicted_critical_min(self) -> Optional[float]:
        """
        Estimate minutes until CRITICAL level at current trend.
        Returns None if stable or improving.
        """
        if self._trend_slope <= 0.1:
            return None
        gap = config.FATIGUE_CRITICAL - self.score_smooth
        if gap <= 0:
            return 0.0
        # trend_slope is score-units per history window
        frames_per_min = config.TARGET_FPS * 60
        slope_per_min  = self._trend_slope * frames_per_min / config.FATIGUE_PREDICTION_WINDOW
        if slope_per_min <= 0:
            return None
        return gap / slope_per_min

    @property
    def level_color(self) -> tuple:
        return {
            "normal":   config.COLOR_FATIGUE_LOW,
            "mild":     config.COLOR_ACCENT,
            "warning":  config.COLOR_FATIGUE_MID,
            "critical": config.COLOR_FATIGUE_HIGH,
        }.get(self.level, config.COLOR_INFO)

class FatigueTrendPredictor:
    """
    Predicts driver fatigue 5 minutes into the future based on recent trend.
    Allows pre-emptive intervention before critical failure.
    """
    def __init__(self):
        self.history = deque(maxlen=config.FATIGUE_PREDICTION_WINDOW)  # e.g., 300 frames

    def predict_future_score(self) -> float:
        if len(self.history) < 50:
            return self.history[-1] if self.history else 0.0

        y = np.array(self.history)
        X = np.arange(len(y)).reshape(-1, 1)

        model = LinearRegression().fit(X, y)

        # Predict 5 minutes ahead (assuming 1 frame/sec logged to history)
        future_X = np.array([[len(y) + 300]])
        future_score = model.predict(future_X)[0]

        return float(np.clip(future_score, 0, 100))
