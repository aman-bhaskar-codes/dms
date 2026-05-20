"""
EAR Detector V2 — Blink Dynamics + Adaptive Threshold

V2 additions:
  1. Blink velocity analysis — slow closures indicate drowsy blinks
  2. Slow-blink ratio tracking — >35% slow blinks = fatigue signal
  3. Blink rate anomaly — too few (<10/min) or too many (>25/min) both signal issues
  4. Adaptive threshold override from calibration profile
  5. Microsleep probability from blink duration histogram

Normal blink: closes in ~150ms (4-5 frames), fast velocity
Drowsy blink:  closes in ~400ms (12+ frames), slow velocity, partial closure
Microsleep:    >500ms closure = not a blink at all
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict, Optional
import time
import config


def compute_ear(eye_points: np.ndarray) -> float:
    """Eye Aspect Ratio — Soukupová & Čech (2016)."""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    if C < 1e-6:
        return 0.0
    return float((A + B) / (2.0 * C))


class BlinkEvent:
    """Represents a single detected blink."""
    __slots__ = ('start_time', 'end_time', 'min_ear', 'close_velocity',
                 'open_velocity', 'duration_ms', 'is_slow')

    def __init__(self, start_time: float, min_ear: float,
                 close_velocity: float, open_velocity: float, duration_ms: float):
        self.start_time    = start_time
        self.min_ear       = min_ear
        self.close_velocity = close_velocity   # EAR/frame (negative = closing)
        self.open_velocity  = open_velocity    # EAR/frame (positive = opening)
        self.duration_ms   = duration_ms
        # Slow blink: closure slower than threshold, OR duration > 400ms
        self.is_slow = (close_velocity > config.BLINK_VELOCITY_SLOW
                        or duration_ms > 400)


class EARDetector:
    """
    V2 EAR-based drowsiness detector with blink dynamics analysis.
    """

    def __init__(self, ear_threshold: Optional[float] = None):
        # Allow calibration to override threshold
        self.ear_threshold = ear_threshold or config.EAR_THRESHOLD

        self.counter        = 0
        self.total_blinks   = 0
        self.drowsy_events  = 0
        self.state          = "normal"

        # Rolling EAR history (Kalman already applied upstream)
        self.ear_history    = deque(maxlen=5)
        self.last_ear       = config.EAR_OPEN_BASELINE
        self.smoothed_ear   = config.EAR_OPEN_BASELINE

        # Blink dynamics
        self._in_blink       = False
        self._blink_start_t  = 0.0
        self._blink_start_ear = 0.0
        self._blink_min_ear  = 1.0
        self._pre_blink_ear  = config.EAR_OPEN_BASELINE
        self._last_ear_val   = config.EAR_OPEN_BASELINE

        # Blink event log (last 5 minutes)
        self.blink_events: deque = deque(maxlen=300)
        self.recent_blink_times: deque = deque(maxlen=60)  # timestamps

        # Slow-blink ratio window
        self._slow_blink_window: deque = deque(maxlen=30)  # last 30 blinks

        # Blink rate tracking
        self._rate_window_start = time.time()
        self._rate_window_count = 0
        
        # Micro-expression V3
        self._micro_detector = MicroExpressionDetector()

        print(f"[EARDetector V2] threshold={self.ear_threshold:.3f}, "
              f"alert={config.EAR_CONSEC_FRAMES/30:.1f}s")

    def set_threshold(self, threshold: float):
        """Override threshold from calibration."""
        self.ear_threshold = threshold
        print(f"[EARDetector V2] Threshold updated to {threshold:.3f} (calibrated)")

    def update(
        self,
        left_eye_pts:  np.ndarray,
        right_eye_pts: np.ndarray,
    ) -> Tuple[float, str, bool]:
        """
        Returns (smoothed_ear, state, is_alert).
        State: "normal" | "warning" | "alert" | "critical"
        """
        ear_l = compute_ear(left_eye_pts)
        ear_r = compute_ear(right_eye_pts)
        ear   = (ear_l + ear_r) / 2.0
        
        # Track micro-expressions
        if config.MICRO_EXPR_ENABLED:
            combined_pts = np.vstack((left_eye_pts, right_eye_pts))
            self._micro_detector.add_landmarks(combined_pts)

        # 3-frame average on top of Kalman (belt-and-suspenders)
        self.ear_history.append(ear)
        smoothed = float(np.mean(self.ear_history))
        self.smoothed_ear = smoothed

        # Compute instantaneous velocity
        velocity = smoothed - self._last_ear_val   # positive = opening
        self._last_ear_val = smoothed

        is_alert = False

        if smoothed < self.ear_threshold:
            self.counter += 1

            # Start of blink
            if not self._in_blink:
                self._in_blink       = True
                self._blink_start_t  = time.time()
                self._blink_start_ear= smoothed
                self._pre_blink_ear  = self.smoothed_ear
                self._blink_min_ear  = smoothed
                self._close_vel      = velocity
            else:
                self._blink_min_ear = min(self._blink_min_ear, smoothed)

            if self.counter >= config.EAR_CONSEC_FRAMES:
                self.state = "critical"
                is_alert   = True
                if self.counter == config.EAR_CONSEC_FRAMES:
                    self.drowsy_events += 1
            elif self.counter >= config.EAR_WARN_FRAMES:
                self.state = "warning"
            else:
                self.state = "alert"
        else:
            # Eyes opened — complete any in-progress blink
            if self._in_blink and self.counter < config.EAR_CONSEC_FRAMES:
                self._finish_blink(smoothed, velocity)

            self._in_blink  = False
            self.counter    = max(0, self.counter - 2)
            if self.counter == 0:
                self.state = "normal"

        self.last_ear = smoothed
        return smoothed, self.state, is_alert

    def _finish_blink(self, open_ear: float, open_velocity: float):
        """Record a completed blink event and analyze its dynamics."""
        now          = time.time()
        duration_ms  = (now - self._blink_start_t) * 1000.0
        close_vel    = getattr(self, '_close_vel', -0.05)

        if duration_ms < 50:  # Noise — ignore sub-50ms "blinks"
            return

        blink = BlinkEvent(
            start_time     = self._blink_start_t,
            min_ear        = self._blink_min_ear,
            close_velocity = close_vel,
            open_velocity  = open_velocity,
            duration_ms    = duration_ms,
        )

        self.blink_events.append(blink)
        self._slow_blink_window.append(1 if blink.is_slow else 0)
        self.total_blinks += 1
        self.recent_blink_times.append(now)
        self._rate_window_count += 1

    @property
    def slow_blink_ratio(self) -> float:
        """Fraction of recent blinks that are slow (0.0–1.0)."""
        if len(self._slow_blink_window) < 5:
            return 0.0
        return float(np.mean(self._slow_blink_window))

    @property
    def blink_rate_per_min(self) -> float:
        """Current blink rate in blinks/minute (last 60 seconds)."""
        now = time.time()
        recent = [t for t in self.recent_blink_times if now - t <= 60.0]
        return len(recent) * (60.0 / max(1.0, min(60.0, now - (recent[0] if recent else now))))

    @property
    def blink_rate_anomaly(self) -> float:
        """
        0.0 = normal rate, 1.0 = severely abnormal (too fast or too slow).
        """
        rate = self.blink_rate_per_min
        if config.BLINK_RATE_MIN <= rate <= config.BLINK_RATE_MAX:
            return 0.0
        elif rate < config.BLINK_RATE_MIN:
            return min(1.0, (config.BLINK_RATE_MIN - rate) / config.BLINK_RATE_MIN)
        else:
            return min(1.0, (rate - config.BLINK_RATE_MAX) / config.BLINK_RATE_MAX)

    def fatigue_components(self) -> Dict[str, float]:
        """Return blink-dynamics fatigue sub-scores for FatigueScore engine."""
        micro_score = 0.0
        if config.MICRO_EXPR_ENABLED:
            micro_res = self._micro_detector.detect()
            if micro_res['micro_expr_detected']:
                micro_score = micro_res['burst_score']

        return {
            "slow_blink_ratio":  self.slow_blink_ratio,
            "blink_rate_anomaly": self.blink_rate_anomaly,
            "ear_deficit":       max(0.0, config.EAR_OPEN_BASELINE - self.smoothed_ear)
                                 / config.EAR_OPEN_BASELINE,
            "consec_fraction":   min(1.0, self.counter / config.EAR_CONSEC_FRAMES),
            "micro_expr_score":  micro_score,
        }

    def reset(self):
        self.counter    = 0
        self.state      = "normal"
        self._in_blink  = False
        self.ear_history.clear()
        self._micro_detector = MicroExpressionDetector()

class MicroExpressionDetector:
    """
    Detects rapid involuntary facial movements indicating:
    - Sudden alertness burst (post-microsleep)
    - Stress response
    - Confusion/disorientation
    Uses landmark velocity burst detection — no neural net needed.
    """
    def __init__(self):
        self.landmark_history = deque(maxlen=5)

    def add_landmarks(self, landmarks: np.ndarray):
        self.landmark_history.append(landmarks)

    def detect(self) -> dict:
        if len(self.landmark_history) < 2:
            return {'micro_expr_detected': False, 'burst_score': 0.0, 'expression_type': 'none'}
            
        velocities = self._compute_landmark_velocities(self.landmark_history)
        if not velocities:
            return {'micro_expr_detected': False, 'burst_score': 0.0, 'expression_type': 'none'}
            
        burst_score = float(max(velocities))
        return {
            'micro_expr_detected': burst_score > config.MICRO_EXPR_THRESH,
            'burst_score':         burst_score,
            'expression_type':     self._classify(burst_score),
        }

    def _compute_landmark_velocities(self, history: deque) -> list:
        vels = []
        for i in range(1, len(history)):
            diff = np.linalg.norm(history[i] - history[i-1], axis=1)
            mean_vel = float(np.mean(diff))
            vels.append(mean_vel)
        return vels

    def _classify(self, burst_score: float) -> str:
        if burst_score > config.MICRO_EXPR_THRESH * 2:
            return 'alert'
        elif burst_score > config.MICRO_EXPR_THRESH:
            return 'stress'
        return 'none'
