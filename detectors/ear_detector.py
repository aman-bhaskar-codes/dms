"""
EAR Detector V2 — Eye Aspect Ratio + Blink Dynamics
Soukupová & Čech (2016) + blink velocity analysis
"""
import numpy as np
from collections import deque
from typing import Dict
import time
from config import settings


def compute_ear(eye_pts: np.ndarray) -> float:
    """EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)"""
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return float((A + B) / (2.0 * C)) if C > 1e-6 else 0.0


class EARDetector:
    def __init__(self, ear_threshold: float = None):
        self.ear_threshold = ear_threshold or settings.ear_threshold
        self.counter = 0
        self.total_blinks = 0
        self.state = "normal"
        self.last_ear = settings.ear_open_baseline
        self._in_blink = False
        self._blink_start_t = 0.0
        self._blink_min_ear = 1.0
        self._last_ear_val = settings.ear_open_baseline
        self._slow_blink_window: deque = deque(maxlen=30)
        self._rate_window_start = time.time()
        self._rate_window_count = 0
        self.blink_rate = 0.0
        self.slow_blink_ratio = 0.0
        self.consecutive_drowsy = 0

    def set_threshold(self, threshold: float):
        self.ear_threshold = threshold
        print(f"[EAR] Threshold updated to {threshold:.3f}")

    def update(self, face_detector, calibration=None) -> Dict:
        left_pts = face_detector.get_pixel_coords_batch(settings.LEFT_EYE_IDX)
        right_pts = face_detector.get_pixel_coords_batch(settings.RIGHT_EYE_IDX)
        if left_pts is None or right_pts is None:
            return {"ear": self.last_ear, "state": self.state,
                    "blink_count": self.total_blinks, "blink_rate": self.blink_rate,
                    "slow_blink_ratio": self.slow_blink_ratio,
                    "consecutive_frames": self.counter, "drowsy_blink_score": 0.0}

        ear_l = compute_ear(left_pts)
        ear_r = compute_ear(right_pts)
        ear = (ear_l + ear_r) / 2.0

        # Update blink rate every 60s
        now = time.time()
        if now - self._rate_window_start >= 60.0:
            self.blink_rate = self._rate_window_count
            self._rate_window_count = 0
            self._rate_window_start = now

        # Blink dynamics
        threshold = calibration.profile.ear_threshold if calibration else self.ear_threshold
        if ear < threshold:
            self.counter += 1
            if not self._in_blink:
                self._in_blink = True
                self._blink_start_t = now
                self._blink_min_ear = ear
            else:
                self._blink_min_ear = min(self._blink_min_ear, ear)
        else:
            if self._in_blink and self.counter >= 2:
                duration_ms = (now - self._blink_start_t) * 1000
                velocity = (self.last_ear - self._blink_min_ear) / max(self.counter, 1)
                is_slow = velocity < abs(settings.blink_velocity_slow) or duration_ms > 400
                self._slow_blink_window.append(is_slow)
                self.slow_blink_ratio = (
                    sum(self._slow_blink_window) / len(self._slow_blink_window)
                    if self._slow_blink_window else 0.0
                )
                self.total_blinks += 1
                self._rate_window_count += 1
            self._in_blink = False
            self._blink_min_ear = 1.0
            if self.counter >= settings.ear_consec_frames:
                self.consecutive_drowsy = self.counter
            self.counter = 0

        # State
        if self.counter >= settings.ear_consec_frames:
            self.state = "critical"
        elif self.counter >= settings.ear_warn_frames:
            self.state = "warning"
        else:
            self.state = "normal"

        self.last_ear = ear
        drowsy_blink_score = min(1.0, self.slow_blink_ratio / settings.blink_slow_ratio_thresh)
        return {
            "ear": ear, "ear_left": ear_l, "ear_right": ear_r,
            "state": self.state, "consecutive_frames": self.counter,
            "blink_count": self.total_blinks, "blink_rate": self.blink_rate,
            "slow_blink_ratio": self.slow_blink_ratio,
            "drowsy_blink_score": drowsy_blink_score,
        }
