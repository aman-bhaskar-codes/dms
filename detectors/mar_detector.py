"""MAR Detector V2 — Mouth Aspect Ratio + 3D yawn analysis"""
import numpy as np
from collections import deque
import time
from config import settings


def compute_mar(mouth_pts: np.ndarray) -> float:
    """Mouth Aspect Ratio — vertical opening vs horizontal width."""
    A = np.linalg.norm(mouth_pts[2] - mouth_pts[6])  # top-bottom
    B = np.linalg.norm(mouth_pts[3] - mouth_pts[7])
    C = np.linalg.norm(mouth_pts[0] - mouth_pts[1])  # left-right
    return float((A + B) / (2.0 * C)) if C > 1e-6 else 0.0


class MARDetector:
    def __init__(self, mar_threshold: float = None):
        self.mar_threshold = mar_threshold or settings.mar_threshold
        self.counter = 0
        self.yawn_count = 0
        self.state = "normal"
        self.last_mar = 0.0
        self._yawn_timestamps: deque = deque(maxlen=60)

    def set_threshold(self, threshold: float):
        self.mar_threshold = threshold

    def update(self, face_detector) -> dict:
        mouth_pts = face_detector.get_pixel_coords_batch(settings.MOUTH_IDX)
        if mouth_pts is None:
            return {"mar": self.last_mar, "state": self.state,
                    "yawn_count": self.yawn_count, "yawn_rate": 0.0}

        mar = compute_mar(mouth_pts)
        self.last_mar = mar

        if mar > self.mar_threshold:
            self.counter += 1
        else:
            if self.counter >= settings.mar_consec_frames:
                self.yawn_count += 1
                self._yawn_timestamps.append(time.time())
            self.counter = 0

        self.state = "yawning" if self.counter >= settings.mar_consec_frames else "normal"

        # Yawn rate last 10 min
        cutoff = time.time() - 600
        yawn_rate = sum(1 for t in self._yawn_timestamps if t > cutoff)
        return {
            "mar": mar, "state": self.state, "consecutive_frames": self.counter,
            "yawn_count": self.yawn_count, "yawn_rate": yawn_rate,
        }
