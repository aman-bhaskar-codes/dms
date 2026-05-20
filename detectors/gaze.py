"""Gaze Tracker V2 — Iris gaze direction + fixation heatmap"""
import numpy as np
from collections import deque
from config import settings


def gaze_direction(iris_pts: np.ndarray, eye_outer: np.ndarray) -> float:
    """Normalized horizontal gaze offset. 0=center, >0=right, <0=left."""
    iris_cx = np.mean(iris_pts[:, 0])
    eye_left_x  = eye_outer[0, 0]
    eye_right_x = eye_outer[1, 0]
    eye_width = eye_right_x - eye_left_x
    if eye_width < 1e-6:
        return 0.0
    return float((iris_cx - (eye_left_x + eye_width / 2)) / (eye_width / 2))


class GazeTracker:
    def __init__(self):
        self.counter = 0
        self.state = "center"
        self._gaze_history: deque = deque(maxlen=settings.fixation_min_frames)
        self.heatmap = np.zeros(
            (settings.frame_height // 8, settings.frame_width // 8), dtype=np.float32
        )

    def update(self, face_detector) -> dict:
        FALLBACK = {
            "gaze_x": 0.0, "state": "unknown",
            "counter": self.counter, "heatmap": self.heatmap,
            "confidence": 0.0, "glasses_mode": True
        }
        
        l_iris = face_detector.get_pixel_coords_batch(settings.LEFT_IRIS_IDX)
        r_iris = face_detector.get_pixel_coords_batch(settings.RIGHT_IRIS_IDX)
        l_outer = face_detector.get_pixel_coords_batch(settings.LEFT_EYE_OUTER)
        r_outer = face_detector.get_pixel_coords_batch(settings.RIGHT_EYE_OUTER)

        if any(x is None for x in [l_iris, r_iris, l_outer, r_outer]):
            self.counter = max(0, self.counter - 1)
            return FALLBACK

        # Confidence: check if iris points form a reasonable circle (prevent glasses issues)
        l_spread = float(np.std(l_iris[:, 0]) + np.std(l_iris[:, 1]))
        if l_spread < 0.5:
            self.counter = max(0, self.counter - 1)
            return FALLBACK

        gaze_l = gaze_direction(l_iris, l_outer)
        gaze_r = gaze_direction(r_iris, r_outer)
        gaze_x = (gaze_l + gaze_r) / 2.0

        # Heatmap update (decay + mark current fixation)
        self.heatmap *= settings.heatmap_decay
        hx = int(np.clip(
            (gaze_x + 1.0) / 2.0 * self.heatmap.shape[1], 0, self.heatmap.shape[1] - 1
        ))
        hy = self.heatmap.shape[0] // 2
        self.heatmap[max(0, hy-2):hy+3, max(0, hx-2):hx+3] += 1.0

        self._gaze_history.append(gaze_x)

        if abs(gaze_x) > settings.gaze_thresh:
            self.counter += 1
        else:
            self.counter = max(0, self.counter - 2)

        state = "off_road" if self.counter >= settings.gaze_consec_frames else "center"
        self.state = state

        return {
            "gaze_x": gaze_x, "state": state,
            "counter": self.counter, "heatmap": self.heatmap.copy(),
            "confidence": 1.0, "glasses_mode": False
        }
