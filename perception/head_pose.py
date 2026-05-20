"""
Head Pose Estimator V2 — solvePnP + Sway Detection + Jerk Detection

V2 additions:
  - Head sway: oscillating pitch/yaw over 3s window = drowsy driving
  - Jerk detection: sudden large pose change after stillness = microsleep recovery
  - Composite head fatigue score
  - More stable Euler angle extraction
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, Dict
import time
import config


class HeadPoseEstimator:
    MODEL_3D = np.array(config.FACE_3D_MODEL_POINTS, dtype=np.float64)

    def __init__(self, frame_width: int = config.FRAME_WIDTH,
                 frame_height: int = config.FRAME_HEIGHT):
        self.frame_w = frame_width
        self.frame_h = frame_height

        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Smoothing
        self.pitch_hist = deque(maxlen=7)
        self.yaw_hist   = deque(maxlen=7)
        self.roll_hist  = deque(maxlen=7)

        self.pitch = 0.0
        self.yaw   = 0.0
        self.roll  = 0.0

        # State counters
        self.counter_distract = 0
        self.counter_nod      = 0

        # V2: Sway detection
        self._sway_window_pitch = deque(maxlen=config.HEAD_SWAY_WINDOW)
        self._sway_window_yaw   = deque(maxlen=config.HEAD_SWAY_WINDOW)
        self._sway_score        = 0.0
        self._last_sway_calc    = time.time()

        # V2: Jerk detection
        self._prev_pitch    = 0.0
        self._prev_yaw      = 0.0
        self._still_counter = 0            # frames of near-stillness
        self._jerk_events   = deque(maxlen=20)
        self._jerk_detected = False

        self.rotation_vector    = None
        self.translation_vector = None

        print(f"[HeadPoseEstimator V2] Yaw:±{config.YAW_THRESH}° "
              f"Pitch:+{config.PITCH_DOWN_THRESH}°  Sway+Jerk detection ON")

    def update(self, landmark_pixel_coords: np.ndarray) -> Tuple[float, float, float, str]:
        image_points = landmark_pixel_coords.astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_3D, image_points,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return self.pitch, self.yaw, self.roll, "unknown"

        self.rotation_vector    = rvec
        self.translation_vector = tvec

        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._euler_from_rotation(rmat)

        self.pitch_hist.append(pitch)
        self.yaw_hist.append(yaw)
        self.roll_hist.append(roll)

        self.pitch = float(np.mean(self.pitch_hist))
        self.yaw   = float(np.mean(self.yaw_hist))
        self.roll  = float(np.mean(self.roll_hist))

        # Sway tracking
        self._sway_window_pitch.append(self.pitch)
        self._sway_window_yaw.append(self.yaw)
        self._update_sway()

        # Jerk detection
        self._detect_jerk()
        self._prev_pitch = self.pitch
        self._prev_yaw   = self.yaw

        state = self._classify_state()
        return self.pitch, self.yaw, self.roll, state

    def _euler_from_rotation(self, rmat: np.ndarray) -> Tuple[float, float, float]:
        """Stable Euler angle extraction with gimbal lock handling."""
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        if sy >= 1e-6:
            pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
            yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll  = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll  = 0.0
        return pitch, yaw, roll

    def _update_sway(self):
        """
        Compute head sway score: measures oscillatory motion in pitch/yaw.
        A drowsy driver shows slow sinusoidal head movement.
        """
        if len(self._sway_window_pitch) < config.HEAD_SWAY_WINDOW // 2:
            return
        # Sway = standard deviation of pose over the window
        # (normalized by threshold so score is 0..1)
        p_std = float(np.std(list(self._sway_window_pitch)))
        y_std = float(np.std(list(self._sway_window_yaw)))
        # Normalize: sway_score=1.0 means std equals threshold
        p_score = min(1.0, p_std / (config.PITCH_DOWN_THRESH * 0.5))
        y_score = min(1.0, y_std / (config.YAW_THRESH * 0.5))
        self._sway_score = (p_score + y_score) / 2.0

    def _detect_jerk(self):
        """
        Detect sudden large head displacement (microsleep recovery jerk).
        Pattern: many still frames → sudden large movement
        """
        d_pitch = abs(self.pitch - self._prev_pitch)
        d_yaw   = abs(self.yaw   - self._prev_yaw)
        motion  = max(d_pitch, d_yaw)

        STILL_THRESH = 1.5  # degrees — "still" threshold

        if motion < STILL_THRESH:
            self._still_counter += 1
            self._jerk_detected  = False
        else:
            if (self._still_counter > 20 and motion > config.HEAD_JERK_THRESH):
                # Large movement after being still = possible microsleep wake-up
                self._jerk_detected = True
                self._jerk_events.append(time.time())
            self._still_counter = 0

    def _classify_state(self) -> str:
        if abs(self.yaw) > config.YAW_THRESH:
            self.counter_distract += 1
            self.counter_nod = max(0, self.counter_nod - 1)
        else:
            self.counter_distract = max(0, self.counter_distract - 2)

        if self.pitch > config.PITCH_DOWN_THRESH:
            self.counter_nod += 1
            self.counter_distract = max(0, self.counter_distract - 1)
        else:
            self.counter_nod = max(0, self.counter_nod - 2)

        if self._jerk_detected:
            return "jerk"  # microsleep wake-up
        elif self.counter_distract >= config.HEAD_DISTRACTION_FRAMES:
            return "distracted"
        elif self.counter_nod >= 15:
            return "nod"
        elif abs(self.roll) > config.ROLL_THRESH:
            return "tilt"
        elif self._sway_score > 0.60:
            return "sway"
        return "normal"

    @property
    def sway_score(self) -> float:
        return self._sway_score

    @property
    def jerk_count_recent(self) -> int:
        now = time.time()
        return sum(1 for t in self._jerk_events if now - t <= 300)

    def fatigue_components(self) -> Dict[str, float]:
        return {
            "sway_score":    self._sway_score,
            "nod_fraction":  min(1.0, self.counter_nod / 45.0),
            "jerk_score":    min(1.0, self.jerk_count_recent / 3.0),
            "distract_frac": min(1.0, self.counter_distract / config.HEAD_DISTRACTION_FRAMES),
        }

    def draw_axes(self, frame: np.ndarray, length: float = 80.0) -> np.ndarray:
        if self.rotation_vector is None:
            return frame
        axis = np.float32([[length,0,0],[0,length,0],[0,0,length]])
        pts, _ = cv2.projectPoints(axis, self.rotation_vector,
                                    self.translation_vector,
                                    self.camera_matrix, self.dist_coeffs)
        origin, _ = cv2.projectPoints(np.array([[0.0,0.0,0.0]]),
                                       self.rotation_vector, self.translation_vector,
                                       self.camera_matrix, self.dist_coeffs)
        p0 = tuple(origin[0][0].astype(int))
        for i, color in enumerate([(0,0,255),(0,255,0),(255,0,0)]):
            cv2.line(frame, p0, tuple(pts[i][0].astype(int)), color, 2)
        return frame

    def reset(self):
        for d in [self.pitch_hist, self.yaw_hist, self.roll_hist,
                  self._sway_window_pitch, self._sway_window_yaw]:
            d.clear()
        self.counter_distract = self.counter_nod = 0
