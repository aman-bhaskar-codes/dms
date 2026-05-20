"""
Head Pose V2 — solvePnP (pitch/yaw/roll) + sway/jerk detection
Detects: head nodding (microsleep), distraction, sway oscillation
"""
import numpy as np
import cv2
from collections import deque
from config import settings


class HeadPoseDetector:
    def __init__(self):
        self._3d_pts = np.array(settings.FACE_3D_PTS, dtype=np.float64)
        self._lm_ids = settings.FACE_3D_LANDMARK_IDS
        self._pitch_history: deque = deque(maxlen=settings.head_sway_window)
        self._yaw_history: deque = deque(maxlen=settings.head_sway_window)
        self._prev_yaw = 0.0
        self._prev_pitch = 0.0
        self.distracted_counter = 0
        self.sway_score = 0.0
        self._prev_gray = None

    def _estimate_camera_motion(self, frame_gray: np.ndarray) -> float:
        """Estimate global camera motion using sparse optical flow."""
        if self._prev_gray is None:
            self._prev_gray = frame_gray
            return 0.0
        
        corners = cv2.goodFeaturesToTrack(self._prev_gray, 50, 0.01, 10)
        if corners is None:
            return 0.0
        
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, frame_gray, corners, None
        )
        
        if new_pts is not None and status is not None:
            good_old = corners[status.flatten() == 1]
            good_new = new_pts[status.flatten() == 1]
            if len(good_old) > 5:
                motion = float(np.mean(np.linalg.norm(good_new - good_old, axis=1)))
                self._prev_gray = frame_gray
                return motion
        return 0.0

    def update(self, face_detector, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        pts_2d = face_detector.get_pixel_coords_batch(self._lm_ids)
        if pts_2d is None:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                    "state": "unknown", "sway_score": self.sway_score,
                    "jerk": False, "distracted_frames": self.distracted_counter}

        camera_matrix = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, _ = cv2.solvePnP(
            self._3d_pts, pts_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                    "state": "unknown", "sway_score": self.sway_score,
                    "jerk": False, "distracted_frames": self.distracted_counter}

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        proj_mat = np.hstack([rot_mat, np.zeros((3, 1))])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
        pitch = float(euler[0].item())
        yaw   = float(euler[1].item())
        roll  = float(euler[2].item())

        # Jerk detection
        d_pitch = abs(pitch - self._prev_pitch)
        d_yaw   = abs(yaw - self._prev_yaw)
        jerk = (d_pitch > settings.head_jerk_thresh or
                d_yaw > settings.head_jerk_thresh)
        self._prev_pitch = pitch
        self._prev_yaw = yaw

        # Sway (oscillation score)
        self._pitch_history.append(pitch)
        self._yaw_history.append(yaw)
        if len(self._pitch_history) >= 10:
            raw_sway = float(
                np.std(list(self._pitch_history)) + np.std(list(self._yaw_history))
            )
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame
            camera_motion = self._estimate_camera_motion(frame_gray)
            self.sway_score = max(0.0, raw_sway - camera_motion * 2.0)

        # State
        nod_down = pitch > settings.pitch_down_thresh
        nod_up   = pitch < settings.pitch_up_thresh
        looking_away = abs(yaw) > settings.yaw_thresh

        if looking_away or nod_down or nod_up:
            self.distracted_counter += 1
        else:
            self.distracted_counter = max(0, self.distracted_counter - 1)

        if nod_down:
            state = "nod_down"
        elif looking_away:
            state = "distracted"
        elif nod_up:
            state = "nod_up"
        else:
            state = "normal"

        return {
            "pitch": pitch, "yaw": yaw, "roll": roll, "state": state,
            "sway_score": self.sway_score, "jerk": jerk,
            "distracted_frames": self.distracted_counter,
        }
