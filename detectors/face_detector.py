"""
Face Detector V2.1 — MediaPipe Tasks API (Python 3.12 compatible)
MediaPipe FaceLandmarker + 1D Kalman filter per landmark.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List
from config import settings

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class KalmanFilter1D:
    """Scalar 1D Kalman filter."""
    __slots__ = ('Q', 'R', 'P', 'x', 'initialized')

    def __init__(self):
        self.Q = settings.kalman_process_noise
        self.R = settings.kalman_measurement_noise
        self.P = 1.0
        self.x = 0.0
        self.initialized = False

    def update(self, z: float) -> float:
        if not self.initialized:
            self.x = z
            self.initialized = True
            return z
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1.0 - K)
        return self.x


class FaceDetector:
    def __init__(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=settings.max_num_faces,
            min_face_detection_confidence=settings.min_detection_conf,
            min_face_presence_confidence=settings.min_tracking_conf,
            output_face_blendshapes=False
        )
        self.face_mesh = FaceLandmarker.create_from_options(options)
        self.results = None
        self.landmarks = None
        self.frame_h = settings.frame_height
        self.frame_w = settings.frame_width
        n_lm = 478
        self._kf_x: List[KalmanFilter1D] = [KalmanFilter1D() for _ in range(n_lm)]
        self._kf_y: List[KalmanFilter1D] = [KalmanFilter1D() for _ in range(n_lm)]
        print("[FaceDetector] MediaPipe Tasks + Kalman ready")

    def process(self, frame_bgr: np.ndarray) -> bool:
        h, w = frame_bgr.shape[:2]
        self.frame_h, self.frame_w = h, w
        # MediaPipe needs RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        self.results = self.face_mesh.detect(mp_image)
        
        if self.results and self.results.face_landmarks and len(self.results.face_landmarks) > 0:
            self.landmarks = self.results.face_landmarks[0]
            return True
        self.landmarks = None
        return False

    def _get_filtered(self, idx: int) -> Tuple[float, float]:
        if idx >= len(self.landmarks):
            return 0.0, 0.0
        lm = self.landmarks[idx]
        if settings.kalman_enabled:
            fx = self._kf_x[idx].update(lm.x)
            fy = self._kf_y[idx].update(lm.y)
        else:
            fx, fy = lm.x, lm.y
        return fx, fy

    def get_pixel_coords_batch(self, indices: list) -> Optional[np.ndarray]:
        if not self.landmarks:
            return None
        coords = []
        for idx in indices:
            fx, fy = self._get_filtered(idx)
            coords.append([fx * self.frame_w, fy * self.frame_h])
        return np.array(coords, dtype=np.float64)

    def get_forehead_roi_pts(self) -> Optional[np.ndarray]:
        return self.get_pixel_coords_batch(settings.FOREHEAD_LANDMARKS)
        
    def draw_mesh(self, frame: np.ndarray) -> np.ndarray:
        if self.landmarks:
            # Draw a sparse mesh using key landmarks for speed
            for idx in [33, 263, 1, 152, 61, 291]:  # eyes, nose, chin, mouth corners
                if idx < len(self.landmarks):
                    fx, fy = self._get_filtered(idx)
                    cv2.circle(frame, (int(fx * self.frame_w), int(fy * self.frame_h)), 
                               2, (0, 255, 0), -1)
        return frame

    def release(self):
        self.face_mesh.close()
