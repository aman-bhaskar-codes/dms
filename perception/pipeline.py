"""Sequential local vision worker; no rPPG, forecasts, or fabricated signals."""

import base64
import time
import cv2
import numpy as np
from core.capture import LatestProcessor
from core.config import Settings

_LEFT_EYE = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_MOUTH_OUTER = [61, 291, 0, 17, 78, 308, 13, 14]


def fit_frame(frame, width, height):
    """Letterbox native camera modes into the configured inference size without distortion."""
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame
    scale = min(width / w, height / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(
        frame, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    fitted = np.zeros((height, width, 3), dtype=frame.dtype)
    y, x = (height - new_h) // 2, (width - new_w) // 2
    fitted[y : y + new_h, x : x + new_w] = resized
    return fitted


def _ear(pts: np.ndarray) -> float:
    """Eye Aspect Ratio — Soukupová & Čech 2016."""
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return float((A + B) / (2.0 * C + 1e-9))


def _mar(pts: np.ndarray) -> float:
    """Mouth Aspect Ratio — vertical opening normalized by width."""
    vertical = (np.linalg.norm(pts[2] - pts[3]) + np.linalg.norm(pts[6] - pts[7])) / 2.0
    horizontal = np.linalg.norm(pts[0] - pts[1]) + 1e-9
    return float(vertical / horizontal)


def _head_pose_euler(lm_3d: np.ndarray, frame_shape: tuple) -> tuple[float, float, float]:
    """Estimate pitch/yaw/roll from 3D landmarks via solvePnP."""
    h, w = frame_shape[:2]
    # 3D model points for 6 canonical face points
    model_pts = np.array(
        [
            [0.0, 0.0, 0.0],  # nose tip
            [0.0, 330.0, 65.0],  # chin
            [-225.0, -170.0, 135.0],  # left eye corner
            [225.0, -170.0, 135.0],  # right eye corner
            [-150.0, 150.0, 125.0],  # left mouth
            [150.0, 150.0, 125.0],  # right mouth
        ],
        dtype=np.float64,
    )

    # Image points from landmarks (indices: nose=1, chin=152, l-eye=226, r-eye=446, l-mouth=57, r-mouth=287)
    indices = [1, 152, 226, 446, 57, 287]
    img_pts = np.array([[lm_3d[i][0] * w, lm_3d[i][1] * h] for i in indices], dtype=np.float64)
    focal = w
    cam_mtx = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)

    try:
        _, rvec, _ = cv2.solvePnP(
            model_pts, img_pts, cam_mtx, np.zeros(4), flags=cv2.SOLVEPNP_SQPNP
        )
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        return float(angles[0]), float(angles[1]), float(angles[2])
    except Exception:
        return None, None, None


class FrameProcessor:
    def __init__(self, config=None):
        self.config = config or Settings()
        self.detector = None
        self.worker = LatestProcessor(self.process, self.cleanup)

    def process(self, packet):
        frame, frame_id, captured_at = packet
        if time.time() - captured_at > self.config.max_frame_age_s:
            return None
        frame = fit_frame(frame, self.config.camera_width, self.config.camera_height)
        import mediapipe as mp

        if self.detector is None:
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(self.config.model_path),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU,
                ),
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        raw = dict(
            frame_id=frame_id,
            captured_at=captured_at,
            face_detected=False,
            reason="face_not_detected",
        )
        if result.face_landmarks:
            lm = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)
            # Aspect ratios use pixel coordinates: normalize x/y anisotropy correctly.
            pixels = lm.copy()
            pixels[:, 0] *= frame.shape[1]
            pixels[:, 1] *= frame.shape[0]
            pixels[:, 2] *= frame.shape[1]
            ear = (_ear(pixels[_LEFT_EYE]) + _ear(pixels[_RIGHT_EYE])) / 2
            mar = _mar(pixels[_MOUTH_OUTER])
            pitch, yaw, roll = _head_pose_euler(lm, frame.shape[:2])
            raw.update(face_detected=True, ear=ear, mar=mar, pitch=pitch, yaw=yaw)
            for p in pixels:
                cv2.circle(frame, (int(p[0]), int(p[1])), 1, (170, 200, 60), -1)
        success, image = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        preview = base64.b64encode(image).decode("ascii") if success else None
        return raw, preview

    def cleanup(self):
        if self.detector:
            self.detector.close()
            self.detector = None
