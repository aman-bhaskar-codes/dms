"""
V5 Perception Pipeline — Non-blocking async frame processor.

Architecture:
  - Camera frames submitted via `submit()` (fire-and-forget from main loop)
  - CPU-bound signal extraction runs in ThreadPoolExecutor (never blocks asyncio)
  - All 9 signals extracted from single face mesh pass for efficiency
  - Results put on asyncio.Queue — consumers await `get_signals()`
  - Integrates VectorizedKalmanBank for landmark smoothing (<0.5ms overhead)
  - Publishes processed signals to EventBus topics
"""
from __future__ import annotations

import asyncio
import time
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, TypedDict

import cv2
import mediapipe as mp
import numpy as np
import base64


from perception.kalman_bank import VectorizedKalmanBank
from perception.fatigue_engine_v5 import FatigueEngineV5, FatigueResult
from core.bus import EventBus, EventTopic
from core.config import settings

logger = logging.getLogger("dms.perception.pipeline")


class RawSignals(TypedDict):
    """All signals extracted from a single frame. Fully typed for mypy."""
    # Eye signals
    ear: float
    perclos: float
    blink_velocity: float
    slow_blink_ratio: float
    # Mouth
    mar: float
    # Head pose
    pitch: float
    yaw: float
    roll: float
    head_sway: float
    # Gaze
    gaze_quality: float
    saccade_velocity: float
    # rPPG
    rppg_hr: float
    rppg_hrv: float
    # Micro-expression
    micro_expression_score: float
    # Scene
    scene_luminance: float
    scene_glare: float
    # Meta
    timestamp: float
    frame_id: int
    face_detected: bool
    processed_frame: str



# MediaPipe FaceMesh landmark indices for key regions
# Reference: https://github.com/google/mediapipe/wiki/MediaPipe-Face-Mesh
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]
_LEFT_IRIS = [474, 475, 476, 477]
_RIGHT_IRIS= [469, 470, 471, 472]
_MOUTH_OUTER = [61, 291, 0, 17, 78, 308, 13, 14]
_FOREHEAD_BAND = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                  361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                  176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                  162, 21, 54, 103, 67, 109]


def _ear(pts: np.ndarray) -> float:
    """Eye Aspect Ratio — Soukupová & Čech 2016."""
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return float((A + B) / (2.0 * C + 1e-9))


def _mar(pts: np.ndarray) -> float:
    """Mouth Aspect Ratio — vertical opening normalized by width."""
    vertical = (np.linalg.norm(pts[2] - pts[6]) + np.linalg.norm(pts[3] - pts[7])) / 2.0
    horizontal = np.linalg.norm(pts[0] - pts[1]) + 1e-9
    return float(vertical / horizontal)


def _head_pose_euler(lm_3d: np.ndarray, frame_shape: tuple) -> tuple[float, float, float]:
    """Estimate pitch/yaw/roll from 3D landmarks via solvePnP."""
    h, w = frame_shape[:2]
    # 3D model points for 6 canonical face points
    model_pts = np.array([
        [0.0,    0.0,    0.0   ],  # nose tip
        [0.0,   -330.0, -65.0 ],  # chin
        [-225.0, 170.0, -135.0],  # left eye corner
        [225.0,  170.0, -135.0],  # right eye corner
        [-150.0,-150.0, -125.0],  # left mouth
        [150.0, -150.0, -125.0],  # right mouth
    ], dtype=np.float64)

    # Image points from landmarks (indices: nose=1, chin=152, l-eye=226, r-eye=446, l-mouth=57, r-mouth=287)
    indices = [1, 152, 226, 446, 57, 287]
    img_pts = np.array(
        [[lm_3d[i][0] * w, lm_3d[i][1] * h] for i in indices],
        dtype=np.float64
    )
    focal = w
    cam_mtx = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)

    try:
        _, rvec, _ = cv2.solvePnP(model_pts, img_pts, cam_mtx,
                                   np.zeros(4), flags=cv2.SOLVEPNP_SQPNP)
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        return float(angles[0]), float(angles[1]), float(angles[2])
    except Exception:
        return 0.0, 0.0, 0.0


class _PERCLOSTracker:
    """Tracks percentage eye closure over rolling 60-second window."""
    def __init__(self, fps: int = 30) -> None:
        self._window: deque[bool] = deque(maxlen=fps * 60)
        self._ear_threshold = 0.20

    def update(self, ear: float) -> float:
        self._window.append(ear < self._ear_threshold)
        if len(self._window) < 30:
            return 0.0
        return float(np.mean(self._window))

    def set_threshold(self, t: float) -> None:
        self._ear_threshold = t


class _BlinkTracker:
    """Tracks blink velocity and slow blink ratio."""
    def __init__(self) -> None:
        self._prev_ear = 0.30
        self._velocities: deque[float] = deque(maxlen=30)
        self._slow_flags: deque[int] = deque(maxlen=50)

    def update(self, ear: float) -> tuple[float, float]:
        vel = ear - self._prev_ear
        self._prev_ear = ear
        self._velocities.append(abs(vel))
        # Slow blink: downward velocity slower than 0.02 EAR/frame
        self._slow_flags.append(1 if (vel < 0 and abs(vel) < 0.02) else 0)
        avg_vel = float(np.mean(self._velocities)) if self._velocities else 0.0
        slow_ratio = float(np.mean(self._slow_flags)) if self._slow_flags else 0.0
        return avg_vel, slow_ratio


class _HeadSwayTracker:
    """Detects oscillatory head movement amplitude indicating nodding."""
    def __init__(self) -> None:
        self._pitch_history: deque[float] = deque(maxlen=90)  # 3s @ 30fps

    def update(self, pitch: float) -> float:
        self._pitch_history.append(pitch)
        if len(self._pitch_history) < 30:
            return 0.0
        arr = np.array(self._pitch_history)
        return float(arr.std())


class FrameProcessor:
    """
    V5 Perception Pipeline.

    Usage:
        processor = FrameProcessor(bus)
        # In camera loop:
        await processor.submit(frame)
        # In consumer:
        fatigue_result = await processor.get_fatigue()
    """

    def __init__(self, bus: EventBus, kalman_enabled: bool = True,
                 lstm_model_path: Optional[str] = None) -> None:
        self._bus = bus
        self._kalman_enabled = kalman_enabled
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="perception"
        )

        # MediaPipe FaceLandmarker (Tasks API)
        base_options = mp.tasks.BaseOptions(model_asset_path='face_landmarker.task')
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._mp_face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._kalman = VectorizedKalmanBank(n_landmarks=478)
        self._fatigue_engine = FatigueEngineV5(lstm_model_path=lstm_model_path)

        # Sub-trackers
        self._perclos = _PERCLOSTracker(fps=settings.camera_fps)
        self._blink   = _BlinkTracker()
        self._sway    = _HeadSwayTracker()

        # Queues
        self._signal_queue: asyncio.Queue[RawSignals] = asyncio.Queue(maxsize=4)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Stats
        self._frame_id: int = 0
        self._latency_window: deque[float] = deque(maxlen=60)

        # State
        self._running = False
        logger.info("[FrameProcessor] V5 perception pipeline initialized.")

    async def submit(self, frame: np.ndarray) -> None:
        """Non-blocking: copies frame and submits to executor. Never awaits."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        self._frame_id += 1
        self._loop.run_in_executor(
            self._executor,
            self._process_sync,
            frame.copy(),
            self._frame_id,
        )

    def _process_sync(self, frame: np.ndarray, frame_id: int) -> None:
        """CPU-bound work. Runs in thread. No asyncio calls here."""
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._mp_face_mesh.detect(mp_image)

        if not results.face_landmarks:
            signals = self._empty_signals(frame_id)
            self._put_signals(signals)
            return

        # Extract raw landmark array (478, 3) — normalized [0,1]
        lm_raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.face_landmarks[0]],
            dtype=np.float32
        )  # shape: (478, 3)

        # Kalman smoothing (vectorized — ~0.3ms)
        if self._kalman_enabled:
            lm = self._kalman.update(lm_raw)
        else:
            lm = lm_raw

        # ── EAR ─────────────────────────────────────────────────────────────
        l_eye = lm[_LEFT_EYE]
        r_eye = lm[_RIGHT_EYE]
        ear_l = _ear(l_eye)
        ear_r = _ear(r_eye)
        ear   = (ear_l + ear_r) / 2.0

        perclos         = self._perclos.update(ear)
        blink_vel, slow = self._blink.update(ear)

        # ── MAR ─────────────────────────────────────────────────────────────
        mouth = lm[_MOUTH_OUTER]
        mar   = _mar(mouth)

        # ── Head Pose ────────────────────────────────────────────────────────
        pitch, yaw, roll = _head_pose_euler(lm, (h, w))
        sway = self._sway.update(pitch)

        # ── Gaze quality ─────────────────────────────────────────────────────
        # Approximate from iris position relative to eye center
        l_iris = lm[_LEFT_IRIS].mean(axis=0)
        l_eye_center = lm[_LEFT_EYE].mean(axis=0)
        gaze_dev = float(np.linalg.norm(l_iris[:2] - l_eye_center[:2]))
        gaze_quality = float(np.clip(1.0 - gaze_dev * 10.0, 0.0, 1.0))

        # ── Scene analysis ───────────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luminance = float(np.mean(gray)) / 255.0
        # Glare: fraction of saturated pixels
        glare = float(np.mean(gray > 240))

        # ── rPPG (simplified green-channel estimate) ──────────────────────
        # Extract forehead region using landmark indices
        forehead_pts = lm[_FOREHEAD_BAND]
        roi_x1 = int(np.clip(forehead_pts[:, 0].min() * w, 0, w - 1))
        roi_x2 = int(np.clip(forehead_pts[:, 0].max() * w, 0, w - 1))
        roi_y1 = int(np.clip(forehead_pts[:, 1].min() * h, 0, h - 1))
        roi_y2 = int(np.clip(forehead_pts[:, 1].max() * h, 0, h - 1))
        rppg_hr, rppg_hrv = 70.0, 0.0
        if roi_x2 > roi_x1 + 5 and roi_y2 > roi_y1 + 5:
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            green_mean = float(np.mean(roi[:, :, 1]))
            # Rough HR proxy (actual CHROM algorithm is in detection/rppg_ensemble.py)
            rppg_hr = max(50.0, min(120.0, 60.0 + (green_mean - 128.0) * 0.3))

        # Draw face mesh overlay on a copy of the frame
        hud_frame = frame.copy()
        for lm_pt in lm:
            cx, cy = int(lm_pt[0] * w), int(lm_pt[1] * h)
            cv2.circle(hud_frame, (cx, cy), 1, (0, 120, 0), -1)  # sleek deep green mesh dots
            
        # Draw eyes and mouth contours in a premium color
        for eye_pts, color in [(_LEFT_EYE, (0, 255, 255)), (_RIGHT_EYE, (0, 255, 255))]:
            pts = np.array([[lm[idx][0] * w, lm[idx][1] * h] for idx in eye_pts], dtype=np.int32)
            cv2.polylines(hud_frame, [pts], True, color, 1, cv2.LINE_AA)
            
        for iris_pts, color in [(_LEFT_IRIS, (255, 100, 100)), (_RIGHT_IRIS, (255, 100, 100))]:
            for idx in iris_pts:
                cx, cy = int(lm[idx][0] * w), int(lm[idx][1] * h)
                cv2.circle(hud_frame, (cx, cy), 1, color, -1)
                
        mouth_pts = np.array([[lm[idx][0] * w, lm[idx][1] * h] for idx in _MOUTH_OUTER], dtype=np.int32)
        cv2.polylines(hud_frame, [mouth_pts], True, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Nose head pose projection vector
        nose_pt = lm[1]
        nx, ny = int(nose_pt[0] * w), int(nose_pt[1] * h)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        end_x = int(nx - np.sin(yaw_rad) * 60)
        end_y = int(ny + np.sin(pitch_rad) * 60)
        cv2.line(hud_frame, (nx, ny), (end_x, end_y), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(hud_frame, (nx, ny), 3, (0, 0, 255), -1)
        
        # Telemetry Card at the bottom
        overlay = hud_frame.copy()
        cv2.rectangle(overlay, (0, h - 45), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, hud_frame, 0.4, 0, hud_frame)
        status_text = f"EAR: {ear:.2f} | MAR: {mar:.2f} | PERCLOS: {perclos:.2f} | Pose: P={pitch:.1f} Y={yaw:.1f}"
        cv2.putText(hud_frame, status_text, (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Encode to base64 JPEG
        _, buffer = cv2.imencode('.jpg', hud_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        processed_frame_str = f"data:image/jpeg;base64,{jpg_as_text}"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._latency_window.append(latency_ms)

        signals: RawSignals = {
            "ear":                  float(ear),
            "perclos":              float(perclos),
            "blink_velocity":       float(blink_vel),
            "slow_blink_ratio":     float(slow),
            "mar":                  float(mar),
            "pitch":                float(pitch),
            "yaw":                  float(yaw),
            "roll":                 float(roll),
            "head_sway":            float(sway),
            "gaze_quality":         float(gaze_quality),
            "saccade_velocity":     0.0,   # Extended in future via iris velocity
            "rppg_hr":              float(rppg_hr),
            "rppg_hrv":             float(rppg_hrv),
            "micro_expression_score": 0.0,
            "scene_luminance":      float(luminance),
            "scene_glare":          float(glare),
            "timestamp":            time.time(),
            "frame_id":             frame_id,
            "face_detected":        True,
            "processed_frame":      processed_frame_str,
        }

        self._put_signals(signals)

    def _put_signals(self, signals: RawSignals) -> None:
        """Thread-safe put to asyncio queue."""
        if self._loop:
            try:
                self._loop.call_soon_threadsafe(
                    self._signal_queue.put_nowait, signals
                )
            except asyncio.QueueFull:
                pass  # Drop stale frame — prefer freshness

    def _empty_signals(self, frame_id: int) -> RawSignals:
        # Generate empty placeholder image so the user is never left with a broken video screen!
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "NO FACE DETECTED / SEARCHING...", (120, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        image_data = f"data:image/jpeg;base64,{jpg_as_text}"

        return {
            "ear": 0.30, "perclos": 0.0, "blink_velocity": 0.0,
            "slow_blink_ratio": 0.0, "mar": 0.0,
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "head_sway": 0.0,
            "gaze_quality": 1.0, "saccade_velocity": 0.0,
            "rppg_hr": 70.0, "rppg_hrv": 0.0,
            "micro_expression_score": 0.0,
            "scene_luminance": 0.5, "scene_glare": 0.0,
            "timestamp": time.time(), "frame_id": frame_id,
            "face_detected": False,
            "processed_frame": image_data
        }


    async def run(self) -> None:
        """
        Main async consumer loop.
        Drains signal queue, runs fatigue fusion, publishes to bus.
        """
        self._running = True
        self._loop = asyncio.get_running_loop()
        logger.info("[FrameProcessor] Consumer loop started.")

        while self._running:
            try:
                signals: RawSignals = await asyncio.wait_for(
                    self._signal_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Publish raw signals to bus
            if signals["face_detected"]:
                await self._bus.publish(EventTopic.FACE_DETECTED, signals,
                                        source="perception")
                await self._bus.publish(EventTopic.SIGNAL_EAR,
                                        {"ear": signals["ear"],
                                         "perclos": signals["perclos"],
                                         "slow_blink_ratio": signals["slow_blink_ratio"]},
                                        source="perception")
                await self._bus.publish(EventTopic.SIGNAL_HEAD_POSE,
                                        {"pitch": signals["pitch"],
                                         "yaw":   signals["yaw"],
                                         "roll":  signals["roll"],
                                         "sway":  signals["head_sway"]},
                                        source="perception")
                await self._bus.publish(EventTopic.SIGNAL_RPPG,
                                        {"hr_bpm": signals["rppg_hr"]},
                                        source="perception")
                
                # Publish the processed frame
                await self._bus.publish(EventTopic.FRAME_PROCESSED,
                                        {"image": signals["processed_frame"], "metrics": signals},
                                        source="perception")

                # Run fatigue fusion
                result: FatigueResult = self._fatigue_engine.update(signals)
                await self._bus.publish(EventTopic.FATIGUE_SCORE, result,
                                        source="fatigue_engine_v5")

                if result.score >= 70.0:
                    await self._bus.publish(EventTopic.FATIGUE_CRITICAL, result,
                                            source="fatigue_engine_v5")
            else:
                await self._bus.publish(EventTopic.FACE_LOST, {},
                                        source="perception")
                # Also publish empty placeholder frame
                await self._bus.publish(EventTopic.FRAME_PROCESSED,
                                        {"image": signals["processed_frame"], "metrics": signals},
                                        source="perception")


    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self._latency_window)) if self._latency_window else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if len(self._latency_window) < 5:
            return 0.0
        return float(np.percentile(list(self._latency_window), 95))

    def set_ear_threshold(self, threshold: float) -> None:
        self._perclos.set_threshold(threshold)

    async def shutdown(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._mp_face_mesh.close()
        logger.info("[FrameProcessor] Shutdown complete.")
