"""
DMS V4 Configuration
All thresholds, paths, and settings in one place.
Edit here — never touch the detector code to tune.
"""
from __future__ import annotations
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── API Keys ─────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Feature Flags ────────────────────────────────────────────────
    ollama_enabled: bool = True
    telegram_enabled: bool = False
    yolo_enabled: bool = True
    rppg_enabled: bool = True
    voice_input_enabled: bool = False

    # ── Camera ───────────────────────────────────────────────────────
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    flip_horizontal: bool = True

    # ── MediaPipe ────────────────────────────────────────────────────
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_conf: float = 0.7
    min_tracking_conf: float = 0.7

    # ── Kalman Filter ────────────────────────────────────────────────
    kalman_enabled: bool = True
    kalman_process_noise: float = 0.005
    kalman_measurement_noise: float = 0.05

    # ── EAR (Eye Aspect Ratio) ───────────────────────────────────────
    ear_threshold: float = 0.25
    ear_consec_frames: int = 48      # 1.6s at 30fps
    ear_warn_frames: int = 20        # 0.67s warning
    ear_open_baseline: float = 0.30
    blink_velocity_slow: float = -0.012
    blink_slow_ratio_thresh: float = 0.35
    blink_rate_normal_min: int = 10
    blink_rate_normal_max: int = 25

    # ── MAR (Mouth Aspect Ratio) ─────────────────────────────────────
    mar_threshold: float = 0.60
    mar_consec_frames: int = 20
    yawn_rate_alert: int = 3

    # ── Head Pose ────────────────────────────────────────────────────
    pitch_down_thresh: float = 20.0
    pitch_up_thresh: float = -15.0
    yaw_thresh: float = 30.0
    roll_thresh: float = 25.0
    head_distraction_frames: int = 45
    head_sway_window: int = 90
    head_jerk_thresh: float = 15.0

    # ── Gaze ─────────────────────────────────────────────────────────
    gaze_thresh: float = 0.35
    gaze_consec_frames: int = 60
    fixation_min_frames: int = 15
    heatmap_decay: float = 0.995

    # ── PERCLOS ──────────────────────────────────────────────────────
    perclos_window_sec: int = 60
    perclos_alert_thresh: float = 0.15
    perclos_warn_thresh: float = 0.08
    perclos_closure_ratio: float = 0.70

    # ── rPPG ─────────────────────────────────────────────────────────
    rppg_window_sec: int = 10
    rppg_hr_min: int = 42
    rppg_hr_max: int = 180
    rppg_hr_stress_thresh: int = 95
    rppg_hr_low_thresh: int = 55

    # ── Fatigue Score ────────────────────────────────────────────────
    fatigue_critical: int = 70
    fatigue_warn: int = 45
    fatigue_mild: int = 25
    fatigue_trend_window: int = 300

    # ── YOLO ─────────────────────────────────────────────────────────
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.50

    # ── Alerts ───────────────────────────────────────────────────────
    alert_cooldown_sec: float = 3.0
    alert_escalation_count: int = 3
    tts_rate: int = 160
    tts_volume: float = 0.95

    # ── Calibration ──────────────────────────────────────────────────
    calibration_duration_sec: int = 30
    auto_calibrate_on_start: bool = True

    # ── Ollama AI ────────────────────────────────────────────────────
    ollama_model_fast: str = "llama3.2:3b"     # real-time tips
    ollama_model_smart: str = "llama3.2:3b"    # session reports
    ollama_report_interval_min: int = 5
    ollama_max_tokens_tip: int = 150
    ollama_max_tokens_report: int = 800

    # ── Dashboard ────────────────────────────────────────────────────
    dashboard_port: int = 8080
    dashboard_host: str = "0.0.0.0"

    # ── Data paths ───────────────────────────────────────────────────
    db_path: str = "data/dms.db"
    profile_dir: str = "data/driver_profiles"
    report_dir: str = "data/reports"
    log_interval_frames: int = 30
    break_suggest_min: int = 90

    # ── Vector & Semantic Memory (RAG) ───────────────────────────────
    chroma_db_path: str = "data/chroma"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    chroma_collection_prefix: str = "dms_v4"
    memory_db_path: str = "data/dms_episodic.db"

    # ── Landmark indices (do not change) ────────────────────────────
    LEFT_EYE_IDX:   list = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_IDX:  list = [33,  160, 158, 133, 153, 144]
    MOUTH_IDX:      list = [61,  291,  0,   17,  39, 181, 269, 405]
    LEFT_IRIS_IDX:  list = [474, 475, 476, 477]
    RIGHT_IRIS_IDX: list = [469, 470, 471, 472]
    LEFT_EYE_OUTER: list = [362, 263]
    RIGHT_EYE_OUTER: list = [33, 133]
    FACE_3D_PTS: list = [
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
    ]
    FACE_3D_LANDMARK_IDS: list = [1, 152, 263, 33, 287, 57]
    FOREHEAD_LANDMARKS: list = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    ]
    FATIGUE_WEIGHTS: dict = {
        "ear": 0.25, "perclos": 0.20, "blink_dynamics": 0.15,
        "head_sway": 0.10, "yawn": 0.10, "gaze": 0.10, "rppg": 0.10,
    }
    TTS_MESSAGES: dict = {
        "drowsy_warn":  "Warning. Drowsiness detected. Please stay alert.",
        "drowsy_crit":  "Critical alert. You appear to be falling asleep. Pull over safely.",
        "yawn":         "Yawn detected. Consider taking a break.",
        "head_nod":     "Head nodding detected. Possible microsleep. Pull over now.",
        "distracted":   "Eyes off the road. Please focus ahead.",
        "phone":        "Phone use detected. Put the phone down.",
        "perclos_crit": "Severe drowsiness. Stop driving immediately.",
        "fatigue_high": "Your fatigue score is critical. You should rest.",
        "hr_stress":    "Elevated heart rate detected. Are you feeling okay?",
        "break_suggest":"You've been driving a while. A short break is recommended.",
    }
    YOLO_CLASSES: dict = {67: "phone", 41: "cup", 73: "book"}
    DEBUG_MODE: bool = False
    QUIT_KEY: int = ord('q')


# Singleton
settings = Settings()

# Create directories
for d in [settings.db_path.replace("dms.db", ""),
          settings.profile_dir, settings.report_dir]:
    Path(d).mkdir(parents=True, exist_ok=True)
