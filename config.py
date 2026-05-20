# config.py  — DMS V3 unified configuration

import os
from typing import Optional

# ─── System ──────────────────────────────────────────────────────────────────
VERSION              = "3.0.0"
DEBUG_MODE           = False
LOG_LEVEL            = "INFO"

# ─── Camera ──────────────────────────────────────────────────────────────────
CAMERA_INDEX         = 0
FRAME_WIDTH          = 640
FRAME_HEIGHT         = 480
TARGET_FPS           = 30
FLIP_HORIZONTAL      = True

# ─── MediaPipe ───────────────────────────────────────────────────────────────
MAX_NUM_FACES        = 1
REFINE_LANDMARKS     = True
MIN_DETECTION_CONF   = 0.7
MIN_TRACKING_CONF    = 0.7

# ─── Kalman Filter ───────────────────────────────────────────────────────────
KALMAN_ENABLED            = True
KALMAN_PROCESS_NOISE      = 0.005
KALMAN_MEASUREMENT_NOISE  = 0.05

# ─── Calibration ─────────────────────────────────────────────────────────────
CALIBRATION_DURATION_SEC  = 30
CALIBRATION_PROFILE_DIR   = "data/driver_profiles"
DEFAULT_DRIVER_ID         = "default"
AUTO_CALIBRATE_ON_START   = True
CONTINUOUS_RECALIBRATE    = True   # V3: background drift correction

# ─── EAR V3 ──────────────────────────────────────────────────────────────────
EAR_THRESHOLD        = 0.25
EAR_OPEN_BASELINE    = 0.30
EAR_CONSEC_FRAMES    = 48
EAR_WARN_FRAMES      = 20
BLINK_VELOCITY_SLOW  = -0.012
BLINK_SLOW_THRESH    = 0.35
BLINK_RATE_MIN       = 10
BLINK_RATE_MAX       = 25
MICRO_EXPR_ENABLED   = True          # V3 NEW
MICRO_EXPR_THRESH    = 0.045

LEFT_EYE_INDICES     = [362,385,387,263,373,380]
RIGHT_EYE_INDICES    = [33,160,158,133,153,144]

# ─── MAR V3 ──────────────────────────────────────────────────────────────────
MAR_THRESHOLD        = 0.60
MAR_CONSEC_FRAMES    = 20
YAWN_RATE_ALERT      = 3

# ─── Head Pose V3 ────────────────────────────────────────────────────────────
PITCH_DOWN_THRESH    = 20
PITCH_UP_THRESH      = -15
YAW_THRESH           = 30
ROLL_THRESH          = 25
HEAD_SWAY_WINDOW     = 90
HEAD_JERK_THRESH     = 15.0

FACE_3D_MODEL_POINTS = [
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
]
FACE_3D_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# ─── Gaze V3 ─────────────────────────────────────────────────────────────────
GAZE_THRESH          = 0.35
GAZE_CONSEC_FRAMES   = 60
FIXATION_MIN_FRAMES  = 15
HEATMAP_DECAY        = 0.995
SACCADE_SLOW_THRESH  = 200.0         # V3 NEW: °/s below = drowsy saccade

LEFT_IRIS_INDICES   = [474, 475, 476, 477]
RIGHT_IRIS_INDICES  = [469, 470, 471, 472]
LEFT_EYE_OUTER      = [362, 263]
RIGHT_EYE_OUTER     = [33, 133]

# ─── PERCLOS V3 ──────────────────────────────────────────────────────────────
PERCLOS_WINDOW_SEC   = 60
PERCLOS_ALERT_THRESH = 0.15
PERCLOS_WARN_THRESH  = 0.08
PERCLOS_CLOSURE_RATIO = 0.70
PERCLOS_PREDICTION   = True          # V3 NEW

# ─── rPPG V3 ─────────────────────────────────────────────────────────────────
RPPG_ENABLED         = True
RPPG_WINDOW_SEC      = 10
RPPG_HR_MIN          = 42
RPPG_HR_MAX          = 180
RPPG_HRV_ENABLED     = True          # V3 NEW
RPPG_STRESS_THRESH   = 95
RPPG_LOW_THRESH      = 55

FOREHEAD_LANDMARKS  = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                       361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                       176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                       162, 21, 54, 103, 67, 109]

# ─── Scene Analyzer V3 ───────────────────────────────────────────────────────
SCENE_ANALYZER_ENABLED = True        # V3 NEW
SCENE_SAMPLE_RATE_SEC  = 5.0         # Analyze scene every 5s

# ─── Fatigue Score V3 ────────────────────────────────────────────────────────
FATIGUE_NORMAL       = 25
FATIGUE_MILD         = 45
FATIGUE_WARNING      = 70
FATIGUE_CRITICAL     = 85
FATIGUE_PREDICTION_ENABLED = True    # V3 NEW
FATIGUE_PREDICTION_WINDOW  = 300     # 5 min history for trend

# Signal weights (total must = 1.0)
FATIGUE_WEIGHTS = {
    'ear':       0.22,
    'perclos':   0.18,
    'blink':     0.13,
    'head_sway': 0.09,
    'mar':       0.09,
    'gaze':      0.10,
    'rppg':      0.10,
    'scene':     0.05,
    'micro_expr':0.04,
}

# ─── Object Detection ────────────────────────────────────────────────────────
YOLO_ENABLED         = True
YOLO_MODEL           = "yolov8n.pt"
YOLO_CONF            = 0.45
YOLO_CLASSES         = [67, 41, 39, 73]  # phone, cup, bottle, laptop
YOLO_INTERVAL_FRAMES = 10

# ─── Alert System ────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SEC   = 8.0
ALERT_ESCALATION_STEPS = 3
ALERT_SOUND_ENABLED  = True
TTS_ENABLED          = True
TTS_RATE             = 175
TTS_VOLUME           = 0.9
TTS_ENGINE           = "coqui"       # coqui / pyttsx3
COQUI_MODEL          = "tts_models/en/vctk/vits"
COQUI_SPEAKER        = "p267"

# V2 Color scheme (BGR)
COLOR_NORMAL        = (80, 220, 100)
COLOR_MILD          = (60, 210, 210)
COLOR_WARN          = (30, 150, 255)
COLOR_CRITICAL      = (50, 50, 255)
COLOR_INFO          = (180, 180, 195)
COLOR_ACCENT        = (255, 180, 50)
COLOR_EAR           = (230, 200, 50)
COLOR_IRIS          = (210, 100, 200)
COLOR_HEAD          = (100, 240, 240)
COLOR_HR            = (80, 80, 255)
COLOR_FATIGUE_LOW   = (80, 220, 100)
COLOR_FATIGUE_MID   = (30, 165, 255)
COLOR_FATIGUE_HIGH  = (50, 50, 255)
PANEL_BG            = (18, 18, 24)
PANEL_HEADER        = (28, 28, 38)

SHOW_LANDMARKS      = False
SHOW_MESH           = False
SHOW_IRIS           = True
SHOW_HEAD_AXES      = True
SHOW_HEATMAP        = True
SHOW_EAR_WAVEFORM   = True
SHOW_FATIGUE_GAUGE  = True
HUD_ALPHA           = 0.80
DASHBOARD_WIDTH     = 320

# ─── Memory System ───────────────────────────────────────────────────────────
MEMORY_DB_PATH       = "data/sessions.db"
CHROMA_DB_PATH       = "data/chroma_db"
CHROMA_COLLECTION_PREFIX = "dms_v3"
METRICS_SAMPLE_RATE_SEC  = 1.0       # Log metrics every 1s (not 30fps)
BULK_EMBED_INTERVAL_SEC  = 60.0      # Embed session chunks every 60s

# ─── Agentic System ──────────────────────────────────────────────────────────
AGENT_ENABLED        = True
AGENT_CHECK_INTERVAL = 1.0           # Agent reasoning cycle (seconds)
AGENT_REASONING_LOG  = True          # Show agent chain-of-thought in UI

# ─── Voice Agent ─────────────────────────────────────────────────────────────
VOICE_AGENT_ENABLED  = True
WHISPER_MODEL        = "base.en"     # base.en / small.en / medium.en
WHISPER_DEVICE       = "cpu"         # cpu / cuda
WHISPER_VAD          = True          # Voice activity detection
VOICE_CONTEXT_TURNS  = 5             # Conversation history length
VOICE_MAX_TOKENS     = 150           # Keep responses brief (spoken)

# ─── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_HOST          = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT       = 30
OLLAMA_REPORT_INTERVAL_MIN = 5
OLLAMA_AUTO_DETECT   = True          # Auto-select best available model
# These will be overridden by auto-detection if OLLAMA_AUTO_DETECT=True:
OLLAMA_AGENT_MODEL   = "llama3.2:3b"
OLLAMA_INTENT_MODEL  = "phi4-mini"
OLLAMA_VOICE_MODEL   = "llama3.2:3b"
OLLAMA_REPORT_MODEL  = "llama3.1:8b"
OLLAMA_EMBED_MODEL   = "nomic-embed-text"

# ─── Web Interface ───────────────────────────────────────────────────────────
WEB_ENABLED          = True
WEB_HOST             = "0.0.0.0"
WEB_PORT             = 8080
WEB_RELOAD           = False

# ─── Dashboard ───────────────────────────────────────────────────────────────
DASHBOARD_MODE       = "pyqt"        # pyqt / headless / web-only
DASHBOARD_THEME      = "dark"
ANALYTICS_HISTORY_MIN = 5            # Minutes of history shown in charts
