# Setup and Configuration Guide

## Prerequisites

- **Operating System:** macOS on Apple Silicon (M1/M2/M3/M4) recommended for hardware camera, microphone, and speech synthesis.
- **Python:** Version 3.12.
- **Package Manager:** [uv](https://docs.astral.sh/uv/) (version 0.4+).
- **Hardware (Optional for live demo):** Built-in FaceTime HD camera or external USB webcam; built-in microphone for local voice push-to-talk.

---

## Clean Installation

From a clean shell, clone the repository and synchronize the locked dependencies:

```sh
git clone https://github.com/aman-bhaskar-codes/dms.git
cd dms
uv sync --locked --extra gemini
```

Verify your local installation:

```sh
uv run dms check
```

Restore and verify the bundled MediaPipe Face Landmarker model asset:

```sh
uv run dms setup-models
```

Start the local server:

```sh
uv run dms run
```

Open **http://127.0.0.1:8000** in your browser. Choose **Start replay** for an automated synthetic scenario or **Start webcam** to monitor live video while stationary.

---

## Optional Components

DMS operates completely offline by default. Two optional feature tiers can be installed on demand:

### 1. Hybrid Knowledge Retrieval
Installs local sentence-transformers embeddings (`all-MiniLM-L6-v2`), ChromaDB vector store, and BM25 rank fusion for searching safety documentation:

```sh
uv sync --locked --extra gemini --extra knowledge
uv run dms setup-models --knowledge
```

### 2. Local Voice Push-to-Talk
Installs `faster-whisper` and downloads the quantized local Whisper model (`whisper-base-en` with VAD filter) for 5-second on-device voice queries:

```sh
uv sync --locked --extra gemini --extra knowledge --extra voice
uv run dms setup-models --knowledge --voice
```

---

## Environment Configuration

Copy the example environment configuration to `.env`:

### POSIX Shell (macOS / Linux)
```sh
cp .env.example .env
```

### PowerShell (Windows / Cross-platform)
```powershell
Copy-Item .env.example .env
```

### Configuration Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | `""` (empty) | Google Gemini API key for curated assistant queries. Must never be placed in client-side code. |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Supported model tier in AI Studio. |
| `GEMINI_DAILY_CAP` | `20` | Maximum daily requests enforced by local SQLite persistence (max 20). |
| `GEMINI_INTERVAL_S` | `10` | Minimum seconds between cloud requests. |
| `CAMERA_INDEX` | `0` | Default video capture device index. |
| `CAMERA_WIDTH` | `640` | Capture resolution width (resized/letterboxed without distortion). |
| `CAMERA_HEIGHT` | `480` | Capture resolution height. |
| `CAMERA_FPS` | `30` | Target frame capture rate. |
| `AUDIO_ENABLED` | `true` | Enable macOS `say` speech output for alert warnings. |
| `FATIGUE_WARN_THRESHOLD` | `55` | Score threshold for warning fatigue level (1–95). |
| `FATIGUE_CRIT_THRESHOLD` | `72` | Score threshold for critical fatigue level (must be $\ge \text{warn} + 5$). |
| `EYE_CLOSURE_S` | `1.5` | Consecutive seconds of closed eyes before triggering immediate critical alert. |
| `RETENTION_DAYS` | `30` | Auto-cleanup threshold for completed historical sessions. |

### Inherited Storage Location
Local SQLite data, user preferences, and calibration are stored in `data/sentinel/sentinel.db`. This path is intentionally preserved across product updates to maintain historical sessions, daily quota reservations, and driver calibrations without interruption.

---

## macOS Permissions

When launching `uv run dms run`:
1. **Camera:** When clicking **Start webcam**, macOS will prompt to allow camera access for the launching terminal application (e.g. Terminal, iTerm2, or IDE). If denied, navigate to:
   **System Settings → Privacy & Security → Camera** and toggle access on.
2. **Microphone:** When clicking **Record locally** in Assistant, grant microphone permission in:
   **System Settings → Privacy & Security → Microphone**.

---

## Platform Support and Limitations

| Platform | Documented Status | Notes |
| --- | --- | --- |
| **Apple Silicon macOS** | **Tier 1 (Supported)** | Full local hardware support: webcam capture, microphone input, native `say` speech synthesis, and local neural network inference. |
| **Linux** | **Headless / CI Only** | Supported for Python runtime, unit tests, and synthetic replays. Physical webcam, desktop audio, and speech synthesis are unverified; macOS `say` is unavailable. |
| **Windows** | **Not Supported** | Native startup is currently blocked by Unix-specific file locking (`fcntl.flock`). Running in WSL2 or Linux containers requires external X11/PulseAudio passthrough. |

---

## Troubleshooting

- **Camera fails to open or produces black frames:** Ensure no other application (FaceTime, Zoom, Photo Booth) is using the camera. Test another index by updating `camera_index: 1` in Settings or `.env`.
- **MediaPipe crashes or segmentation fault:** Ensure `uv run dms setup-models` has verified the asset checksum. On macOS, run outside restricted sandbox environments.
- **Speech synthesis produces no sound:** Verify system volume and confirm `audio_enabled: true` in Settings. Click **Test audio** under Settings to dispatch a test utterance.
- **Gemini returns quota or availability errors:** The application automatically falls back to local approved safety guidance excerpts. Cloud calls are strictly optional and rate-limited.
