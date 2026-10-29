# Architecture and API Reference

DMS is structured around a local-first, privacy-by-design reactive architecture. Real-time video processing and deterministic safety evaluation execute on-device without cloud dependencies.

---

## 1. System Architecture

```
   [Webcam Device / Replay Fixture]
                  │
                  ▼
   [Latest-Frame Buffer: Capacity 1]
                  │
                  ▼
     [Perception FrameProcessor] ─── (MediaPipe FaceLandmarker, solvePnP)
                  │
                  ▼
     [Result Buffer: Capacity 1]
                  │
                  ▼
      [Core Runtime Ingestion]
                  │
     ┌────────────┼───────────────────────────┐
     ▼            ▼                           ▼
[Monitor]   [SafetyPolicy]             [Persistence Queue: Max 256]
 (PERCLOS)   (State Machine)                          │
     │            │                                   ▼
     │            ▼                           [SQLite WAL Batch Writer]
     │     [AlertEngine]                      (1 Hz Metrics, Events)
     │     (Priority Queue: 8)
     │            │
     │            ▼
     │     [macOS `say` TTS]
     │
     ▼
[EventBus / WebSocket Broadcaster]
     ├── Telemetry WS (5 Hz)
     └── Preview WS (<7 Hz)
```

---

## 2. Core Components and Ownership

- **Runtime Coordinator (`core/runtime.py`):** Owns application lifecycle, component health monitoring, capture tasks, and storage persistence. Acquires an exclusive file lock (`.runtime.lock`) on startup to prevent multi-instance conflicts.
- **Perception Pipeline (`perception/pipeline.py`):** Runs a dedicated thread worker via `LatestProcessor`. Resizes/letterboxes frames into a 640×480 canvas, executes MediaPipe Face Landmarker on CPU (478 3D landmarks), calculates Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and estimates Euler angles (pitch/yaw/roll) using OpenCV SQPNP `solvePnP`.
- **Temporal Monitoring & Calibration (`core/monitoring.py`):** Calculates rolling 60-second time-weighted PERCLOS and consecutive eye-closure duration. Fuses metrics into a 0–100 heuristic fatigue score. Manages guided 20-second frontal-face calibration.
- **Safety Policy (`agents/safety_agent.py`):** Fully deterministic state machine with zero LLM participation. Evaluates Warning (55) and Critical (72) score thresholds, immediate critical alerts on prolonged eye closure ($\ge 1.5\text{s}$), 5-point exit hysteresis, 3-second recovery stabilization, and 8-second reminder cooldowns.
- **Alert Engine (`alerts/alert_engine.py`):** Serialized background audio worker driving native macOS `say`. Features critical alert preemption: urgent safety alerts immediately terminate in-flight speech and purge pending informational announcements.
- **Local Persistence (`memory/store.py`):** SQLite database in WAL mode with foreign-key cascades. Queues database writes through an asynchronous 256-item buffer and commits in batches of up to 32 records.
- **Knowledge & Retrieval (`rag/knowledge.py`):** Ingests local markdown documentation from `docs/knowledge/`. Provides lexical search and optional hybrid dense vector search using local SentenceTransformer embeddings and ChromaDB.
- **AI Assistant Boundary (`ai/assistant.py`):** Enforces strict cloud boundaries. Only curated topic questions or synthetic demo scenarios combined with local approved excerpts are sent to Google Gemini via the official `google-genai` SDK 2 (`store=False`). Enforces local daily quota limits (20 req/day UTC) and falls back to local excerpts during outages.
- **Web Server (`ui/web/server.py`):** FastAPI application bound strictly to loopback (`127.0.0.1`). Enforces Origin and Host verification, Content Security Policy, and request size bounds ($\le 8\text{ KB}$).

---

## 3. Queue Isolation and Bounds

| Queue | Owner | Capacity | Behavior on Overflow |
| --- | --- | --- | --- |
| **Capture Input** | `LatestProcessor` | 1 frame | Drops older un-processed frames to guarantee minimal latency. |
| **Inference Results** | `LatestProcessor` | 1 item | Drops unconsumed inference output if processing stalls. |
| **Audio Priority Queue** | `AlertEngine` | 8 items | Priority 0 for critical alerts, priority 10 for normal; critical alerts purge lower-priority items. |
| **Persistence Queue** | `Runtime` | 256 items | Batches up to 32 items. Throws error and marks monitoring `unavailable` if storage is full. |
| **Assistant Queue** | `AssistantService` | 3 items | Returns "Assistant is busy" if queue capacity is exceeded. |
| **WebSocket Telemetry** | `server.py` | 1 per client | Throttled to 5 Hz poll rate with 1.0s send timeout. |
| **WebSocket Preview** | `server.py` | 1 per client | Throttled to $<7\text{ Hz}$ with 1.0s send timeout; skips duplicate frames. |

---

## 4. REST and WebSocket APIs

All endpoints are hosted locally under `/api/v1` and require loopback Host/Origin headers.

### HTTP Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Current system health, active session, latency percentiles, and errors. |
| `GET` | `/api/v1/setup` | Non-intrusive readiness check of models, storage, audio, and cameras. |
| `POST` | `/api/v1/sessions` | Start a new session (`{"mode": "live"}` or `{"mode": "replay"}`). |
| `POST` | `/api/v1/sessions/stop` | Finalize and close the current monitoring session. |
| `GET` | `/api/v1/sessions` | List recorded sessions for the default local profile. |
| `GET` | `/api/v1/sessions/{id}/report` | Retrieve aggregated metrics, summary stats, and event timeline. |
| `GET` | `/api/v1/sessions/{id}/export?format={json\|csv\|html}` | Export session data in structured JSON, CSV, or printable HTML format. |
| `DELETE` | `/api/v1/sessions/{id}` | Permanently delete a completed session and its cascaded metrics. |
| `POST` | `/api/v1/calibration` | Initiate guided frontal-face calibration in a live session. |
| `POST` | `/api/v1/alerts/acknowledge` | Silence the active audio alert announcement. |
| `POST` | `/api/v1/assistant` | Query assistant with validated `{mode, topic, scenario}` payload. |
| `POST` | `/api/v1/voice/local` | Submit typed question (`{"text": "..."}`) for local intent processing. |
| `POST` | `/api/v1/voice/listen` | Trigger 5-second local microphone capture and Whisper transcription. |
| `POST` | `/api/v1/audio/test` | Queue a local speech test utterance. |
| `GET` | `/api/v1/settings` | Read current configurable runtime settings. |
| `PATCH` | `/api/v1/settings` | Update settings (blocked while a monitoring session is running). |
| `GET` | `/api/v1/knowledge/{id}` | Retrieve approved local knowledge excerpt by unique hash ID. |

### WebSocket Channels

- `/ws/v1/telemetry`: Broadcasts JSON envelopes containing health, snapshot measurements, and active alert state at 5 Hz.
- `/ws/v1/preview`: Broadcasts letterboxed JPEG preview frames with landmark annotations as base64 strings at $<7\text{ Hz}$.

---

## 5. Failure and Degradation Modes

- **Camera Disconnect:** If video capture fails or yields empty frames, the runtime marks the session state as `unavailable` with reason `capture_or_perception_failure`.
- **Stale Video Stream:** If incoming frame processing is delayed beyond `max_frame_age_s` (default 2.0s), the watchdog marks the state as `unavailable` with reason `stale_capture`.
- **Head Pose Tracking Loss:** If facial landmarks are detected but head pose SQPNP decomposition fails, the state transitions to `degraded` rather than fabricating zero angles.
- **Storage Backlog / Disk Full:** If disk writes fail, persistence fails closed, flagging `storage_failure` and stopping live session recording to prevent unrecorded data loss.
- **Cloud Outage / Rate Limit:** If Gemini API fails, times out (15s deadline), or exceeds daily usage quotas (20 req/day), the assistant automatically serves cited excerpts from local documentation.
