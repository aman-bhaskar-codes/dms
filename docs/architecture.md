# Architecture and API

## Runtime

`core/runtime.py` owns the local Store, Monitor, SafetyPolicy, audio worker, assistant, capture lifecycle, and component health. Blocking camera calls use a dedicated thread. MediaPipe and temporal processing use a single worker with a capacity-one latest-frame buffer and capacity-one result buffer. Stale and out-of-order measurements are rejected.

MonitoringSnapshot is a validated media-free contract. The runtime evaluates safety and queues local audio **before** publishing optional updates. Subscribers have independent bounded queues. HTTP/WebSocket consumers poll the latest state at 5 Hz; preview is independently throttled below 7 Hz. Slow socket sends time out. Persistence has a bounded 256-item queue, commits batches of up to 32 records, and fails monitoring closed on storage failure rather than silently claiming successful recording.

A watchdog makes stalled monitoring unavailable. Shutdown stops capture, finishes pending worker cleanup, flushes records, ends the session, closes assistant/audio, cancels background workers, and closes SQLite. A camera driver that never returns from a native read may still delay cooperative shutdown; physical-device fault testing is required before deployment.

## APIs

All routes are local, under `/api/v1`. No cloud authentication or fleet membership is implemented. One default local profile owns sessions. Host and Origin checks protect browser access; arbitrary web origins cannot mutate the app or open its WebSockets.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Current state, component status, latency and queue metrics |
| GET | `/setup` | Passive readiness; does not activate camera or call Gemini |
| POST | `/sessions` | `{ "mode": "live" }` or `{ "mode": "replay" }` |
| POST | `/sessions/stop` | Stop and finalize current session |
| GET | `/sessions` | Default-profile session history |
| GET | `/sessions/{id}/report` | Recorded summary, samples, event timeline |
| GET | `/sessions/{id}/export?format=json` | JSON, CSV, or HTML export |
| DELETE | `/sessions/{id}` | Delete a completed session and its derived records |
| POST | `/calibration` | Start calibration in an active live session |
| POST | `/alerts/acknowledge` | Silence current announcement |
| POST | `/assistant` | Validated mode/topic/scenario only; no free-form cloud text |
| POST | `/voice/local` | `{ "text": "status" }`; local processing only |
| POST | `/voice/listen` | Five-second local microphone capture and transcription |
| POST | `/audio/test` | Queue local audio test |
| GET/PATCH | `/settings` | Allowlisted settings; active session blocks edits |
| GET | `/knowledge/{id}` | Supporting excerpt, provenance and checksum |

`/ws/v1/telemetry` sends the latest health/snapshot/alert envelope. `/ws/v1/preview` sends `{ "image": "base64 JPEG" }`. `/ws/telemetry` is a transitional output adapter using `{topic,payload}`; legacy UI command messages are not supported. WebSocket clients must supply a loopback Origin matching the configured port.

## Storage and privacy

SQLite schema version 1 uses prefixed tables to avoid collisions with historical DMS schemas. An existing pre-v1 database is backed up with SQLite's backup API, including committed WAL contents. Newer schema versions are refused. An exclusive runtime lock prevents a second instance from interrupting a live session. Startups mark unclosed prior sessions interrupted. Explicit legacy imports do not perform startup recovery and copy only sessions and allowlisted event metadata; raw notes, media, and transcripts remain in the untouched source database. Foreign-key cascades remove associated metrics and events. Raw media never enters the Store API.

Cloud prompts are constructed from a closed set of topic/scenario constants and approved knowledge excerpts. The endpoint forbids unknown fields. Local driver history is not indexed into the general knowledge base. Logging does not include frames, transcripts, secrets or provider exception contents. Free-tier limitations are not bypassed by automatic paid fallback.

## Scope of confidence

Tracking availability is not statistical model accuracy. Scores are fixed heuristic indicators. The local RAG engine checks retrieval overlap and citation identity; it does not prove the truth of arbitrary generated claims. Gemini outages change assistant availability only. Raw camera data remains in process memory for preview/inference and is not persisted.
