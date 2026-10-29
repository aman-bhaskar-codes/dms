# Local API reference

[Architecture](architecture.md) · [Configuration](configuration.md) · [Privacy](privacy.md)

Default base URL: `http://127.0.0.1:8000`. FastAPI exposes interactive documentation at `/docs` and the generated schema at `/openapi.json` while running. The examples below use the default port and a server that you started intentionally.

## Local boundary

There is no token-based login. Allowed Host names are loopback names plus the test host. State-mutating HTTP requests reject a present Origin outside the configured loopback origin set; requests without Origin, such as local command-line clients, are allowed. Mutating bodies are bounded to 8192 bytes, including streamed bodies. WebSockets require an allowed Origin explicitly. These controls do not authenticate other local processes.

Standard validation failures generally return HTTP 422; missing records return 404; lifecycle conflicts return 409. Origin rejection returns 403 and body overflow returns 413. A provider outage normally appears in an assistant response's `limitation`, not as an HTTP failure. Do not assume every error body has the same shape.

## Settings and readiness

| Method | Route                | Result / important behavior                                                                                                                        |
| ------ | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| GET    | `/api/v1/health`     | Latest `snapshot`, `alert`, active `session_id`, component states, errors, and metrics. Idle monitoring is `idle`; snapshot and alert can be null. |
| GET    | `/api/v1/setup`      | Readiness check without camera/microphone probing or cloud authentication.                                                                         |
| GET    | `/api/v1/settings`   | Editable camera, audio, score/closure thresholds, and retention fields only.                                                                       |
| PATCH  | `/api/v1/settings`   | Partial object of those fields; rejects unknown fields, invalid threshold order, or any active session. Saves locally.                             |
| POST   | `/api/v1/audio/test` | Queues speech and returns `{"queued":true}`; 409 if audio disabled. Check health and audible output separately.                                    |

Health metrics include processed frame count, P95 capture-to-state and policy-to-audio-queue dispatch timing, recorder backlog, and event-bus counters. Dispatch timing does not measure physical speaker onset. The health payload contains an inherited metadata string; use route/schema contracts rather than treating it as a release compatibility guarantee.

## Sessions, alerts, and calibration

| Method | Route                                       | Request / result                                                                                             |
| ------ | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| POST   | `/api/v1/sessions`                          | `{"mode":"replay"}` or `{"mode":"live"}`; defaults to replay; returns `{"id":"…"}`. Unknown fields rejected. |
| POST   | `/api/v1/sessions/stop`                     | Finalizes current session, drains records, releases capture; returns `{"stopped":true}`.                     |
| GET    | `/api/v1/sessions`                          | Most recent 200 sessions for the default profile.                                                            |
| GET    | `/api/v1/sessions/{sid}/report`             | Object with `session`, `summary`, `metrics`, and `events`.                                                   |
| GET    | `/api/v1/sessions/{sid}/export?format=json` | Download JSON; `csv` and `html` also accepted. Other formats rejected.                                       |
| DELETE | `/api/v1/sessions/{sid}`                    | Deletes a completed session and dependent metrics/events; active sessions rejected.                          |
| POST   | `/api/v1/alerts/acknowledge`                | Silences current audio and records acknowledgement when applicable; returns `{"acknowledged":true}`.         |
| POST   | `/api/v1/calibration`                       | Starts calibration in an active live session; replay/idle rejected. Returns progress fields.                 |

```sh
curl http://127.0.0.1:8000/api/v1/health
curl -X POST http://127.0.0.1:8000/api/v1/sessions \
  -H 'Content-Type: application/json' -d '{"mode":"replay"}'
curl -X POST http://127.0.0.1:8000/api/v1/sessions/stop
```

Use the returned session ID for reports and exports. Stop before deleting; export first if you need a copy. Reports are restricted to the default profile in this local application, not arbitrary supplied profile IDs.

## Assistant and voice

| Method | Route                          | Request / result                                                                                                                                                     |
| ------ | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GET    | `/api/v1/assistant/status`     | Configured flag, model, daily usage/cap, interval, knowledge state, last worker source/limitation. Never returns the key.                                            |
| POST   | `/api/v1/assistant/reload`     | Reloads Gemini environment settings; 409 while active/queued, 422 for invalid environment. Returns status.                                                           |
| POST   | `/api/v1/assistant`            | Validated topic-only `AssistantRequest`; response described below.                                                                                                   |
| POST   | `/api/v1/voice/local`          | `{"text":"How should I calibrate?"}`; at most 1000 characters, local processing only.                                                                                |
| POST   | `/api/v1/voice/listen`         | Starts five-second local recording/transcription; returns transcript plus assistant response. 409 for unavailable setup/busy, 503 for capture/transcription failure. |
| GET    | `/api/v1/knowledge/{identity}` | Excerpt content and metadata for a citation; 404 if unknown.                                                                                                         |

Cloud request examples:

```json
{ "mode": "local" }
```

```json
{ "mode": "general", "topic": "calibration" }
```

```json
{ "mode": "synthetic", "scenario": "eye_closure" }
```

Allowed topics: `fatigue`, `breaks`, `calibration`, `limitations`. Allowed scenarios: `eye_closure`, `trip_summary`. No free-form question, telemetry, transcript, or driver ID field exists on this endpoint. Unknown fields are rejected.

An `AssistantResponse` has `answer`, `source`, `citations`, `limitation`, and `usage`. Source is `local`, `gemini`, or `synthetic`; `synthetic` means a generated response to a predefined artificial scenario. A local fallback uses `local`, even when the requested mode was cloud. Citations contain `id`, `title`, `section`, `source`, and a local excerpt URL. `usage` may be empty. Inspect `limitation` and the current response, not just status: the status endpoint's last-result fields are updated by cloud-worker requests and are not a complete conversation history.

## Telemetry and preview WebSockets

| Channel            | Payload / behavior                                                                                                                 |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `/ws/v1/telemetry` | Whole current health object, approximately 5 Hz. Includes nested snapshot and latest alert; not a durable event stream.            |
| `/ws/v1/preview`   | `{"image":"<base64 JPEG>"}` only when a distinct preview exists; checked every 0.15 seconds. Decode as `data:image/jpeg;base64,…`. |
| `/ws/telemetry`    | Compatibility wrapper: `{"topic":"FATIGUE_SCORE","payload":<snapshot or null>}`. New consumers should use the versioned route.     |

Each connection has a send task plus disconnect watcher; send deadline is one second. Idle preview connections can receive no frames. The server's inbound WebSocket message bound is 8192 bytes. Use an Origin such as `http://127.0.0.1:8000`; a missing Origin is rejected. The protocol provides snapshots, not guaranteed replay of every alert transition—query session history for durable events.

### MonitoringSnapshot

Fields: contract `version`, `session_id`, increasing `frame_id`, epoch-second `captured_at` and `processed_at`, `state`, `validity`, `measurements`, `score`, `level`, `reasons`, `synthetic`, and `calibration`.

States are `initializing`, `calibrating`, `monitoring`, `degraded`, and `unavailable`. `idle` belongs to the top-level health object when no snapshot exists. Measurements contain EAR, MAR, pitch/yaw, eye-closure seconds, and PERCLOS where available. Treat absent values and JSON null as unavailable; never replace them with healthy-looking zeroes.

### AlertEvent

Fields: event `id`, `session_id`, `severity`, `cause`, `transition`, epoch-second `timestamp`, `acknowledged`, and `message`. Transitions are `triggered`, `reminder`, `resolved`, and `acknowledged`. A resolved event can still have severity `critical`; check transition. An alert can also be critical due to prolonged closure while the numeric score's level is lower.
