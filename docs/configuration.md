# Configuration reference

[Setup](setup.md) · [API settings](api.md#settings-and-readiness) · [Troubleshooting](troubleshooting.md)

[core/config.py](../core/config.py) defines validated settings. Field names are accepted as environment variables, conventionally uppercase. `.env` is resolved relative to the repository, not the terminal's current directory. Unknown `.env` entries are ignored. Invalid recognized values can prevent startup.

## Precedence and reload

For the CLI, settings are assembled in this order:

1. Pydantic defaults, overlaid by repository `.env`.
2. Process environment overrides `.env`.
3. Saved Settings-page fields in `DATA_DIR/settings.json` override those values when the CLI starts.

A path change to `DATA_DIR` selects a different settings/history directory. Relative configured paths resolve against the process working directory; use absolute paths for custom locations.

Settings-page changes are allowed only with no active session. They update runtime configuration and are saved atomically; monitoring picks them up at the next start. They do not rename or move the database.

**Reload assistant settings** rereads only key/model/daily-cap/spacing from environment and `.env`. It refuses reload while a cloud request is active or queued. It preserves monitoring and database usage counters. Other `.env` changes require a server restart; saved UI settings can still take precedence.

## Camera and server

| Variable          | Default     | Validation / behavior                                                      |
| ----------------- | ----------- | -------------------------------------------------------------------------- |
| `CAMERA_INDEX`    | `0`         | Integer 0–20; operating-system capture device index.                       |
| `CAMERA_WIDTH`    | `640`       | 160–1920 pixels; requested capture width and letterboxed inference width.  |
| `CAMERA_HEIGHT`   | `480`       | 120–1080 pixels; requested height and inference height.                    |
| `CAMERA_FPS`      | `30`        | Target 5–60; the device may supply a different mode/rate.                  |
| `FLIP_HORIZONTAL` | `true`      | Mirror live frames before perception. Environment-only.                    |
| `WEB_HOST`        | `127.0.0.1` | Must be `127.0.0.1`, `localhost`, or `::1`.                                |
| `WEB_PORT`        | `8000`      | Integer 1024–65535; HTTP and allowed browser origins use this port.        |
| `MAX_FRAME_AGE_S` | `2`         | 0.2–5 seconds; stale-frame bound and watchdog threshold. Environment-only. |

On the dashboard, camera fields can be saved along with audio, warning/critical thresholds, closure duration, and retention. Server binding, paths, and Gemini configuration are not writable through that settings endpoint.

## Signal and alert defaults

| Variable                   | Default | Validation / behavior                                                                                                  |
| -------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| `FATIGUE_WARN_THRESHOLD`   | `55`    | Score 1–95.                                                                                                            |
| `FATIGUE_CRIT_THRESHOLD`   | `72`    | Score 6–100; must be at least warning + 5.                                                                             |
| `MICROSLEEP_EAR_THRESHOLD` | `0.18`  | EAR 0.05–0.4 before a saved calibration overrides it. The historical variable name is not a clinical microsleep claim. |
| `EYE_CLOSURE_S`            | `1.5`   | 0.5–5 seconds for prolonged-closure escalation.                                                                        |
| `ALERT_COOLDOWN_S`         | `8`     | At least 8 seconds between reminders.                                                                                  |
| `AUDIO_ENABLED`            | `true`  | macOS speech output; visual monitoring is independent.                                                                 |
| `RETENTION_DAYS`           | `30`    | Integer 1–365; expires completed sessions by end time.                                                                 |

Score crossing must persist for one second; exit hysteresis is five points and recovery is three seconds. These timings are code constants, not additional environment settings. [Monitoring reference](components.md#monitoring-and-calibration) explains scoring and validity.

## Gemini

| Variable            | Default                 | Behavior                                                                                           |
| ------------------- | ----------------------- | -------------------------------------------------------------------------------------------------- |
| `GEMINI_API_KEY`    | Empty                   | Optional backend secret; omit for fully local operation.                                           |
| `GEMINI_MODEL`      | `gemini-3.5-flash-lite` | Provider model identifier. Check access for your own project.                                      |
| `GEMINI_DAILY_CAP`  | `20`                    | 0–20 attempted generations per UTC day, persisted in SQLite. Set 0 to disable outbound generation. |
| `GEMINI_INTERVAL_S` | `10`                    | At least ten seconds between attempt reservations.                                                 |

Output cap, single-worker concurrency, waiting-queue capacity, and generation deadline are fixed in the assistant implementation. They are not environment tunables. A queued request may wait longer than the generation deadline, which starts in the worker after retrieval and usage reservation.

## Paths and local state

Defaults below are relative to the repository unless stated otherwise.

| Variable         | Default                    | Purpose                                                       |
| ---------------- | -------------------------- | ------------------------------------------------------------- |
| `DATA_DIR`       | `data/sentinel/`           | SQLite, settings, runtime lock, and optional knowledge index. |
| `MODEL_PATH`     | `face_landmarker.task`     | Face model validated against the manifest.                    |
| `KNOWLEDGE_DIR`  | `docs/knowledge/`          | Approved top-level Markdown/TXT corpus.                       |
| `EMBEDDING_PATH` | `models/all-MiniLM-L6-v2/` | Explicitly installed local embedding model.                   |
| `WHISPER_PATH`   | `models/whisper-base-en/`  | Explicitly installed local speech model.                      |

The inherited names `data/sentinel/`, `sentinel.db`, SQLite table names, and dense collection identifier remain for compatibility. Product naming changes do not migrate stored data. The database is `DATA_DIR/sentinel.db`; preferences are `DATA_DIR/settings.json`. Preserve both when backing up.
