# Component reference

[Architecture](architecture.md) · [API](api.md) · [Configuration](configuration.md)

This reference describes the active implementation, not historical detectors or planned capabilities. Paths below identify ownership for contributors.

## Entry point and configuration

| Source                                          | Responsibility                                                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| [main.py](../main.py)                           | Supported `dms` CLI; settings load, saved preferences, setup/import commands, Uvicorn launch and optional replay autostart. |
| [core/config.py](../core/config.py)             | Validated defaults, repository-relative `.env`, loopback binding, thresholds, model/data paths.                             |
| [core/setup.py](../core/setup.py)               | Readiness inspection, face checksum verification, explicit optional downloads, installed-file inventory.                    |
| [models/manifest.json](../models/manifest.json) | Expected asset identity, size/checksum and upstream source/license metadata.                                                |

`check` reports configuration/readiness rather than proving end-to-end health. For example, a Gemini key can be “configured, not contacted”; an embedding directory can exist even if optional imports later fail. The face model is verified byte-for-byte. Optional asset inventory records provenance at installation but is not a full recurring integrity audit.

## Capture and perception

[core/capture.py](../core/capture.py) implements `LatestProcessor`. Its input and result queues hold at most one item. A single perception executor runs inference and cleanup; there are no parallel MediaPipe tracking owners.

[perception/pipeline.py](../perception/pipeline.py) letterboxes native camera frames into the configured dimensions, default 640×480, preserving aspect ratio. The model runs on CPU for one face. Eye and mouth ratios use pixel-scaled landmarks; normalized coordinates alone would distort ratios on non-square frames.

| Output      | Meaning and limits                                                                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EAR         | Mean ratio of vertical eye distances to horizontal width across two eyes; a geometry measure, not a probability.                                                    |
| MAR         | Mouth opening normalized by mouth width; an open mouth is not a validated yawn classification.                                                                      |
| Pitch / yaw | Approximate angles from six image points, a canonical 3D face, and an assumed camera matrix using OpenCV SQPNP. No per-camera intrinsic calibration is implemented. |
| Preview     | Annotated JPEG encoded in memory, delivered only over the separate local preview socket.                                                                            |

Roll is computed inside the helper but is not part of normal monitoring telemetry. There is no active heart-rate estimator, learned forecasting model, gaze score, or micro-expression classifier. Synthetic replay never invokes this image-processing component.

## Monitoring and calibration

[core/monitoring.py](../core/monitoring.py) accepts increasing capture timestamps and frame IDs. It rejects non-finite or implausible scalar inputs. Required EAR/MAR and face presence determine measurement validity; pose validity is tracked separately.

### Temporal signals

Eye closure begins when EAR falls below the configured/calibrated threshold. Missing required geometry or a gap greater than two seconds resets closure accumulation and the PERCLOS window. PERCLOS weights retained valid intervals by elapsed time over an approximately 60-second window; it uses the previous sample's eye state for each interval. It is not a long-term driver confidence estimate, and the window may contain far less than 60 seconds just after startup or tracking loss.

### Fixed heuristic score

The following describes the implementation, not clinically established weights. Let `clip(x)` mean clamp to 0–1; `baseline` defaults to 0.30 until calibration changes it:

```text
score = 35 × clip((baseline − EAR) / (baseline × 0.5))
      + 35 × clip(PERCLOS / 0.4)
      + 15 × clip((MAR − 0.35) / 0.4)
      + 15 × clip(abs(yaw) / 45)     [only when pose is valid]
```

Clamp the sum to 100 and round to one decimal. Missing pose contributes zero without reweighting the other terms. Score labels are safe below 25, mild from 25, warning at the configured warning threshold, and critical at the configured critical threshold. The word `safe` is a code/UI category, not a declaration that driving is safe.

A prolonged-closure reason can trigger a critical alert even while the numeric score is below the critical threshold. Clients must not equate the snapshot's `level` with the separate active alert state.

### Calibration

Calibration runs only in a live session. Frontal samples require absolute pitch and yaw below 15 degrees and EAR between 0.20 and 0.50. Each valid interval contributes at most 0.25 seconds. Invalid samples break accumulation between adjacent samples but do not erase valid seconds already collected. At 20 valid seconds, median EAR becomes the baseline and 65% of that baseline becomes the closure threshold. At the 60-second deadline, insufficient progress ends with an explanatory message.

The runtime stores each completed baseline and loads the latest one for the default profile at session start. There is no profile-selection UI or calibration-management screen. Calibration adjusts a threshold; it does not validate classification accuracy.

## Safety policy and audio

[agents/safety_agent.py](../agents/safety_agent.py) evaluates snapshots in `monitoring` or `degraded` state with a non-null score. It is deterministic and receives no assistant response.

| Behavior                 | Default                                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Warning / critical score | 55 / 72, sustained for one second                                                                                  |
| Prolonged eye closure    | Critical escalation at 1.5 seconds, bypassing the score-sustain wait                                               |
| Recovery                 | Score below the active threshold minus five points for three seconds, without prolonged closure                    |
| Reminder                 | At least eight seconds apart while the matching condition remains active and unacknowledged                        |
| Acknowledgement          | Silences speech and suppresses further reminders for that active alert; does not change measurements or resolve it |

Invalid/unavailable snapshots stop pending evaluation; they do not prove recovery. A resolved event retains its prior severity but has transition `resolved`, so severity alone is insufficient for rendering an active alarm.

[alerts/alert_engine.py](../alerts/alert_engine.py) owns a serialized macOS `say` worker. Critical submissions terminate current speech and clear waiting messages. It does not invoke a vehicle horn or emergency service. Unsupported audio or subprocess failure is surfaced through health; enqueueing a speech test does not establish audibility.

## Runtime and optional events

[core/runtime.py](../core/runtime.py) owns tasks, session IDs, latest snapshot/preview, latency windows, calibration persistence, and resource cleanup. It evaluates safety before optional publication. [core/bus.py](../core/bus.py) gives each subscriber a bounded queue and consumer task; old optional events can be coalesced. Event statistics report bus activity, not durable delivery guarantees.

[core/contracts.py](../core/contracts.py) contains Pydantic telemetry/event contracts. Unknown fields are forbidden in these models; measurements must be finite or null. `ComponentHealth` is defined, but the health endpoint currently returns component/error dictionaries rather than a stream of typed `ComponentHealth` objects.

## Storage and reports

[memory/store.py](../memory/store.py) manages SQLite WAL, foreign-key cascades, transactions, interrupted-session recovery, retention, and cloud usage reservations. One recorder samples ordinary snapshots approximately once per second and records transitions separately. Unavailability can add extra snapshots outside that cadence.

The report contains:

| Field               | Exact interpretation                                                                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `duration_s`        | Session end minus start, or current time minus start while still open.                                                                                                          |
| `samples`           | Number of stored metric records.                                                                                                                                                |
| `tracking_coverage` | Fraction of stored records whose state is exactly `monitoring`; excludes degraded/calibrating/unavailable. Null with no records. This is sample-based, not exact time coverage. |
| `alert_count`       | Number of stored events with transition `triggered`; excludes reminders/acknowledgements.                                                                                       |
| `max_score`         | Maximum non-null stored score, or null if none.                                                                                                                                 |

JSON exports the complete report object; CSV exports selected metric fields; HTML exports a printable summary and event list. These formats are not identical replicas of the dashboard. Retention removes completed sessions older than the cutoff, including derived metrics/events, but does not remove calibrations, usage counters, downloaded models, or backups.

Legacy import reads the source database without changing it, maps supported session/event fields, and omits arbitrary payload data. It is not a complete reconstruction of old metrics. Schema migrations back up an existing older database first; newer unsupported schemas are refused. See [storage and backup](privacy.md).

## Assistant, retrieval, and voice

| Source                                  | Responsibility                                                                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [ai/assistant.py](../ai/assistant.py)   | Local templates, topic/scenario allowlists, cloud worker, usage reservations, structured-output/citation validation, fallbacks. |
| [rag/knowledge.py](../rag/knowledge.py) | Top-level Markdown/TXT ingestion, checksums, bounded paragraph excerpts, BM25-style scoring, optional local dense ranking.      |
| [voice/local.py](../voice/local.py)     | Explicit five-second in-memory capture, English CPU/int8 Faster-Whisper transcription with VAD.                                 |
| `docs/knowledge/`                       | Small approved corpus supplied to retrieval; operational docs elsewhere are not automatically included.                         |

See [Assistant and retrieval](assistant.md) for modes, source labels, limits, reindex timing, and cloud boundaries. Retrieval is not a web search agent. Real history is not placed in the knowledge index.

## Web application and verification tools

[ui/web/server.py](../ui/web/server.py) owns routes, origin/size restrictions, security headers, reports/exports, and independent WebSocket senders. [ui/web/static/app.js](../ui/web/static/app.js) owns navigation, rendering, reconnect feedback, API actions, citation expansion, and the deletion dialog. `style.css` owns tokens, responsive layout, focus styles, and printing; `index.html` defines six views.

| Tool                                                    | Purpose                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `tests/`                                                | Deterministic unit/integration regressions; cloud uses mocks, device access is excluded except a synthetic-image native model smoke check. |
| [scripts/browser_check.py](../scripts/browser_check.py) | Starts/deletes a replay and changes/restores preferences against a running server. Use temporary data. Screenshot saving is opt-in.        |
| [scripts/benchmark.py](../scripts/benchmark.py)         | Accelerated monitor/policy load test; no video inference.                                                                                  |
| [scripts/soak.py](../scripts/soak.py)                   | Real-time synthetic ingestion into a temporary store; no webcam or audible output.                                                         |
| [.github/workflows/ci.yml](../.github/workflows/ci.yml) | Ubuntu Python checks, CLI readiness smoke, and browser workflow job.                                                                       |

None of these tools establishes real-driver detection accuracy. [Verification](verification.md) separates test results from physical acceptance.
