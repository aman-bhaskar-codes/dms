# Verification record

[Contributing commands](contributing.md) · [Component behavior](components.md) · [Known limits](troubleshooting.md)

This page separates existing test evidence, short hardware observations, and acceptance still outstanding. DMS is a supervised demonstration, not a certified vehicle-safety device. No detection-accuracy study is included.

## Evidence at a glance

| Evidence                 | What it establishes                                                                                                                  | What it does not establish                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Python regressions       | Deterministic monitoring, alerts, lifecycle, storage, assistant/privacy boundaries, local voice mocks, native synthetic-image smoke. | Robustness across real drivers, lighting, or long hardware runs.                    |
| Chromium workflow script | Replay-to-report, local citations, settings validation/restoration, deletion, responsive layout and basic focus checks.              | A full accessibility audit, every browser, or comprehensive reconnect/zoom testing. |
| `benchmark.json`         | Accelerated monitor/policy processing of 108,000 synthetic frames.                                                                   | Camera FPS, MediaPipe inference speed, or sensor-to-speaker latency.                |
| `soak-smoke.json`        | Approximately 30 seconds of real-time synthetic ingestion and a bounded final Python allocation snapshot.                            | A completed one-hour soak or proof that native/process memory cannot grow.          |
| Short webcam observation | One machine's local processing rate and application latency over 15 seconds.                                                         | Repeatable performance on every PC or detection accuracy.                           |

## Current documentation audit

Documentation was checked against the application at baseline commit `82810fb` on 2026-10-30. The documentation update does not change runtime behavior. The current regression run passed **44 tests** with normal native-library permissions; cloud calls used mocks and no camera or microphone was opened.

The documentation update also passed Chromium acceptance against an isolated local replay server with no key or device capture, plus Ruff, focused mypy, Markdown formatting, and whitespace checks. The documentation audit resolved 131 local links and checked coverage of all 24 configuration fields and 22 API/WebSocket routes. Dashboard and assistant screenshots were refreshed from that current application.

This is a local result. The repository defines CI workflows, but a workflow file alone is not evidence of a successful remote run. Check GitHub's actual result for the commit being reviewed.

## Python and browser checks

The suite covers measurement validation, time-based closure/PERCLOS, calibration, alert timing/recovery/acknowledgement, bounded queues, storage recovery/export/deletion, assistant limits and citation validation, provider request shape, serialized reload, local question retrieval, voice VAD, neutral head pose, frame letterboxing, and optional-model startup.

The MediaPipe test processes a synthetic black image; it is not a face-detection accuracy test. macOS automation sandboxes can abort native MediaPipe initialization; use normal process permissions to distinguish that environment failure from a test assertion failure.

The browser script performs real local API actions and deletes its demo session. Use a temporary store, no cloud key, and audio off as described in [Contributing](contributing.md). It checks basic keyboard focus and layout overflow at selected viewport sizes, not complete WCAG conformance or every 200%-zoom flow. Static design audit findings are supplemental, not accessibility certification.

## Committed benchmark evidence

### Signal-policy load test

[benchmark.json](benchmark.json) records:

- 108,000 frames, representing 3600 simulated seconds.
- 16.898 seconds wall time; approximately 6391 signal frames/second.
- P95 measured policy-loop timing: 0.158 ms.
- Peak traced Python allocation: 340,813 bytes; retained window: 1801 samples.

The timing is an accelerated Python monitor/policy loop, not webcam throughput. It is a historical artifact without a complete machine/commit provenance record. Re-run with recorded environment details before comparing changes:

```sh
uv run --locked python scripts/benchmark.py --frames 108000
```

### Real-time synthetic smoke

[soak-smoke.json](soak-smoke.json) records 30.024 seconds, 861 synthetic frames, no reported component errors, and a final Python peak allocation of 148,373 bytes. Its capture-to-state metric refers to artificial timestamps, not physical camera input. `tracemalloc` excludes much native memory and is not a process RSS monitor.

The longer command exists but has not been established by this artifact:

```sh
uv run --locked python scripts/soak.py --seconds 3600
```

The script sleeps between generated samples and uses temporary storage; it is real-time synthetic ingestion, not “unthrottled” processing. Installed dense dependencies can still initialize during runtime startup; choose the environment deliberately for comparable runs.

## Historical local hardware and provider checks

The prior implementation session on 2026-10-29 recorded these observations on an Apple Silicon Mac:

| Check               | Observation                                                                                                                                                                            |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Webcam runtime      | 350 processed frames in 15.05 seconds: 23.26 FPS; P95 application capture-to-state latency 22.72 ms; no reported component errors.                                                     |
| Microphone          | Short default-input capture succeeded and was discarded.                                                                                                                               |
| Local transcription | Whisper recognized an artificial speech sentence; silence with VAD yielded no transcript.                                                                                              |
| Speech output       | A local macOS announcement completed without audio-worker error.                                                                                                                       |
| Dense knowledge     | Local SentenceTransformers/Chroma loaded offline and returned relevant excerpts.                                                                                                       |
| Gemini              | One real application request returned a calibration answer with a validated citation in 11.79 seconds; other requests timed out or returned provider failures and used local fallback. |

These are session observations; their complete raw hardware logs are not committed. They predate the repository consolidation and have not all been repeated in this documentation update. Treat them as limited historical evidence, not fresh acceptance of every current path.

Live cloud inputs used curated general questions and approved excerpts, not private driver data. Provider access is time-dependent. A configured key or model listing does not establish successful generation or ongoing quota.

### Latency definitions

The application timestamps capture after `cap.read()` returns, so capture-to-state excludes sensor exposure and time already spent inside that blocking read. Alert dispatch timing measures local policy/audio-queue dispatch, not audible speaker onset. P95 values are calculated from bounded in-memory samples, not a durable timing trace.

## Remaining supervised acceptance

- Complete a real human calibration and confirm baseline persistence/reuse.
- Exercise glasses, occlusion, camera position, and varied lighting while stationary.
- Physically disconnect/reconnect the camera and verify state, cleanup, and restart behavior.
- Verify audible warning/critical output, acknowledgement, and critical preemption in real use.
- Run a full 60-minute physical-camera test with process RSS, native memory, queue sizes, thermal behavior, and shutdown evidence.
- Evaluate full keyboard/screen-reader behavior, contrast, zoom, reconnect, and supported-browser coverage beyond current smoke checks.
- Confirm optional model asset provenance/license references; manifest metadata alone is not a completed redistribution audit.

Performance targets remain 15 processed FPS at default resolution, P95 application capture-to-state below 200 ms, and P95 local safety dispatch below 100 ms. They are targets, not guaranteed specifications. Report a failed target rather than silently lowering image quality or changing the measurement definition.
