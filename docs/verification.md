# Verification record

This is an implementation candidate, not a certified driver-safety product.

## Automated evidence

The test suite exercises signal validity, time-based PERCLOS/closure, calibration, state transitions, acknowledgement and recovery, queue isolation, replay lifecycle, SQLite reports/deletion/restart, persistent Gemini caps, privacy validation, cited-response validation, and local API origin enforcement. The MediaPipe smoke test uses a black image and no camera. Browser checks cover replay → report, local assistance, invalid settings, keyboard focus, reduced motion and responsive layout.

See the final implementation ledger for command results and docs/design/premium-audit.json for the static UI audit. Static audit success does not establish accessibility conformance.

## Performance

`python scripts/benchmark.py` is an accelerated **signal-policy-only** load test. It does not measure MediaPipe, camera capture, driver detection accuracy, or end-to-end video latency. `python scripts/soak.py --seconds 3600` is an explicit real-time synthetic soak that writes only to a temporary directory.

Hardware acceptance targets remain: 15 processed frames/s at 640×480, capture-to-state P95 <200 ms, safety-to-audio-dispatch P95 <100 ms, and a 60-minute physical-camera soak without sustained memory growth. Report target failures rather than reducing model accuracy silently.

## Checks requiring local hardware or credentials

- Real webcam permissions, calibrated face geometry, glasses/lighting variation, camera unplug/replug and measured FPS.
- Microphone permission, speech transcription quality and audible critical preemption.
- Live Gemini model availability, current free quota and human review of answer grounding.
- Full one-hour camera soak and device-driver shutdown behavior.

No API key was supplied during implementation. No live cloud inference or camera recording is required for automated checks. Optional dense/voice models must be explicitly installed; missing extras are a supported degraded mode.

## Recorded results — 2026-10-29

| Check | Result |
| --- | --- |
| Full Python suite, native model smoke included | 33 passed |
| Ruff lint / formatting | Passed |
| Focused mypy | Passed |
| JavaScript syntax / Prettier | Passed |
| Chromium acceptance flow | Passed; no page errors |
| Premium static UI audit | Zero findings |
| Signal-policy load test | 108000 frames / 3600 simulated seconds; see benchmark.json |
| Real-time synthetic smoke | 30 seconds; no component errors; see soak-smoke.json |

The signal-policy benchmark is not a vision FPS measurement, and the 30-second smoke is not the required one-hour physical-camera soak. Do not promote this implementation candidate to a real-driver release using these results alone.

Independent code review findings were fixed with regression tests: integer legacy session IDs, per-attempt Gemini retry accounting, and inherited mouth-opening geometry. A later regression check also made nonzero audio subprocess exits visible in health status.

Idle preview disconnection has a regression test; sender and disconnect watcher are cancelled and awaited together, so closed idle sockets cannot retain the runtime during server shutdown.

A real temporary Uvicorn process accepted and closed an idle preview WebSocket, then completed graceful shutdown in 0.182 seconds. This integration check used no camera, microphone, or cloud API.
