# Verification record

This is an implementation candidate, not a certified driver-safety product.

## Automated evidence

The test suite exercises signal validity, time-based PERCLOS/closure, calibration, state transitions, acknowledgement and recovery, queue isolation, replay lifecycle, SQLite reports/deletion/restart, persistent Gemini caps, privacy validation, cited-response validation, and local API origin enforcement. The MediaPipe smoke test uses a black image and no camera. Browser checks cover replay → report, local assistance, invalid settings, keyboard focus, reduced motion and responsive layout.

See the final implementation ledger for command results and docs/design/premium-audit.json for the static UI audit. Static audit success does not establish accessibility conformance.

## Performance

`python scripts/benchmark.py` is an accelerated **signal-policy-only** load test. It does not measure MediaPipe, camera capture, driver detection accuracy, or end-to-end video latency. `python scripts/soak.py --seconds 3600` is an explicit real-time synthetic soak that writes only to a temporary directory.

Hardware acceptance targets remain: 15 processed frames/s at 640×480, capture-to-state P95 <200 ms, safety-to-audio-dispatch P95 <100 ms, and a 60-minute physical-camera soak without sustained memory growth. Report target failures rather than reducing model accuracy silently.

## Recorded verification — 2026-10-29

The current automated suite covers monitoring, alerts, queue isolation, SQLite history and exports, quota/privacy/citations, assistant reload concurrency, local question retrieval, voice VAD, neutral head pose, camera sizing, and background optional-model startup. Chromium verified replay-to-report, local cited answers and excerpt expansion, assistant reload, settings validation/restoration, history deletion/cancel, responsive layout, keyboard focus and reduced motion; no page errors. Static premium UI audit returned zero findings.

### Live local hardware

- A 15.05-second webcam run processed 350 frames: **23.26 FPS**, **22.72 ms P95 capture-to-state latency**, with no component errors. Camera frames were resized/letterboxed to configured 640×480 before inference; no raw frames were saved or uploaded.
- MacBook Air microphone and speakers were detected. A short microphone capture succeeded and was discarded. Local Whisper transcribed an artificial macOS speech sentence correctly. Silence with VAD produced an empty transcript. Local audible speech completed without an audio-worker error.
- Local SentenceTransformer/Chroma retrieval loaded successfully offline and returned supporting safety documents. Optional dense loading now runs in the background so core startup remains available.

### Live Gemini

The key authenticated. The legacy SDK and generation path were replaced with official google-genai SDK 2 Interactions requests, `store=False`, standard service tier, structured JSON output, and bounded retries/timeouts. The configured/default model is gemini-3.5-flash-lite.

A real application request returned a generated calibration answer with one validated supporting citation in 11.79 seconds. Other live requests encountered provider availability/deadline failures; the dashboard returned actual local excerpts and citations, clearly marked local. Cloud availability is therefore not guaranteed. Only curated general questions and approved excerpts were sent; no video, audio, biometric values, driver history, or arbitrary transcripts were sent. Usage counters were preserved, and the key remains ignored by Git.

### Remaining physical acceptance

A full human calibration, glasses/lighting variation, camera disconnect/reconnect, audible critical interruption under real use, and a one-hour physical-camera/memory soak still require supervised acceptance. The short camera measurement is not an accuracy study or a one-hour soak. Accelerated signal-policy results (108000 frames/3600 simulated seconds) and the prior 30-second real-time synthetic smoke remain in benchmark.json and soak-smoke.json.

Native MediaPipe aborts under the restricted macOS automation sandbox; native-enabled regression runs are the relevant result. No product security policy was weakened for browser tests.

Final checks: **44 Python tests passed** with native permissions; Ruff lint/format, focused mypy, JavaScript syntax, Git whitespace checks, Chromium acceptance, and strict static UI audit passed. Moved-checkout executable entry points were reinstalled for the current path.
