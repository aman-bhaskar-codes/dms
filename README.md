# Sentinel V6

A local driver-monitoring **demonstration** for Apple Silicon Macs. Webcam geometry, deterministic alerts, calibration, session history, reports, and replay work locally. Optional Gemini assistance handles curated general questions and predefined synthetic scenarios.

![Sentinel dashboard](docs/design/dashboard.png)

## Start here

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). From this checkout:

```sh
uv sync --locked --extra gemini
uv run sentinel check
uv run sentinel run
```

Open **http://127.0.0.1:8000**. Choose **Start replay** for a 30-second artificial scenario, or **Start webcam** while stationary. Allow camera access for the launching terminal in macOS System Settings → Privacy & Security → Camera.

For automatic replay startup:

```sh
uv run sentinel run --replay
```

The bundled Face Landmarker asset is verified by checksum. To restore it explicitly:

```sh
uv run sentinel setup-models
```

Application startup never downloads model assets. A new machine needs internet for dependency installation; core operation then works offline. The supported installation is this source checkout with `uv`, not the historical V4 setup commands.

## Gemini, optionally

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` on the backend. Never put a key in JavaScript. `GEMINI_MODEL` selects the model; the default is `gemini-2.5-flash`. Verify availability and your project's quota in [AI Studio](https://aistudio.google.com/).

The UI supports **local status**, **general knowledge**, and **synthetic demo**. General knowledge sends only a curated question and approved local knowledge excerpts. Arbitrary user text, webcam frames, recordings, driver identifiers, measurements, and real trip history are not accepted by the cloud endpoint. Real trip reports use local templates.

Gemini is limited to 20 attempts per UTC day, at least 10 seconds apart, 512 output tokens, one active request, three queued requests, and a 15-second generation deadline. Usage persists in SQLite. Provider quotas can be lower; no paid tools or paid-model fallback are enabled. Keep the Gemini project on its unpaid tier for zero service charges. Account billing is controlled in AI Studio, not enforceable by this application.

Read [Google's unpaid-service data terms](https://ai.google.dev/gemini-api/terms) before using free cloud inference. A valid citation ID is checked mechanically; semantic answer correctness still needs human evaluation.

## Local knowledge and voice extras

The base installation includes approved text documents and local lexical retrieval. Optional hybrid retrieval uses Chroma, local sentence-transformer embeddings, and BM25 rank fusion:

```sh
uv sync --locked --extra gemini --extra knowledge
uv run sentinel setup-models --knowledge
```

Local voice requires a downloaded Whisper model. Downloads happen only on explicit setup; model revisions, sizes, and checksums are recorded locally:

```sh
uv sync --locked --extra gemini --extra knowledge --extra voice
uv run sentinel setup-models --knowledge --voice
```

In Assistant, **Record locally · 5 sec** captures five seconds after the click, transcribes locally, and handles status/fatigue/break/calibration intents. The microphone is off otherwise. macOS `say` provides local speech; critical alerts preempt assistant speech. No raw video or audio is persisted.

## Monitoring behavior

- Time-based eye closure and rolling PERCLOS; EAR, mouth ratio, and head pose.
- Missing face means **unavailable**. Missing head pose means **degraded**, not a fabricated zero reading.
- Guided calibration needs 20 valid frontal-face seconds within a 60-second window.
- Initial demonstration thresholds: warning 55, critical 72; one-second sustained score crossing; prolonged eye closure 1.5 seconds; five-point exit hysteresis; three-second recovery; eight-second reminders.
- Acknowledgement silences the current announcement without clearing the condition.
- No brightness-derived heart rate, micro-expression score, or three-minute prediction.

These heuristic defaults are not a validated fatigue classifier. The app cannot determine fitness to drive or control a vehicle. Stop safely and rest if tired, regardless of the score.

## History and settings

SQLite data is stored under `data/sentinel/`. Session snapshots are sampled at 1 Hz; transitions and failures are recorded separately in bounded, batched storage. No-face samples remain missing in reports. Export JSON, CSV, or printable HTML from History. Delete sessions through a confirmation dialog. Completed sessions older than 30 days expire by default; retention is configurable. Settings are saved locally and apply to the next session; environment settings initialize the application before saved UI preferences.

To import old session history without modifying the source database:

```sh
uv run sentinel import-legacy /absolute/path/to/old.db
```

See [architecture and API](docs/architecture.md), [demo walkthrough](docs/demo.md), and [verification record](docs/verification.md).

## Development checks

```sh
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run playwright install chromium
# In a separate terminal, start `uv run sentinel run` first:
uv run python scripts/browser_check.py
uv run python scripts/benchmark.py
# Explicit one-hour, real-time synthetic soak:
uv run python scripts/soak.py --seconds 3600
```

MediaPipe's native initialization and Chromium need normal process permissions on macOS; a restricted automation sandbox may abort native libraries. CI runs the Python tests on Linux. Browser checks use a running local server and mock no cloud responses.

## Design and legacy code

[Figma design specification](https://www.figma.com/design/xSyBgQHfVPA0M9Fm9A9Wqu) · [design contract](DESIGN.md) · [interaction contract](UX-CONTRACT.md).

The supported entry point is `sentinel` (implemented by `main_v5.py`). Older `main.py`, legacy dashboards, Ollama agents, research estimators, and evaluation scripts remain as historical source. They are not wired into V6 and are not supported launch paths. V5.1 stabilization and V6 capability changes are delivered together on this feature branch; hardware acceptance remains a release gate rather than an implied certification.
