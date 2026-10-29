# Verification Record

DMS is a local driver-monitoring demonstration and research prototype, not a certified vehicle safety device.

---

## 1. Automated Verification Suite

The automated test suite verifies signal validity, time-based PERCLOS calculation, eye-closure timing, guided calibration, deterministic alert state transitions, alert preemption and acknowledgement, queue isolation, SQLite persistence, and cloud boundary privacy constraints.

### Execution Commands

```sh
# Unit and integration test suite
uv run pytest -q

# Code formatting and linting
uv run ruff check .
uv run ruff format --check .

# Static type verification
uv run mypy

# End-to-end browser workflow verification
uv run python scripts/browser_check.py
```

### Test Coverage Highlights

- **Signal Validation:** Rejects non-finite/NaN values, out-of-bounds metrics, and stale timestamps.
- **PERCLOS & Eye Closure:** Verifies rolling 60-second time-weighted accumulation and immediate alert triggers upon sustained closure ($\ge 1.5\text{s}$).
- **Guided Calibration:** Validates baseline calculation over 20 seconds of frontal samples within a 60-second window.
- **Alert State Machine:** Exercises Warning (55), Critical (72), 5-point exit hysteresis, 3-second recovery, and 8-second reminder cooldowns.
- **Queue Isolation:** Confirms that video, audio, telemetry, and storage buffers remain strictly isolated with independent backpressure bounds.
- **Persistence:** Tests SQLite WAL transactions, 1 Hz metric downsampling, cascaded session deletion, and graceful recovery of interrupted sessions.
- **Cloud Boundary:** Enforces daily usage caps (20 requests/UTC day), strict JSON schemas, citation ID validation against local documents, and refusal of free-form driver data.

---

## 2. Performance and Soak Benchmarks

### Offline Signal-Policy Load Test

`scripts/benchmark.py` evaluates the speed and memory scaling of signal processing and safety policy evaluation without camera or MediaPipe overhead:

```sh
uv run python scripts/benchmark.py --frames 108000
```

- **Target:** $>1000\text{ frames/s}$, P95 policy evaluation latency $<1\text{ ms}$, bounded memory growth.
- **Recorded Evidence (`docs/benchmark.json`):** 108,000 simulated frames processed in 16.9 seconds ($6,391\text{ frames/s}$ throughput, P95 policy latency $0.16\text{ ms}$, peak memory $340\text{ KB}$).

### Synthetic Replay Soak Test

`scripts/soak.py` executes an unthrottled real-time synthetic replay soak writing to temporary storage:

```sh
uv run python scripts/soak.py --seconds 3600
```

- **Target:** Continuous 60-minute execution without memory leaks or buffer accumulation.
- **Recorded Evidence (`docs/soak-smoke.json`):** Verified stable heap memory and queue sizes during extended execution.

---

## 3. Browser End-to-End Verification

`scripts/browser_check.py` exercises full user workflows in headless Chromium:

1. **Replay to Report:** Starts synthetic replay, observes live metric updates, stops session, opens history, and verifies chart rendering.
2. **Local Assistant:** Queries local status and tests general question retrieval with citation expander.
3. **Settings Validation:** Submits invalid threshold order (warning $>$ critical) and verifies inline error rejection; restores valid settings.
4. **Session Deletion:** Tests cancel behavior and keyboard focus trap inside the deletion dialog, then confirms permanent removal.
5. **Accessibility & Responsive:** Verifies viewport scaling across $1280\times 800$, $640\times 900$, and $640\times 400$, reduced motion media emulation, and keyboard tab focus traversal.

---

## 4. Remaining Physical Acceptance Checks

The following hardware evaluations require a supervised physical testbed:

1. **In-Vehicle Lighting Variation:** Direct sunlight, nighttime road glare, tunnel transitions, and backlighting.
2. **Eyewear & Occlusions:** Testing tracking robustness with polarized sunglasses, prescription eyeglasses, hats, and partial face coverings.
3. **Hardware Disconnection:** Physical USB camera unplug/replug during active live monitoring.
4. **Physical In-Vehicle Soak:** Continuous 60-minute live webcam capture with thermal monitoring on Apple Silicon hardware.
