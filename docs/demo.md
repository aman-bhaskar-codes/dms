# Demonstration Guide

This guide walks through a 5-minute demonstration of the **DMS** driver-monitoring application using both the synthetic replay scenario and live camera capture.

---

## 1. Pre-flight Setup Check

Open a terminal in the project directory and verify system readiness:

```sh
uv run dms check
```

Ensure Python, the face landmark model, and SQLite storage report `ready`. If the model is unverified or missing, run:

```sh
uv run dms setup-models
```

---

## 2. Launch the Application

Start the local web server:

```sh
uv run dms run
```

Open **http://127.0.0.1:8000** in your web browser.

---

## 3. Synthetic Replay Walkthrough

Click **Start replay** on the Live Monitoring dashboard. A 30-second pre-recorded scenario will execute:

1. **Initial Monitoring (0–5s):** Notice the status badge switches to `Synthetic replay`. Facial metrics (EAR ~0.30, MAR ~0.20, pitch/yaw ~0°) are updated in real-time.
2. **Prolonged Eye Closure (5–7s):** The simulated driver closes their eyes (EAR drops to 0.10). After 1.5 continuous seconds, a `CRITICAL` alert triggers:
   - Audible voice warning: *"Stop at a safe place and take a break."*
   - Red visual banner on the dashboard.
3. **Alert Acknowledgement:** Click **Acknowledge alert**. Notice the audio is immediately silenced while the critical alert banner remains visible until the condition clears.
4. **Yawning / Mouth Opening (10–14s):** MAR spikes above 0.65, demonstrating yawn detection.
5. **Head Distraction (15–18s):** Head yaw exceeds 30°, illustrating attention loss.
6. **Face Occlusion / Loss (20–25s):** The simulated face disappears; tracking state switches to `unavailable` rather than fabricating safe zeroes.
7. **Session Completion:** Click **Stop session** to finalize the trip.

---

## 4. Session History and Local Reports

1. Navigate to the **History** tab in the sidebar.
2. Click **Open report** on the most recent session.
3. Review:
   - Trip duration and tracking coverage percentage.
   - Interactive SVG fatigue score trend chart.
   - Chronological event timeline showing trigger, acknowledgement, and resolution timestamps.
4. Test data exports:
   - Click **Export JSON** to inspect the structured payload.
   - Click **Export CSV** to open metrics in spreadsheet software.
   - Click **Export HTML** to view a printable summary report.

---

## 5. Sandboxed Assistant Demonstration

1. Navigate to the **Assistant** tab.
2. Click **Get local status** to retrieve immediate heuristic telemetry without network activity.
3. Under **General Knowledge**, select a curated topic (e.g. *Fatigue signs & rest precautions*):
   - If `GEMINI_API_KEY` is configured in `.env`, a concise answer with cited local evidence IDs appears.
   - Click **Citations** to expand and review the verbatim supporting excerpt.
   - If offline or unconfigured, DMS returns approved local knowledge directly.
4. Under **Synthetic Scenarios**, select *Prolonged eye closure* to see the assistant interpret an artificial session without exposing real driver data.

---

## 6. Stationary Live Camera & Calibration (Optional)

1. Return to the **Live** tab and click **Start webcam** while stationary.
2. Grant camera permissions in macOS if prompted.
3. Navigate to **Setup** and click **Start calibration**:
   - Maintain a natural forward gaze for 20 continuous valid frontal seconds.
   - The system computes personalized baseline EAR and adjusts the microsleep threshold.
   - A success notification confirms the calibrated baseline is stored.
4. Click **Stop session** when finished.
