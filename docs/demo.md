# Guided demonstration

[Setup](setup.md) · [Assistant](assistant.md) · [Verification](verification.md)

Use synthetic replay for a repeatable walkthrough without anyone on camera. It is timestamped telemetry, **not prerecorded video or evidence of real-person detection**. Keep its synthetic label visible in recordings.

## Prepare

Run `uv run --locked dms check`, then `uv run --locked dms run`. Open `http://127.0.0.1:8000`. Python, the face asset, and storage should be ready. Camera/microphone status can correctly say “not probed.” A key is optional.

For a private demonstration that must not touch existing history, launch with a fresh data directory. On macOS/Linux:

```sh
DEMO_DATA=$(mktemp -d)
DATA_DIR="$DEMO_DATA" GEMINI_API_KEY='' AUDIO_ENABLED=false \
  uv run --locked dms run
```

Keep that terminal open. This example intentionally disables cloud generation and speech. It still loads installed optional local models if present. When finished, use Ctrl+C; export any demo records before discarding the temporary directory.

## Replay timeline

Click **Start replay**. Current fixture intervals are:

| Elapsed time | Input / observation                                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| 0–4.9 s      | Frontal valid geometry; EAR 0.30 and MAR 0.20.                                                                                         |
| 5–6.9 s      | Artificial eye closure, EAR 0.10. With default uncalibrated settings, closure reaches the 1.5-second critical rule around 6.5 seconds. |
| 7–9.9 s      | Eyes reopen; recovery depends on score, accumulated PERCLOS, and hysteresis.                                                           |
| 10–12.9 s    | Artificial mouth opening, MAR 0.80; this is not a validated yawn classification.                                                       |
| 13–14.9 s    | Normal mouth geometry.                                                                                                                 |
| 15–17.9 s    | Artificial head turn, yaw 35 degrees.                                                                                                  |
| 18–19.9 s    | Frontal geometry returns.                                                                                                              |
| 20–21.9 s    | Face loss; unavailable measurements and null score.                                                                                    |
| 22–27.9 s    | Valid geometry returns.                                                                                                                |
| 28–30 s      | Face loss, then replay completion/unavailable state.                                                                                   |

Timing/alerts can differ with changed thresholds or a saved calibration. Use an isolated default-profile store for a reproducible baseline. Not every scripted movement must trigger a separate warning: the policy evaluates combined signals and timing.

When the critical banner appears, explain its cause. **Acknowledge alert** silences the announcement without asserting recovery. Audible speech requires macOS, enabled audio, and a working output device. Stop the session explicitly at the end.

## Inspect the record

Open **History → Open report**. Review duration, sample-based tracking coverage, triggered alert count, score trend, and event timeline. Coverage excludes degraded/calibrating/unavailable records; a chart is not a continuous raw-video record.

Use **Export JSON**, **Export CSV**, and **Export HTML**. JSON carries the full report object; CSV contains selected metrics; HTML is a printable summary/event list. To demonstrate deletion, use only a disposable session and show Cancel before confirming. Exports/backups are independent copies and are not removed by session deletion.

## Explain assistance

1. Open Assistant and use **Get local status**. After stopping, an unavailable/no-current-measurement response is expected.
2. Ask locally: “How should calibration use lighting?” Open a citation and inspect the supporting text.
3. Select a curated cloud topic or artificial scenario. Without a key, show the explicit local fallback. With a configured key, use one intentional bounded request and identify whether the response source is generated or local.
4. Show model/usage status and **Reload assistant settings**. Do not open `.env` or reveal credentials while recording.
5. Explain optional five-second local voice without implying a recording occurred. If demonstrating it, click intentionally and keep private speech out of the recording.

Never replace failed provider output with an unlabeled prerecorded answer. If you shorten waits in an edited tour, disclose that editing. The assistant never receives this real session's measurements or history.

## Optional stationary webcam and calibration

Start webcam only when ready to request camera access. Remain stationary, improve lighting, and confirm measurement validity. In Setup, start calibration and collect 20 valid frontal-face seconds within 60 seconds. Observe the actual status; do not present replay or a manually edited progress bar as a completed human calibration.

Stop the session and close the server with Ctrl+C when the demonstration ends. A tutorial can show calibration controls without completing physical calibration, provided the narration says so.
