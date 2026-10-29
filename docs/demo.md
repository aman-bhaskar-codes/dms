# Five-minute demonstration

1. Run `uv run sentinel check`; show the valid model checksum and local storage. Explain that voice and cloud are optional.
2. Start the dashboard and choose **Start replay**. The synthetic label remains visible throughout. At five seconds the fixture closes the eyes; after 1.5 seconds a critical alert appears. Acknowledge it and show that the condition remains visible.
3. At ten seconds show mouth opening; at fifteen show head turn; at twenty show missing face; at twenty-eight show synthetic camera failure. Missing tracking must never be displayed as safe.
4. Stop the session. Open History, inspect the report timeline and coverage, and export JSON/CSV/HTML. Explain that the report is generated locally from stored samples.
5. In Assistant, show local status without an API key. Select a general topic for Gemini if a key is configured. Expand the returned source excerpt. No free-form driver data is sent.
6. For a separate stationary hardware demonstration, start Webcam, grant permission, and calibrate. Check a normal blink and brief head turn. Do not stage fatigue while driving.
7. Disable internet or unset the Gemini key, restart, and show replay/local monitoring remains available. Test microphone-denied and missing-model guidance through Setup.

Before a public demo: verify audible output manually; confirm camera permissions; preinstall optional models; check current Gemini free quotas; use synthetic input for cloud examples. Never put credentials in recordings or screenshots.
