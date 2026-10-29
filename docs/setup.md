# Setup

[Documentation index](README.md) · [Configuration](configuration.md) · [Troubleshooting](troubleshooting.md)

This guide installs the current source checkout. No hosted backend, cloud database, Docker setup, or frontend build step is required. Keep the repository alongside its assets: runtime paths assume the source tree contains `models/`, `fixtures/`, and `docs/knowledge/`.

## Platform support

| Environment         | Evidence and limits                                                                                                                                                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Apple Silicon macOS | Main development environment. Short webcam, local microphone, Whisper, speech output, and retrieval checks have run on one Mac. Hardware accuracy and long-duration behavior remain unvalidated.                                      |
| Linux               | GitHub Actions defines Python and Chromium jobs on Ubuntu. Core replay is intended to run without a webcam. Camera/microphone hardware has not been exercised here. `say` speech is unavailable; use visual alerts and disable audio. |
| Native Windows      | Startup uses `fcntl.flock`, so the current application cannot start natively. Installing Python or copying `.env` does not resolve this.                                                                                              |
| WSL / Intel macOS   | Not validated deployment targets. WSL hardware forwarding and audio are outside this guide; no working parity is claimed.                                                                                                             |

Use a local desktop browser. The server deliberately binds to loopback; access from another computer or phone is not supported.

## Prerequisites

1. Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Use Python 3.12; other Python minor releases are excluded by project metadata.
3. Allow network access during dependency installation and explicit model downloads.
4. For webcam use, provide a camera the operating system can open. A microphone is needed only for optional push-to-talk.

No universal RAM, disk-space, or frame-rate minimum has been established. Knowledge and voice extras add substantial packages and downloaded assets; start with core replay on an unfamiliar machine.

### Linux system libraries

The repository's Ubuntu workflow installs these libraries before Python dependencies:

```sh
sudo apt-get update
sudo apt-get install -y libgl1 libgles2 libegl1 libglib2.0-0 libportaudio2
```

These package names are for Ubuntu/Debian, not every Linux distribution. Disable speech in `.env` with `AUDIO_ENABLED=false`. Visual alert behavior remains available.

## Install and run core replay

```sh
git clone https://github.com/aman-bhaskar-codes/dms.git
cd dms
uv python install 3.12
uv sync --locked
uv run --locked dms --help
uv run --locked dms check
uv run --locked dms run --replay
```

Open **http://127.0.0.1:8000**. Confirm the interface says **Synthetic replay**. Replay feeds measurements into the same monitor and alert policy as live capture, without running image inference.

After about 30 seconds, the fixture ends and monitoring becomes unavailable. Click **Stop session** to finalize the record. Stop the server with **Ctrl+C**. Let shutdown finish so the recorder flushes, the database closes, and the camera/port are released.

`dms check` creates/checks writable storage, verifies the face asset, and reports optional readiness. It does not authenticate Gemini, open the microphone, or prove that speech is audible. Camera access is checked only with `dms check --probe-camera` or a live session.

## Restore the face model

The face asset is tracked in the checkout. If missing or different from the manifest checksum:

```sh
uv run --locked dms setup-models
```

This command downloads only when the face asset needs restoration. Startup never silently downloads it. Model provenance and expected size/checksum are in [the manifest](../models/manifest.json).

## Configure your environment

No API key is required for replay or webcam monitoring. To customize defaults, create `.env` from the template **only if you do not already have one**:

```sh
cp .env.example .env
```

Equivalent PowerShell file operation, for editing a checkout rather than claiming native Windows runtime support:

```powershell
Copy-Item .env.example .env
```

Edit `.env` with a text editor. Keep `GEMINI_API_KEY` private and backend-only. Do not paste it into the browser, a screenshot, logs, a commit, or a support issue. Existing Settings-page preferences override some `.env` values after restart; see [configuration precedence](configuration.md#precedence-and-reload).

## Add optional features

| Installation          | Commands                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------- |
| Gemini only           | `uv sync --locked --extra gemini`                                                        |
| Local dense knowledge | `uv sync --locked --extra knowledge` then `uv run --locked dms setup-models --knowledge` |
| Local voice           | `uv sync --locked --extra voice` then `uv run --locked dms setup-models --voice`         |
| All extras            | Commands below                                                                           |

```sh
uv sync --locked --extra gemini --extra knowledge --extra voice
uv run --locked dms setup-models --knowledge --voice
```

Use the same extras on later `uv sync` calls. An exact sync can remove extras omitted from that command; see [uv's sync behavior](https://docs.astral.sh/uv/concepts/projects/sync/). Development tools are included by the project's default development group.

Optional setup contacts Hugging Face, resolves a repository revision, and records installed-file checksums in `models/installed-models.json`. It may take time and disk space. Dense retrieval loads from disk in the background at startup; lexical retrieval works while it loads. Whisper loads on the first voice request, so the first recording can start later than the button click. Startup does not download either model.

Install the Gemini extra before using a configured key. Select a model your provider project can access; the repository's default is not a guarantee of availability. [Assistant behavior](assistant.md) describes limits and fallbacks.

## First stationary webcam session

1. Run `uv run --locked dms run` and open the local dashboard.
2. Choose **Start webcam**. Grant camera permission to the launching terminal or app.
3. Keep your face visible, frontal, and evenly lit. Unavailable readings are not a healthy-driver result.
4. Open **Setup → Start calibration**. Accumulate 20 valid frontal-face seconds before the 60-second deadline. Poor frames pause valid accumulation; this is not necessarily 20 consecutive wall-clock seconds.
5. Review the completion status. The baseline is stored locally for the default profile. It changes the EAR threshold, not the application's demonstrated accuracy.
6. Use **Stop session**, then inspect History and a report.

On macOS, camera and microphone access are controlled under **System Settings → Privacy & Security**. Check the permission for the terminal/IDE that actually launches Python. Camera use is explicit; microphone use begins only through **Record locally · 5 sec**. Close other camera applications if the device is busy.

## Audio and voice

Enable local audio in Settings while no session is active, save, and use Setup's audio test. The test queues speech; check that you hear it and that health reports no audio error. A successful queue response alone does not prove output.

Optional voice captures five seconds at 16 kHz from the default input, transcribes English locally on CPU, and returns the transcript and answer to the local browser. No continuous listener is implemented. Typed local questions remain available if permissions or dependencies fail.

## Updating an existing checkout

Stop the server before updating code or dependencies. Back up local data first; do not overwrite `.env` or delete the inherited data directory. Synchronize the desired extras with `uv sync --locked`, run the readiness check, then start replay before another webcam demonstration. See [backup and restore](privacy.md#backup-and-restore).
