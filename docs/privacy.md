# Privacy, storage, and security boundaries

[Configuration](configuration.md) · [Assistant](assistant.md) · [Architecture](architecture.md)

DMS is designed for one trusted user on one machine. It is not a hardened multi-user service. The boundaries here describe the application code, not a guarantee about the operating system, browser extensions, device drivers, or upstream service policies.

## Where data goes

| Data                              | Handling                                                                                                       |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Webcam frames / annotated preview | In process memory and a local loopback preview socket to the browser; not recorded to disk by the application. |
| Microphone audio                  | Five-second in-memory buffer on explicit request; local transcription.                                         |
| Voice transcript                  | Returned to the local browser and processed locally; no transcript-history table.                              |
| Metrics and alerts                | SQLite records for the local profile; exportable and potentially sensitive.                                    |
| Calibration                       | Baseline/threshold saved in SQLite for the default profile.                                                    |
| UI preferences                    | JSON in the data directory; excludes the API key.                                                              |
| Gemini key                        | Backend environment or ignored `.env`; no key-returning API.                                                   |
| Cloud context                     | Only curated questions/artificial scenarios and approved excerpts.                                             |
| Installed models                  | Downloaded explicitly from their upstream sources; stored on local disk.                                       |

“No raw media persistence” is the application's default behavior, not an assurance that screen recording, OS crash dumps, browser memory, or manually exported data cannot contain sensitive information. Exported JSON/CSV/HTML leave the application's retention/deletion controls once copied elsewhere.

## Local state layout

```text
data/sentinel/
├── sentinel.db          # sessions, metrics, events, calibration, usage
├── sentinel.db-wal      # SQLite sidecar while applicable
├── sentinel.db-shm      # SQLite sidecar while applicable
├── settings.json        # saved UI preferences
├── .runtime.lock        # exclusive instance lock
└── knowledge-index/     # optional dense index
```

The inherited directory/table/index names are preserved to avoid losing existing data. The database uses WAL mode and foreign-key cascades. No application-level encryption is implemented. Protect the OS account and device storage appropriately.

## Retention and deletion

Completed sessions older than `RETENTION_DAYS` expire at startup and during the daily maintenance check. Deleting a completed session removes its metric and event records. Active-session deletion is rejected. Only the most recent 200 sessions are returned by the list endpoint, so the list is not an unlimited archive browser.

Calibration records, usage counters, model files, exported files, and backups are not deleted by session retention. Deletion is logical SQLite deletion, not certified secure erasure of database pages or independent backups. There is no dedicated UI for calibration deletion or a full-account wipe.

## Backup and restore

1. Stop the application with Ctrl+C and wait for shutdown.
2. Copy the **entire data directory** to a private backup location. Preserve SQLite sidecars if any remain; do not copy just the main file while a writer is active.
3. Back up `.env` separately with restricted access if needed. Never include it in a public archive or Git commit.
4. To restore, stop DMS, preserve the current directory as a fallback, then restore the saved directory to the configured `DATA_DIR`.
5. Restart and inspect History, settings, and quota state before recording another session.

Before an older schema is upgraded, the store uses SQLite's backup API to create a `.pre-v1.bak` backup including committed WAL data. An unsupported newer schema is refused. Do not manually edit schema numbers to force compatibility.

The legacy import command reads supported session/event fields from another database without modifying that source. It is not a backup-restore command and does not import all historical metrics:

```sh
uv run --locked dms import-legacy /absolute/path/to/old.db
```

## Browser and local API boundary

The server restricts binding and allowed Hosts to loopback. Mutating HTTP requests reject unapproved Origin headers, WebSockets require approved origins, bodies are bounded, and responses use a Content Security Policy restricted to local assets. Assistant output and excerpts are rendered without arbitrary generated HTML.

There is no user login, CSRF token for other local clients, operating-system permission sandbox, per-user network authentication, or TLS reverse-proxy deployment. A local process can make requests without a browser Origin. Do not expose the port to a LAN, public tunnel, or shared hosting and assume these controls provide authentication.

## External services

Dependencies and explicit model downloads require network access to their package/model hosts. Optional Gemini requests go to Google under that project's terms. `store=False` disables stored Interactions state for the request; it is not a replacement for reviewing provider data handling. Chroma telemetry is disabled in the local client configuration.

There is no automatic Telegram messaging, cloud database, or raw-media upload feature in the active application. Adding new integrations requires an explicit review of their payloads and consent boundaries.

## Sharing diagnostics

Report the command, OS/Python, readiness status, error category, and reproduction steps. Review health snapshots, exports, paths, and screenshots before sharing: they can contain session IDs, measurements, or transcripts. Never include `.env`, tokens, full private databases, or recordings. Git ignore rules reduce accidental tracking; they are not a secret scanner or access-control system.
