# DMS documentation

DMS combines local webcam geometry, deterministic monitoring alerts, SQLite history, and optional grounded assistance. This documentation follows the current source checkout; it distinguishes implemented behavior from unverified hardware acceptance.

## Start using DMS

1. [Project overview](../README.md) — features, architecture, stack, quick start.
2. [Setup](setup.md) — clean installation, platform support, permissions, optional models.
3. [Configuration](configuration.md) — all settings, defaults, precedence, and reload behavior.
4. [Guided demonstration](demo.md) — repeatable synthetic replay and optional stationary calibration.
5. [Troubleshooting](troubleshooting.md) — symptoms, likely causes, and recovery steps.

## Understand the system

- [Architecture](architecture.md) — data flow, queues, lifecycle, and failure boundaries.
- [Component reference](components.md) — responsibilities, score formula, calibration, alerts, reports, and tools.
- [Local API](api.md) — routes, contracts, requests, WebSockets, and error semantics.
- [Assistant and retrieval](assistant.md) — local/cloud modes, citations, quotas, model loading, voice.
- [Privacy and storage](privacy.md) — data inventory, retention, deletion, backup, and local security.

## Evaluate and contribute

- [Verification](verification.md) — evidence, benchmark definitions, and remaining acceptance checks.
- [Contributing](contributing.md) — isolated testing, review, and documentation conventions.
- [Design specification](design/implementation.md) — Figma reference, tokens, and interaction behavior.
- [Design contract](../DESIGN.md) · [Interaction contract](../UX-CONTRACT.md)
- [Model manifest](../models/manifest.json) · [CI workflow](../.github/workflows/ci.yml)

## Documentation publishing

The repository is the source of truth. Root [SUMMARY.md](../SUMMARY.md) provides an import-ready page order for GitBook. No hosted GitBook site or Git Sync connection is implied. Import the existing Git documentation rather than creating a second independent content tree; review links/assets in the target space before publication. Git Sync connection is configured through GitBook's UI.
