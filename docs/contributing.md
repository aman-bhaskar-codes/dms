# Contributing to DMS

Thank you for contributing to the DMS driver-monitoring project. This guide outlines the development workflow, quality checks, branch conventions, and security requirements.

---

## Local Development Environment

1. Ensure Python 3.12 and `uv` are installed.
2. Clone and install all development dependencies:
   ```sh
   git clone https://github.com/aman-bhaskar-codes/dms.git
   cd dms
   uv sync --locked --extra gemini --extra knowledge --extra voice
   ```
3. Set up pre-commit or run tests manually before pushing.

---

## Code Quality Standards

All pull requests must pass the automated verification suite before review:

```sh
# 1. Run unit and integration tests
uv run pytest -q

# 2. Check code style and formatting
uv run ruff check .
uv run ruff format --check .

# 3. Check static typing
uv run mypy

# 4. Run browser e2e checks (requires a running dms server)
uv run dms run &
SERVER_PID=$!
uv run python scripts/browser_check.py
kill $SERVER_PID
```

---

## Branch and Commit Conventions

- Create a feature or bugfix branch off `main` (e.g. `feat/feature-name` or `fix/issue-description`).
- Use Conventional Commits formatting:
  - `feat:` for new capabilities.
  - `fix:` for bug fixes.
  - `refactor:` for code restructuring without behavioral changes.
  - `docs:` for documentation additions and edits.
  - `ci:` for CI/CD pipeline changes.
- Keep commits focused and logically grouped.
- Pull requests are squash merged into `main` with descriptive titles.

---

## Security and Privacy Guidelines

1. **No Media Persistence:** Raw video frames and raw audio streams must never be written to disk, logged, or transmitted over the network.
2. **Cloud Boundary:** Never pass user telemetry, driver identifiers, webcam images, audio recordings, or free-form user input to the cloud Gemini API. Cloud requests are strictly limited to curated question identifiers and approved local documentation excerpts.
3. **Secrets Safety:** Never commit `.env` or API keys. Ensure secrets are loaded via environment variables or `.env` using Pydantic `SecretStr`.
4. **Origin Validation:** All WebSocket and state-mutating HTTP endpoints must enforce loopback origin checks to prevent Cross-Site WebSocket Hijacking (CSWSH) and unauthorized browser manipulation.
