# Contributing

[Architecture](architecture.md) · [Components](components.md) · [Verification](verification.md)

Start from the current `main` and use a focused feature, fix, or documentation branch. Keep pull requests reviewable; separate runtime changes from unrelated cleanup. The repository currently has no declared top-level license grant—do not assume a public repository alone licenses redistribution.

## Development environment

```sh
uv python install 3.12
uv sync --locked --extra gemini
uv run --locked dms --help
```

Add `knowledge`/`voice` extras only when needed. Run their explicit setup commands for local models. Do not require downloaded optional models or credentials for deterministic CI tests.

## Standard checks

```sh
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
```

mypy is intentionally focused on the files configured in `pyproject.toml`; a passing result is not a whole-repository type guarantee. MediaPipe's native smoke check needs normal local process permissions. Do not skip a regression or weaken CSP just to accommodate a restricted automation environment.

### Browser workflow without private data

Install Chromium once:

```sh
uv run --locked playwright install chromium
```

Terminal 1, macOS/Linux:

```sh
TEST_DATA=$(mktemp -d)
DATA_DIR="$TEST_DATA" GEMINI_API_KEY='' AUDIO_ENABLED=false \
  uv run --locked dms run
```

Terminal 2, from the checkout:

```sh
uv run --locked python scripts/browser_check.py http://127.0.0.1:8000
```

Stop Terminal 1 with Ctrl+C afterward. Keep/delete the temporary directory intentionally. The script starts/stops/deletes a replay and changes/restores preferences: never point it at a valuable live user session. A screenshot is written only with `--save-screenshot`; review it before committing. A configured key is unnecessary for this workflow.

CI defines separate Ubuntu Python and Chromium jobs, installs native libraries, runs CLI checks, and uses temporary browser data. Confirm actual check results on the pull request rather than inferring success from the workflow definition.

## Change requirements

- Trace affected code through capture, monitor, policy, persistence, and UI before changing an interface.
- Keep deterministic alerts independent of optional cloud, retrieval, and voice work.
- Preserve null/validity semantics; never fabricate normal values for missing measurements.
- Add meaningful regression tests for changed behavior. Mock cloud responses and verify no real driver data enters prompts.
- Preserve queue bounds, resource cleanup, storage compatibility, and existing user data.
- Keep secrets out of source, fixtures, screenshots, and logs. Raw media should stay in memory and the local preview path by default.
- Render generated answers/excerpts as text. Preserve origin/body limits and CSP.
- Document added configuration, routes, dependencies, failure behavior, and observable limitations.

## Documentation standards

Use **DMS** as the product name without release branding. Keep technical identifiers such as dependency pins, model revisions, schema numbers, and API route versions when they are needed for reproducibility or compatibility.

The README is the entry point; detailed truth belongs in the relevant reference page. Link instead of duplicating long configuration/API tables. Distinguish implemented behavior, historical observation, target, and unverified acceptance. Do not describe heuristics as detection accuracy or a short synthetic test as a physical soak.

Update `SUMMARY.md` when adding a public guide. It is an import-ready navigation outline for GitBook; it does not create a site or establish Git Sync. Keep the approved assistant corpus separate from general engineering docs so new operational text is not silently supplied to Gemini.

## Review and integration

Use descriptive commit messages, for example `docs: clarify calibration and report semantics` or `fix: release preview socket on disconnect`. Explain the user-visible result, validation, compatibility, and remaining limitations in the PR. Squash merge only after required checks and review; do not bypass branch protection. No release tag is needed for a documentation update.
