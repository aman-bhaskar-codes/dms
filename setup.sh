#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
uv sync --locked --extra gemini
uv run sentinel setup-models
uv run sentinel check
