"""Keep unit/API tests independent of installed optional models and real credentials."""

import pytest


@pytest.fixture(autouse=True)
def isolate_optional_assets_and_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_PATH", str(tmp_path / "not-installed-embeddings"))
    monkeypatch.setenv("WHISPER_PATH", str(tmp_path / "not-installed-whisper"))
