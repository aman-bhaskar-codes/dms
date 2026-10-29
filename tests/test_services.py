import pytest
from core.contracts import MonitoringSnapshot
from memory.store import Store
from ai.assistant import AssistantService, AssistantRequest
from rag.knowledge import KnowledgeBase


async def test_session_report_export_delete_and_restart(tmp_path):
    path = tmp_path / "test.db"
    store = Store(path)
    await store.connect()
    session = await store.start_session("default", "replay")
    s = MonitoringSnapshot(
        session_id=session, frame_id=1, captured_at=100, state="monitoring", score=30, level="safe"
    )
    await store.record_snapshot(s)
    await store.end_session(session)
    report = await store.report(session)
    assert report["summary"]["samples"] == 1
    assert report["summary"]["tracking_coverage"] == 1
    assert "image" not in str(report)
    await store.close()
    await store.connect()
    assert len(await store.sessions()) == 1
    await store.delete_session(session)
    assert await store.sessions() == []
    await store.close()


async def test_missing_key_and_private_input_never_reach_cloud(tmp_path):
    store = Store(tmp_path / "test.db")
    await store.connect()
    kb = KnowledgeBase(tmp_path / "knowledge")
    kb.ingest()
    service = AssistantService(store, kb, api_key="")
    response = await service.answer(AssistantRequest(mode="general", topic="fatigue"))
    assert response.source == "local"
    assert response.limitation
    with pytest.raises(ValueError):
        AssistantRequest(mode="general", question="my pulse is 90")
    await store.close()


async def test_persistent_daily_quota(tmp_path):
    store = Store(tmp_path / "test.db")
    await store.connect()
    assert await store.reserve_usage("2000-01-01", 1, 0, 100)
    assert not await store.reserve_usage("2000-01-01", 1, 0, 200)
    await store.close()
    await store.connect()
    assert not await store.reserve_usage("2000-01-01", 1, 0, 300)
    await store.close()


def test_knowledge_citations_and_changed_document_reindex(tmp_path):
    (tmp_path / "rest.md").write_text(
        "---\ntitle: Rest\nsource: local-demo\nrevision: 2026-10-29\n---\n# Fatigue\nFatigue requires a safe rest stop."
    )
    kb = KnowledgeBase(tmp_path)
    assert kb.ingest() == 1
    assert kb.ingest() == 0
    results = kb.search("fatigue")
    assert results and results[0]["checksum"]
    assert kb.document(results[0]["id"])["content"]
    assert kb.search("quantum banana") == []


async def test_gemini_citation_validation_retry_and_private_boundary(tmp_path):
    import json

    (tmp_path / "guide.md").write_text(
        "Fatigue and breaks require rest. Calibration and limitations affect webcam monitoring."
    )
    store = Store(tmp_path / "test.db")
    await store.connect()
    kb = KnowledgeBase(tmp_path)
    kb.ingest()

    class Provider:
        def __init__(self):
            self.prompts = []

        async def generate(self, prompt):
            self.prompts.append(prompt)
            content = json.loads(prompt.split("\n", 1)[1])
            return {"answer": "Rest at a safe place.", "citations": [content["excerpts"][0]["id"]]}

    provider = Provider()
    service = AssistantService(store, kb, provider=provider)
    result = await service.answer(AssistantRequest(mode="general", topic="fatigue"))
    assert result.source == "gemini" and result.citations
    assert len(provider.prompts) == 1
    assert "session_id" not in provider.prompts[0]
    assert "captured_at" not in provider.prompts[0]
    assert "measurements" not in provider.prompts[0]
    limited = await service.answer(AssistantRequest(mode="general", topic="fatigue"))
    assert limited.source == "local" and "limit" in limited.answer
    await service.close()
    await store.close()


async def test_invalid_provider_citations_abstain(tmp_path):
    (tmp_path / "guide.md").write_text("Fatigue requires rest.")
    store = Store(tmp_path / "test.db")
    await store.connect()
    kb = KnowledgeBase(tmp_path)
    kb.ingest()

    class Provider:
        async def generate(self, prompt):
            return {"answer": "A fabricated fact", "citations": ["invented"]}

    service = AssistantService(store, kb, provider=Provider())
    response = await service.answer(AssistantRequest(mode="general"))
    assert response.source == "local"
    assert response.limitation and "fabricated" not in response.answer
    await service.close()
    await store.close()


async def test_future_database_version_is_not_downgraded(tmp_path):
    import sqlite3

    path = tmp_path / "future.db"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=999")
    store = Store(path)
    with pytest.raises(ValueError, match="newer"):
        await store.connect()
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 999


async def test_record_batch_is_atomic_and_preserves_alerts(tmp_path):
    from core.contracts import AlertEvent

    store = Store(tmp_path / "batch.db")
    await store.connect()
    sid = await store.start_session()
    snapshot = MonitoringSnapshot(
        session_id=sid, frame_id=1, captured_at=1, state="monitoring", score=0
    )
    alert = AlertEvent(session_id=sid, severity="critical", cause="test", transition="triggered")
    await store.record_batch([("snapshot", snapshot), ("event", (sid, alert, "alert"))])
    await store.end_session(sid)
    report = await store.report(sid)
    assert len(report["metrics"]) == 1
    assert report["summary"]["alert_count"] == 1
    await store.close()


async def test_import_real_v4_integer_session_ids(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as source:
        source.execute(
            "CREATE TABLE sessions(id INTEGER PRIMARY KEY,driver_id TEXT,start_time REAL,end_time REAL)"
        )
        source.execute("INSERT INTO sessions VALUES(1,'driver',100,200)")
    store = Store(tmp_path / "target.db")
    await store.connect()
    try:
        assert await store.import_legacy(path) == 1
        sessions = await store.sessions()
        assert sessions[0]["id"] == "legacy-1"
    finally:
        await store.close()


async def test_retry_never_bypasses_daily_attempt_cap(tmp_path):
    from datetime import datetime, timezone

    (tmp_path / "guide.md").write_text("Fatigue requires rest.")
    store = Store(tmp_path / "test.db")
    await store.connect()
    kb = KnowledgeBase(tmp_path)
    kb.ingest()

    class Transient(Exception):
        code = 503

    class Provider:
        calls = 0

        async def generate(self, prompt):
            self.calls += 1
            raise Transient()

    provider = Provider()
    service = AssistantService(store, kb, provider=provider, daily_cap=1)
    try:
        response = await service.answer(AssistantRequest(mode="general"))
        assert response.source == "local"
        assert provider.calls == 1
        assert await store.usage(datetime.now(timezone.utc).date().isoformat()) == 1
    finally:
        await service.close()
        await store.close()


async def test_legacy_import_preserves_event_timeline_without_raw_media(tmp_path):
    import sqlite3
    import hashlib

    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as source:
        source.execute(
            "CREATE TABLE sessions(id INTEGER PRIMARY KEY,driver_id TEXT,start_time REAL,end_time REAL)"
        )
        source.execute("INSERT INTO sessions VALUES(1,'driver',100,200)")
        source.execute(
            "CREATE TABLE events(id INTEGER PRIMARY KEY,session_id INTEGER,timestamp REAL,event_type TEXT,severity TEXT,fatigue_score REAL,details TEXT)"
        )
        source.execute(
            "INSERT INTO events VALUES(1,1,150,'eye_closure','critical',80,?)",
            ('{"image":"private-frame","transcript":"private words"}',),
        )
    before = hashlib.sha256(legacy.read_bytes()).hexdigest()
    store = Store(tmp_path / "new.db")
    await store.connect()
    try:
        await store.import_legacy(legacy)
        report = await store.report("legacy-1")
        assert report["summary"]["alert_count"] == 1
        assert report["events"][0]["cause"] == "eye_closure"
        assert "private-frame" not in str(report) and "private words" not in str(report)
        assert hashlib.sha256(legacy.read_bytes()).hexdigest() == before
    finally:
        await store.close()


@pytest.mark.parametrize(
    "model,level", [("gemini-3.8-flash", "LOW"), ("gemini-3.5-flash-lite", "MINIMAL")]
)
async def test_gemini3_request_uses_low_reasoning_for_bounded_latency(model, level):
    import pytest

    pytest.importorskip("google.genai")
    from types import SimpleNamespace
    from ai.assistant import GeminiProvider

    observed = {}

    class Interactions:
        async def create(self, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(
                output_text='{"answer":"General rest guidance.","citations":["source"]}'
            )

    provider = GeminiProvider("test-placeholder-not-a-key", model)
    provider.client = SimpleNamespace(aio=SimpleNamespace(interactions=Interactions()))
    result = await provider.generate("General safety question only")
    assert result["answer"] == "General rest guidance."
    assert observed["generation_config"]["thinking_level"].upper() == level
    assert "temperature" not in observed["generation_config"]
    assert observed["store"] is False
    assert observed["service_tier"] == "standard"
    assert observed["generation_config"]["max_output_tokens"] == 512


async def test_cloud_outage_returns_cited_local_guidance(tmp_path):
    (tmp_path / "guide.md").write_text(
        "---\ntitle: Approved rest guide\nsource: local-approved\nrevision: 2026-10-29\n---\nFatigue requires a safe rest stop. Adequate sleep matters before driving."
    )
    store = Store(tmp_path / "fallback.db")
    await store.connect()
    kb = KnowledgeBase(tmp_path)
    kb.ingest()

    class Busy(Exception):
        code = 503

    class Provider:
        async def generate(self, prompt):
            raise Busy()

    service = AssistantService(store, kb, provider=Provider(), daily_cap=1)
    try:
        result = await service.answer(AssistantRequest(mode="general", topic="fatigue"))
        assert result.source == "local"
        assert "safe rest stop" in result.answer
        assert result.citations
        assert "Gemini" in result.limitation
    finally:
        await service.close()
        await store.close()


async def test_local_question_retrieves_evidence_without_cloud(tmp_path):
    (tmp_path / "guide.md").write_text(
        "Calibration needs even lighting and a frontal face for twenty seconds."
    )
    store = Store(tmp_path / "local.db")
    await store.connect()
    kb = KnowledgeBase(tmp_path)
    kb.ingest()
    service = AssistantService(store, kb, api_key="")
    try:
        result = await service.answer_question("How does calibration use lighting?")
        assert result.source == "local" and result.citations
        assert "lighting" in result.answer
        assert await store.usage("2000-01-01") == 0
    finally:
        await service.close()
        await store.close()


def test_unrelated_question_does_not_match_only_stopwords(tmp_path):
    (tmp_path / "guide.md").write_text("The calibration guide explains the camera and the driver.")
    kb = KnowledgeBase(tmp_path)
    kb.ingest()
    assert kb.search("What is the capital of France?") == []
