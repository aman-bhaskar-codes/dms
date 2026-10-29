import asyncio
import time
from core.capture import LatestProcessor
from core.config import Settings
from core.runtime import Runtime
from ui.web.server import create_app
from fastapi.testclient import TestClient


async def test_latest_frame_buffer_is_bounded_and_shutdown_waits():
    seen = []

    def process(frame):
        time.sleep(0.02)
        seen.append(frame)
        return frame

    processor = LatestProcessor(process)
    task = asyncio.create_task(processor.run())
    for i in range(100):
        processor.submit(i)
    assert processor.queue.qsize() == 1
    assert await asyncio.wait_for(processor.results.get(), 1) == 99
    await processor.close()
    await task
    assert seen == [99]
    assert processor.dropped == 99


async def test_runtime_replay_records_and_stops_cleanly(tmp_path):
    config = Settings(data_dir=tmp_path, audio_enabled=False)
    runtime = Runtime(config)
    await runtime.open()
    await runtime.start("replay")
    await asyncio.sleep(0.25)
    assert runtime.snapshot.synthetic
    assert runtime.snapshot.state == "monitoring"
    assert not runtime.component_errors
    sid = runtime.session_id
    await runtime.stop()
    report = await runtime.store.report(sid)
    assert report["session"]["end"] is not None
    assert report["summary"]["samples"] >= 1
    assert runtime.session_id is None
    await runtime.close()


def test_api_session_report_origin_and_private_input(tmp_path):
    config = Settings(data_dir=tmp_path, audio_enabled=False)
    with TestClient(create_app(config)) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.post("/api/v1/sessions", json={"mode": "replay"}).status_code == 200
        assert client.post("/api/v1/sessions", json={"mode": "replay"}).status_code == 409
        assert (
            client.post(
                "/api/v1/assistant", json={"mode": "general", "question": "my pulse"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/sessions/stop", headers={"origin": "https://evil.example"}
            ).status_code
            == 403
        )
        assert client.post("/api/v1/sessions/stop").status_code == 200
        sid = client.get("/api/v1/sessions").json()[0]["id"]
        assert client.get(f"/api/v1/sessions/{sid}/report").status_code == 200
        assert (
            client.get(f"/api/v1/sessions/{sid}/export?format=csv")
            .headers["content-type"]
            .startswith("text/csv")
        )
        assert client.delete(f"/api/v1/sessions/{sid}").status_code == 200
        assert client.get(f"/api/v1/sessions/{sid}/report").status_code == 404


async def test_missing_model_marks_live_session_unavailable(tmp_path):
    runtime = Runtime(
        Settings(data_dir=tmp_path, model_path=tmp_path / "missing.task", audio_enabled=False)
    )
    await runtime.open()
    try:
        await runtime.start("live")
        for _ in range(40):
            if runtime.snapshot.state == "unavailable":
                break
            await asyncio.sleep(0.025)
        assert runtime.snapshot.state == "unavailable"
        assert runtime.snapshot.score is None
        assert runtime.component_errors
    finally:
        await runtime.close()


async def test_slow_assistant_does_not_delay_critical_alert(tmp_path):
    runtime = Runtime(Settings(data_dir=tmp_path, audio_enabled=False))
    await runtime.open()
    sid = await runtime.start("replay")
    runtime.session_task.cancel()
    await asyncio.gather(runtime.session_task, return_exceptions=True)
    runtime.session_task = None
    from ai.assistant import AssistantRequest

    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowProvider:
        async def generate(self, prompt):
            entered.set()
            await release.wait()

    runtime.assistant.provider = SlowProvider()
    assistant_task = asyncio.create_task(runtime.assistant.answer(AssistantRequest(mode="general")))
    await asyncio.wait_for(entered.wait(), 1)
    try:
        now = time.time()
        await runtime.ingest(
            dict(
                frame_id=1000,
                captured_at=now,
                face_detected=True,
                ear=0.1,
                mar=0.2,
                pitch=0,
                yaw=0,
                synthetic=True,
            )
        )
        started = time.perf_counter()
        await runtime.ingest(
            dict(
                frame_id=1001,
                captured_at=now + 1.6,
                face_detected=True,
                ear=0.1,
                mar=0.2,
                pitch=0,
                yaw=0,
                synthetic=True,
            )
        )
        assert time.perf_counter() - started < 0.1
        assert runtime.last_alert.severity == "critical"
        await runtime.stop()
        report = await runtime.store.report(sid)
        assert report["summary"]["alert_count"] == 1
    finally:
        assistant_task.cancel()
        await asyncio.gather(assistant_task, return_exceptions=True)
        await runtime.close()


async def test_second_runtime_cannot_interrupt_active_session(tmp_path):
    first = Runtime(Settings(data_dir=tmp_path, audio_enabled=False))
    second = Runtime(Settings(data_dir=tmp_path, audio_enabled=False))
    await first.open()
    sid = await first.start("replay")
    try:
        import pytest

        with pytest.raises(RuntimeError, match="already"):
            await second.open()
        assert (await first.store.report(sid))["session"]["end"] is None
    finally:
        await first.close()
        await second.close()


def test_audio_check_does_not_claim_success_when_audio_disabled(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, audio_enabled=False))) as client:
        response = client.post("/api/v1/audio/test")
        assert response.status_code == 409
        assert "disabled" in response.json()["detail"].lower()


async def test_idle_preview_exits_when_browser_disconnects(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, audio_enabled=False))
    endpoint = next(
        route.endpoint for route in app.routes if getattr(route, "path", None) == "/ws/v1/preview"
    )

    class ClosedBrowser:
        headers = {"origin": "http://127.0.0.1:8000"}
        closed = False

        async def accept(self):
            pass

        async def receive(self):
            return {"type": "websocket.disconnect", "code": 1000}

        async def close(self, code=1000):
            self.closed = True

    browser = ClosedBrowser()
    await asyncio.wait_for(endpoint(browser), 0.2)
    assert browser.closed


async def test_optional_dense_loading_does_not_delay_core_startup(tmp_path, monkeypatch):
    import time
    from rag.knowledge import KnowledgeBase

    model = tmp_path / "present-model"
    model.mkdir()

    def slow_load(self):
        time.sleep(0.25)
        self.status = "hybrid"

    monkeypatch.setattr(KnowledgeBase, "enable_dense", slow_load)
    runtime = Runtime(Settings(data_dir=tmp_path, embedding_path=model, audio_enabled=False))
    try:
        started = time.monotonic()
        await runtime.open()
        assert time.monotonic() - started < 0.15
        assert runtime.audio.task is not None
    finally:
        await runtime.close()


def test_assistant_status_and_reload_never_expose_key(tmp_path, monkeypatch):
    import ui.web.server as server

    fresh = Settings(
        data_dir=tmp_path, audio_enabled=False, gemini_api_key="private-test-placeholder"
    )
    with TestClient(
        create_app(Settings(data_dir=tmp_path, audio_enabled=False, gemini_api_key=""))
    ) as client:
        assert client.get("/api/v1/assistant/status").json()["configured"] is False
        monkeypatch.setattr(server, "Settings", lambda: fresh)
        result = client.post("/api/v1/assistant/reload")
        assert result.status_code == 200
        assert result.json()["configured"] is True
        assert "private-test-placeholder" not in result.text
        assert client.get("/api/v1/health").json()["session_id"] is None


async def test_concurrent_assistant_reloads_close_each_instance_once(tmp_path, monkeypatch):
    import ui.web.server as server

    app = create_app(Settings(data_dir=tmp_path, audio_enabled=False))
    runtime = app.state.runtime
    await runtime.open()
    original = runtime.assistant
    original_close = original.close
    calls = 0

    async def slow_close():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.03)
        await original_close()

    original.close = slow_close
    fresh = Settings(data_dir=tmp_path, audio_enabled=False, gemini_api_key="")
    monkeypatch.setattr(server, "Settings", lambda: fresh)
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/v1/assistant/reload"
    )
    try:
        await asyncio.gather(endpoint(), endpoint())
        assert calls == 1
    finally:
        await runtime.close()
