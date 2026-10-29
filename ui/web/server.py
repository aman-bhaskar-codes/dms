"""Loopback-only HTTP API and independently throttled local WebSockets."""

import asyncio
import csv
import html
import io
import json
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict

from ai.assistant import AssistantRequest, AssistantService
from core.config import Settings
from core.runtime import Runtime


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["live", "replay"] = "replay"


class LocalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


def create_app(config=None):
    config = config or Settings()
    runtime = Runtime(config)
    assistant_reload_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        await runtime.open()
        if getattr(app.state, "autostart", None):
            await runtime.start(app.state.autostart)
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="DMS", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"]
    )
    origins = {f"http://{h}:{config.web_port}" for h in ("localhost", "127.0.0.1", "[::1]")}

    @app.middleware("http")
    async def boundaries(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin not in origins:
                return Response("Origin rejected", 403)
            if int(request.headers.get("content-length", "0") or 0) > 8192:
                return Response("Request too large", 413)
            # Read a bounded body including chunked requests.
            body = b""
            async for chunk in request.stream():
                body += chunk
                if len(body) > 8192:
                    return Response("Request too large", 413)
            request._body = body
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    async def root():
        return RedirectResponse("/static/index.html")

    @app.get("/api/v1/health")
    async def health():
        return runtime.health()

    @app.get("/api/v1/setup")
    async def setup():
        from core.setup import check_setup

        return await asyncio.to_thread(check_setup, runtime.config)

    @app.post("/api/v1/sessions")
    async def start(body: SessionRequest):
        try:
            return {"id": await runtime.start(body.mode)}
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/v1/sessions/stop")
    async def stop():
        await runtime.stop()
        return {"stopped": True}

    @app.post("/api/v1/alerts/acknowledge")
    async def acknowledge():
        await runtime.acknowledge()
        return {"acknowledged": True}

    @app.post("/api/v1/calibration")
    async def calibrate():
        if runtime.session_id is None or runtime.snapshot.synthetic:
            raise HTTPException(409, "Start a live session while stationary before calibration")
        runtime.monitor.calibration.start(time.time())
        runtime.calibration_saved = False
        return runtime.monitor.calibration.status()

    @app.get("/api/v1/sessions")
    async def sessions():
        return await runtime.store.sessions()

    @app.get("/api/v1/sessions/{sid}/report")
    async def report(sid: str):
        try:
            return await runtime.store.report(sid)
        except KeyError as error:
            raise HTTPException(404, "Session not found") from error

    @app.delete("/api/v1/sessions/{sid}")
    async def delete(sid: str):
        if sid == runtime.session_id:
            raise HTTPException(409, "Stop the session before deleting it")
        if not await runtime.store.delete_session(sid):
            raise HTTPException(404, "Session not found")
        return {"deleted": True}

    @app.get("/api/v1/sessions/{sid}/export")
    async def export(sid: str, format: Literal["json", "csv", "html"] = "json"):
        data = await report(sid)
        headers = {"Content-Disposition": f'attachment; filename="dms-session.{format}"'}
        if format == "json":
            return Response(json.dumps(data), media_type="application/json", headers=headers)
        if format == "csv":
            stream = io.StringIO()
            writer = csv.writer(stream)
            writer.writerow(
                ["captured_at", "state", "score", "ear", "perclos", "eye_closed_s", "yaw"]
            )
            for m in data["metrics"]:
                values = m["measurements"]
                writer.writerow(
                    [
                        m["captured_at"],
                        m["state"],
                        m["score"],
                        *[values.get(k) for k in ("ear", "perclos", "eye_closed_s", "yaw")],
                    ]
                )
            return Response(stream.getvalue(), media_type="text/csv", headers=headers)
        rows = "".join(
            f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>"
            for k, v in data["summary"].items()
        )
        timeline = "".join("<li>" + html.escape(str(e)) + "</li>" for e in data["events"])
        return HTMLResponse(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            "<title>DMS local report</title><h1>DMS session report</h1>"
            "<p>Local heuristic demonstration. No medical or predictive interpretation.</p>"
            "<dl>" + rows + "</dl><h2>Event timeline</h2><ul>" + timeline + "</ul></html>",
            headers=headers,
        )

    @app.get("/api/v1/assistant/status")
    async def assistant_status():
        day = datetime.now(timezone.utc).date().isoformat()
        return dict(
            configured=runtime.assistant.provider is not None,
            model=runtime.assistant.model,
            requests_today=await runtime.store.usage(day),
            daily_cap=runtime.assistant.daily_cap,
            interval_seconds=runtime.assistant.interval,
            knowledge=runtime.knowledge.status,
            last_source=runtime.assistant.last_source,
            last_limitation=runtime.assistant.last_limitation,
        )

    @app.post("/api/v1/assistant/reload")
    async def reload_assistant():
        async with assistant_reload_lock:
            if runtime.assistant.active or not runtime.assistant.queue.empty():
                raise HTTPException(
                    409, "Wait for the current assistant request before reloading settings"
                )
            try:
                fresh = Settings()
            except ValueError as error:
                raise HTTPException(422, "Check the backend environment configuration") from error
            previous = runtime.assistant
            await previous.close()
            runtime.assistant = AssistantService(
                runtime.store,
                runtime.knowledge,
                api_key=fresh.gemini_api_key.get_secret_value(),
                model=fresh.gemini_model,
                daily_cap=fresh.gemini_daily_cap,
                interval=fresh.gemini_interval_s,
            )
            runtime.config = runtime.config.model_copy(
                update={
                    key: getattr(fresh, key)
                    for key in (
                        "gemini_api_key",
                        "gemini_model",
                        "gemini_daily_cap",
                        "gemini_interval_s",
                    )
                }
            )
            return await assistant_status()

    @app.post("/api/v1/assistant")
    async def assistant(body: AssistantRequest):
        return await runtime.assistant.answer(body, runtime.snapshot)

    @app.post("/api/v1/voice/listen")
    async def listen():
        try:
            transcript = await runtime.voice.listen()
        except (ValueError, ImportError) as error:
            raise HTTPException(
                409, "Local voice is unavailable. Check model setup and microphone permission."
            ) from error
        except Exception as error:
            raise HTTPException(503, "Microphone capture or local transcription failed") from error
        response = await runtime.assistant.answer_question(transcript, runtime.snapshot)
        runtime.audio.speak(response.answer[:600])
        return {"transcript": transcript, **response.model_dump()}

    @app.post("/api/v1/voice/local")
    async def typed(body: LocalQuestion):
        if len(body.text) > 1000:
            raise HTTPException(422, "Keep local questions below 1000 characters")
        return await runtime.assistant.answer_question(body.text, runtime.snapshot)

    @app.post("/api/v1/audio/test")
    async def audio_test():
        runtime.audio.enabled = runtime.config.audio_enabled
        if not runtime.audio.enabled:
            raise HTTPException(
                409, "Audio is disabled. Enable local audio in Settings before testing."
            )
        runtime.audio.speak("DMS audio test. Local alerts are ready.")
        return {"queued": True}

    safe_settings = (
        "camera_index",
        "camera_width",
        "camera_height",
        "camera_fps",
        "audio_enabled",
        "fatigue_warn_threshold",
        "fatigue_crit_threshold",
        "eye_closure_s",
        "retention_days",
    )

    @app.get("/api/v1/settings")
    async def get_settings():
        return {k: getattr(runtime.config, k) for k in safe_settings}

    @app.patch("/api/v1/settings")
    async def update_settings(body: dict):
        if runtime.session_id:
            raise HTTPException(409, "Stop monitoring before changing settings")
        if set(body) - set(safe_settings):
            raise HTTPException(422, "Unknown setting")
        try:
            updated = Settings(**(runtime.config.model_dump() | body))
        except ValueError as error:
            raise HTTPException(422, "Invalid settings or threshold order") from error
        runtime.config = updated
        path = updated.data_dir / "settings.json"
        data = {k: getattr(updated, k) for k in safe_settings}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data))
        temporary.replace(path)
        return data

    @app.get("/api/v1/knowledge/{identity}")
    async def document(identity: str):
        try:
            return runtime.knowledge.document(identity)
        except KeyError as error:
            raise HTTPException(404, "Excerpt not found") from error

    async def stream(ws, preview=False, legacy=False):
        if ws.headers.get("origin") not in origins:
            await ws.close(code=1008)
            return
        await ws.accept()

        async def send_updates():
            last = -1
            while True:
                if preview:
                    if runtime.preview and last != runtime.preview_version:
                        last = runtime.preview_version
                        await asyncio.wait_for(ws.send_json({"image": runtime.preview}), timeout=1)
                    await asyncio.sleep(0.15)
                else:
                    payload = runtime.health()
                    if legacy:
                        payload = {"topic": "FATIGUE_SCORE", "payload": payload["snapshot"]}
                    await asyncio.wait_for(ws.send_json(payload), timeout=1)
                    await asyncio.sleep(0.2)

        async def watch_disconnect():
            # Idle preview may never send. Observe ASGI disconnects independently.
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return

        tasks = [asyncio.create_task(send_updates()), asyncio.create_task(watch_disconnect())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, RuntimeError, asyncio.TimeoutError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await ws.close()
            except RuntimeError:
                pass

    @app.websocket("/ws/v1/telemetry")
    async def telemetry(ws: WebSocket):
        await stream(ws)

    @app.websocket("/ws/v1/preview")
    async def preview(ws: WebSocket):
        await stream(ws, preview=True)

    @app.websocket("/ws/telemetry")
    async def legacy(ws: WebSocket):
        await stream(ws, legacy=True)

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app
