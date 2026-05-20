import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path

from core.bus import EventBus, EventTopic

logger = logging.getLogger("dms.ui.server")


class WebServer:
    def __init__(self, bus: EventBus, host: str = "0.0.0.0", port: int = 8000):
        self.bus = bus
        self.host = host
        self.port = port
        self.app = FastAPI(title="DMS V5 SENTINEL")
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Serve static files
        static_dir = Path(__file__).parent / "static"
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        # Active websocket connections
        self.active_connections: list[WebSocket] = []

        # Setup routes
        self._setup_routes()

        # Subscribe to bus events
        self.bus.subscribe(EventTopic.FATIGUE_SCORE, self._broadcast_telemetry)
        self.bus.subscribe(EventTopic.FATIGUE_CRITICAL, self._broadcast_critical)
        self.bus.subscribe(EventTopic.SIGNAL_HEAD_POSE, self._broadcast_head_pose)
        self.bus.subscribe(EventTopic.UI_OVERLAY_UPDATE, self._broadcast_overlay)
        
        self.server = None

    def _setup_routes(self):
        @self.app.get("/")
        async def root():
            # Redirect to static index
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/static/index.html")

        @self.app.websocket("/ws/telemetry")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    # Keep connection alive
                    data = await websocket.receive_text()
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"[WebServer] WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

    async def _broadcast(self, topic: str, payload: dict):
        message = json.dumps({"topic": topic, "payload": payload})
        stale_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                stale_connections.append(connection)
                
        for stale in stale_connections:
            if stale in self.active_connections:
                self.active_connections.remove(stale)

    async def _broadcast_telemetry(self, payload: dict):
        # Throttle telemetry if needed, but for now push directly
        await self._broadcast(EventTopic.FATIGUE_SCORE.name, payload)

    async def _broadcast_critical(self, payload: dict):
        await self._broadcast(EventTopic.FATIGUE_CRITICAL.name, payload)
        
    async def _broadcast_head_pose(self, payload: dict):
        await self._broadcast(EventTopic.SIGNAL_HEAD_POSE.name, payload)

    async def _broadcast_overlay(self, payload: dict):
        await self._broadcast(EventTopic.UI_OVERLAY_UPDATE.name, payload)

    async def run(self):
        config = uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        logger.info(f"[WebServer] Starting on http://{self.host}:{self.port}")
        await self.server.serve()

    async def shutdown(self):
        if self.server:
            self.server.should_exit = True
        logger.info("[WebServer] Shutdown requested.")
