from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import time
import asyncio

app = FastAPI(title="DMS V3 API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock in-memory state for demonstration
live_state = {
    "fatigue_score": 0.0,
    "fatigue_level": "normal",
    "ear": 0.35,
    "perclos": 0.0,
    "hr_bpm": 70.0,
    "active_alerts": [],
    "agent_status": "Idle",
    "fatigue_prediction": 0.0
}

@app.get("/")
def read_root():
    return {"status": "ok", "service": "DMS V3 API"}

@app.get("/api/live/metrics")
def get_live_metrics():
    return live_state

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send current state
            payload = {
                'timestamp': time.time(),
                **live_state
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)  # 1Hz live metrics stream
    except WebSocketDisconnect:
        print("[API] WebSocket client disconnected")
        
@app.get("/api/sessions")
def list_sessions():
    return {"sessions": []}

@app.get("/api/fleet/overview")
def get_fleet_overview():
    return {
        "active_drivers": 1,
        "critical_alerts": 0,
        "average_fatigue": live_state["fatigue_score"]
    }
