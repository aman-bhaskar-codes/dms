"""
FastAPI WebSocket Dashboard for DMS V4.
Serves the HTML UI and streams metrics at 30Hz.
Replaces PyQt6 for cross-platform, zero-install viewing.
"""
import asyncio
import json
import cv2
import threading
import base64
import uvicorn
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from config import settings


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


app = FastAPI(title="DMS V4 Dashboard")

# Global state to share between main camera thread and FastAPI
_metrics_lock = threading.Lock()
_latest_metrics: dict = {}
_latest_frame = None

def update_state_safe(metrics: dict, frame=None):
    """Thread-safe state update from camera loop."""
    with _metrics_lock:
        global _latest_metrics, _latest_frame
        _latest_metrics = metrics.copy()
        if frame is not None:
            _latest_frame = frame.copy()

def read_state_safe() -> tuple:
    """Thread-safe state read from WebSocket."""
    with _metrics_lock:
        metrics = _latest_metrics.copy()
        frame = _latest_frame.copy() if _latest_frame is not None else None
        return metrics, frame

app.state.trigger_calibrate = False

html_path = Path("static/dashboard.html")


@app.post("/api/calibrate")
async def trigger_calibration():
    app.state.trigger_calibrate = True
    return {"status": "success", "message": "Calibration triggered"}


@app.get("/")
async def get_dashboard():
    if not html_path.exists():
        return HTMLResponse("Dashboard HTML not found.", status_code=404)
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # We stream at ~30 FPS
            await asyncio.sleep(1.0 / settings.target_fps)
            
            # Send metrics
            metrics, frame = read_state_safe()
            
            payload = {"metrics": metrics, "image": None}
            
            if frame is not None:
                # Convert frame to JPEG base64 for web stream
                # Encode at lower quality (50) for fast streaming
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                payload["image"] = f"data:image/jpeg;base64,{jpg_as_text}"
                
            await websocket.send_text(json.dumps(payload, cls=NumpyEncoder))
            
    except WebSocketDisconnect:
        print("[Dashboard] Client disconnected")
    except Exception as e:
        print(f"[Dashboard] Error: {e}")


def start_server():
    """Runs Uvicorn in the current thread. Call this via asyncio or a thread."""
    import uvicorn
    uvicorn.run(app, host=settings.dashboard_host, port=settings.dashboard_port, log_level="warning")
