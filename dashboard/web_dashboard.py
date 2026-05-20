"""
FastAPI WebSocket Dashboard for DMS V4.
Serves the HTML UI and streams metrics at 30Hz.
Replaces PyQt6 for cross-platform, zero-install viewing.
"""
import asyncio
import json
import cv2
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from config import settings


app = FastAPI(title="DMS V4 Dashboard")

# Global state to share between main camera thread and FastAPI
app.state.latest_frame = None
app.state.latest_metrics = {}
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
            metrics = app.state.latest_metrics
            frame = app.state.latest_frame
            
            payload = {"metrics": metrics, "image": None}
            
            if frame is not None:
                # Convert frame to JPEG base64 for web stream
                # Encode at lower quality (50) for fast streaming
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                payload["image"] = f"data:image/jpeg;base64,{jpg_as_text}"
                
            await websocket.send_text(json.dumps(payload))
            
    except WebSocketDisconnect:
        print("[Dashboard] Client disconnected")
    except Exception as e:
        print(f"[Dashboard] Error: {e}")


def start_server():
    """Runs Uvicorn in the current thread. Call this via asyncio or a thread."""
    import uvicorn
    uvicorn.run(app, host=settings.dashboard_host, port=settings.dashboard_port, log_level="warning")
