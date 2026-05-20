"""
DMS V5 SENTINEL — The Ultimate Agentic Driving Orchestrator
"""
import asyncio
import logging
import signal
import sys
from pathlib import Path

from core.bus import EventBus, EventTopic
from core.config import settings

# Memory
from memory.working_memory import WorkingMemory
from memory.episodic import EpisodicMemory

# Perception
from perception.pipeline import FrameProcessor

# Agents
from agents.orchestrator import Orchestrator
from agents.safety_agent import SafetyAgent

# Alerts
from alerts.alert_engine_v5 import AlertEngineV5

# Voice (Phase 4)
try:
    from voice.voice_pipeline import VoicePipeline
except ImportError:
    VoicePipeline = None

from rag.rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("dms.main_v5")


async def camera_loop(perception: FrameProcessor):
    import cv2
    logger.info("[Camera] Initializing OpenCV video capture...")
    
    camera_idx = settings.camera_index
    cap = cv2.VideoCapture(camera_idx)
    
    # Check if opened, fallback if needed
    if not cap.isOpened():
        logger.warning(f"[Camera] Failed to open camera index {camera_idx}. Retrying index 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("[Camera] Could not open any camera device. Check macOS camera permissions under System Settings > Privacy & Security > Camera.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    fps = settings.camera_fps if hasattr(settings, "camera_fps") else 30
    frame_delay = 1.0 / fps
    
    logger.info(f"[Camera] OpenCV capture started on index {camera_idx} at {fps} FPS.")
    
    try:
        while True:
            t0 = asyncio.get_event_loop().time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("[Camera] Frame read failed. Retrying in 0.5s...")
                await asyncio.sleep(0.5)
                continue

            if settings.flip_horizontal:
                frame = cv2.flip(frame, 1)

            # Submit frame to the perception pipeline (non-blocking thread pool execution)
            await perception.submit(frame)

            # Dynamic sleep to maintain target frame-rate
            elapsed = asyncio.get_event_loop().time() - t0
            sleep_time = max(0.001, frame_delay - elapsed)
            await asyncio.sleep(sleep_time)
    except asyncio.CancelledError:
        logger.info("[Camera] Capture loop cancelled.")
    finally:
        cap.release()
        logger.info("[Camera] OpenCV capture released.")


async def main():
    logger.info("🚗⚡ DMS V5 SENTINEL starting...")

    # 1. Initialize Event Bus
    bus = EventBus()
    bus_task = asyncio.create_task(bus.run())

    # 2. Initialize Memory Tier
    working_memory = WorkingMemory()
    episodic_memory = EpisodicMemory()
    await episodic_memory.connect()
    
    # Generate new session ID
    import uuid
    session_id = str(uuid.uuid4())
    await episodic_memory.log_session_start(session_id, "driver_1")

    # 3. Initialize Perception
    perception = FrameProcessor(bus=bus, kalman_enabled=True)
    perception_task = asyncio.create_task(perception.run())

    # Start Camera capture loop background task
    camera_task = asyncio.create_task(camera_loop(perception))

    # 4. Initialize RAG Engine
    rag = RAGEngine(bus=bus)
    rag_task = asyncio.create_task(rag.run())

    # 5. Initialize Agents
    orchestrator = Orchestrator(bus=bus, memory=working_memory)
    safety_agent = SafetyAgent(bus=bus, memory=working_memory)
    
    # Register agents to the Orchestrator
    orchestrator.register_agent("safety", safety_agent)
    orchestrator.register_agent("rag", rag)
    orchestrator_task = asyncio.create_task(orchestrator.run())

    # 6. Initialize Alerts
    alert_engine = AlertEngineV5(bus=bus)
    alert_task = asyncio.create_task(alert_engine.start())
    
    # 7. Initialize Voice (if available)
    voice_task = None
    if VoicePipeline:
        voice = VoicePipeline(bus=bus)
        voice_task = asyncio.create_task(voice.run())

    # 8. Initialize UI WebServer
    from ui.web.server import WebServer
    web_server = WebServer(bus=bus)
    web_task = asyncio.create_task(web_server.run())

    # Shutdown Handler
    shutdown_event = asyncio.Event()

    def shutdown_handler():
        logger.info("Shutdown signal received.")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGINT, shutdown_handler)
        loop.add_signal_handler(signal.SIGTERM, shutdown_handler)

    # Wait until shutdown
    await shutdown_event.wait()
    logger.info("Shutting down components...")

    # Broadcast shutdown event
    await bus.publish(EventTopic.SYSTEM_SHUTDOWN, {}, source="main")

    # Gracefully stop components
    camera_task.cancel()
    try:
        await camera_task
    except asyncio.CancelledError:
        pass

    await perception.shutdown()
    await orchestrator.shutdown()
    await alert_engine.shutdown()
    await episodic_memory.close()
    
    if voice_task:
        voice.stop()
        voice_task.cancel()
        
    await web_server.shutdown()
    
    bus.stop()
    await bus_task

    logger.info("DMS V5 SENTINEL stopped cleanly.")

if __name__ == "__main__":
    asyncio.run(main())
