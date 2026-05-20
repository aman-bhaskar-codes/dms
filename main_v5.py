"""
Thin orchestrator — wires modules to the bus and starts tasks.
No logic lives here. All logic lives in modules.
"""

import asyncio
import signal
import logging
from core.bus import bus, EventTopic
from core.config import settings

# --- Module imports (each self-registers on import) ---
# NOTE: These modules will be created in upcoming phases.
# For Phase 1 testing, we will mock them if they are missing.
try:
    from perception.pipeline import PerceptionPipeline
    from detection.yolo_engine import YOLOEngine
    from detection.traffic_detector import TrafficDetector
    from memory.memory_system import MemorySystem
    from rag.rag_engine import RAGEngine
    from voice.voice_pipeline import VoicePipeline
    from agents.orchestrator import AgentOrchestrator
    from alerts.alert_engine import AlertEngine
    from ui.web.server import WebServer
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Warning: Not all V5 modules are available yet ({e}). Running in core-only mode.")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("dms.main")


async def main():
    logger.info("🚗⚡ DMS V5 SENTINEL starting...")

    tasks = [bus.run()]

    if MODULES_AVAILABLE:
        # Init all modules — they subscribe to bus in __init__
        memory   = MemorySystem(bus)
        rag      = RAGEngine(bus)
        voice    = VoicePipeline(bus)
        agents   = AgentOrchestrator(bus)
        alerts   = AlertEngine(bus)
        web      = WebServer(bus)
        percept  = PerceptionPipeline(bus)
        yolo     = YOLOEngine(bus)
        traffic  = TrafficDetector(bus)
        
        tasks.extend([
            percept.run(),            # camera → bus
            yolo.run(),               # background detection
            traffic.run(),            # traffic context
            memory.run(),             # persistence
            rag.run(),                # RAG queries
            voice.run(),              # STT + TTS
            agents.run(),             # agent loop
            alerts.run(),             # alert dispatcher
            web.run(),                # HTTP + WebSocket server
        ])
    else:
        logger.info("Running Event Bus in isolated mode (Phase 1).")

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    def shutdown_handler():
        logger.info("Shutdown signal received.")
        asyncio.create_task(
            bus.publish(EventTopic.SYSTEM_SHUTDOWN, {}, source="main")
        )
    
    try:
        loop.add_signal_handler(signal.SIGINT, shutdown_handler)
        loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
    except NotImplementedError:
        # Add signal handler doesn't work on Windows, safe to ignore for local dev if needed
        pass

    # Run everything concurrently
    await asyncio.gather(*tasks)

    logger.info("DMS V5 SENTINEL stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
