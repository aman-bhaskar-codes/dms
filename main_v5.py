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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("dms.main_v5")


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
    # (Assuming we have a dummy camera or cv2.VideoCapture loop somewhere,
    # for now we just initialize the processor)
    perception = FrameProcessor(bus=bus, kalman_enabled=True)
    perception_task = asyncio.create_task(perception.run())

    # 4. Initialize Agents
    orchestrator = Orchestrator(bus=bus, memory=working_memory)
    safety_agent = SafetyAgent(bus=bus, memory=working_memory)
    
    # Register agents to the Orchestrator
    orchestrator.register_agent("safety", safety_agent)
    orchestrator_task = asyncio.create_task(orchestrator.run())

    # 5. Initialize Alerts
    alert_engine = AlertEngineV5(bus=bus)
    alert_task = asyncio.create_task(alert_engine.start())
    
    # 6. Initialize Voice (if available)
    voice_task = None
    if VoicePipeline:
        voice = VoicePipeline(bus=bus)
        voice_task = asyncio.create_task(voice.run())

    # 7. Initialize UI WebServer
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
    await perception.shutdown()
    await orchestrator.shutdown()
    await alert_engine.shutdown()
    await episodic_memory.close()
    
    bus.stop()
    await bus_task

    logger.info("DMS V5 SENTINEL stopped cleanly.")

if __name__ == "__main__":
    asyncio.run(main())
