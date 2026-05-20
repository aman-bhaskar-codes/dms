"""
Typed Async Event Bus — the backbone of DMS V5.
Every module communicates ONLY through this bus.
Zero direct imports between modules.
"""

from __future__ import annotations
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional
import logging

logger = logging.getLogger("dms.bus")


class EventTopic(str, Enum):
    # Perception events
    FRAME_RAW          = "frame.raw"
    FRAME_PROCESSED    = "frame.processed"
    FACE_DETECTED      = "perception.face_detected"
    FACE_LOST          = "perception.face_lost"

    # Signal events (one per detector)
    SIGNAL_EAR         = "signal.ear"
    SIGNAL_MAR         = "signal.mar"
    SIGNAL_PERCLOS     = "signal.perclos"
    SIGNAL_HEAD_POSE   = "signal.head_pose"
    SIGNAL_GAZE        = "signal.gaze"
    SIGNAL_RPPG        = "signal.rppg"
    SIGNAL_MICRO_EXPR  = "signal.micro_expression"
    SIGNAL_OBJECTS     = "signal.objects"        # YOLOv8 detections
    SIGNAL_TRAFFIC     = "signal.traffic"        # NEW: traffic context

    # Fusion events
    FATIGUE_SCORE      = "fatigue.score"         # FatigueFrame dataclass
    FATIGUE_CRITICAL   = "fatigue.critical"

    # Alert events
    ALERT_TRIGGER      = "alert.trigger"
    ALERT_RESOLVED     = "alert.resolved"

    # Voice events
    VOICE_INPUT_RAW    = "voice.input_raw"       # bytes from mic
    VOICE_TRANSCRIPT   = "voice.transcript"      # str from Whisper
    VOICE_INTENT       = "voice.intent"          # parsed intent
    VOICE_RESPONSE     = "voice.response"        # str to speak
    VOICE_SPEAKING     = "voice.speaking"        # TTS state

    # Memory events
    MEMORY_WRITE       = "memory.write"
    MEMORY_READ_REQ    = "memory.read_request"
    MEMORY_READ_RESP   = "memory.read_response"

    # Agent events
    AGENT_TASK         = "agent.task"
    AGENT_RESULT       = "agent.result"

    # System events
    SYSTEM_SHUTDOWN    = "system.shutdown"
    SYSTEM_CALIBRATE   = "system.calibrate"
    SYSTEM_HEALTH      = "system.health"


@dataclass
class Event:
    topic: EventTopic
    payload: Any
    source: str = "unknown"
    timestamp: float = field(default_factory=time.monotonic)
    correlation_id: Optional[str] = None  # for request/response pairs


Handler = Callable[[Event], Coroutine]


class EventBus:
    """
    Async pub/sub event bus with:
    - Topic wildcards (future)
    - Back-pressure via bounded queues
    - Dead-letter queue for failed handlers
    - Metrics collection
    """

    def __init__(self, queue_size: int = 512):
        self._handlers: Dict[EventTopic, List[Handler]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_size)
        self._running = False
        self._stats: Dict[str, int] = defaultdict(int)
        self._dead_letter: List[Event] = []

    def subscribe(self, topic: EventTopic, handler: Handler) -> None:
        self._handlers[topic].append(handler)
        logger.debug(f"[Bus] Subscribed {handler.__qualname__} → {topic}")

    async def publish(self, topic: EventTopic, payload: Any,
                      source: str = "unknown",
                      correlation_id: Optional[str] = None) -> None:
        event = Event(topic=topic, payload=payload,
                      source=source, correlation_id=correlation_id)
        try:
            self._queue.put_nowait(event)
            self._stats["published"] += 1
        except asyncio.QueueFull:
            logger.warning(f"[Bus] Queue full — dropping {topic} from {source}")
            self._stats["dropped"] += 1

    async def run(self) -> None:
        """Main dispatch loop. Run as asyncio task."""
        self._running = True
        logger.info("[Bus] Event bus running.")
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                handlers = self._handlers.get(event.topic, [])
                for handler in handlers:
                    try:
                        await handler(event)
                        self._stats["dispatched"] += 1
                    except Exception as e:
                        logger.error(f"[Bus] Handler {handler.__qualname__} "
                                     f"failed for {event.topic}: {e}")
                        self._dead_letter.append(event)
                        self._stats["errors"] += 1
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        self._running = False

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


# Singleton — import this everywhere
bus = EventBus()
