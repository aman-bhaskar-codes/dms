"""Independent bounded subscriber queues. Safety runs directly before publication."""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventTopic(str, Enum):
    FRAME_PROCESSED = "frame.processed"
    FATIGUE_SCORE = "fatigue.score"
    FATIGUE_CRITICAL = "fatigue.critical"
    ALERT_TRIGGER = "alert.trigger"
    ALERT_RESOLVED = "alert.resolved"
    SYSTEM_HEALTH = "system.health"
    SYSTEM_SHUTDOWN = "system.shutdown"
    FACE_DETECTED = "perception.face_detected"
    FACE_LOST = "perception.face_lost"
    VOICE_RESPONSE = "voice.response"
    VOICE_TRANSCRIPT = "voice.transcript"
    VOICE_INTENT = "voice.intent"
    UI_OVERLAY_UPDATE = "ui.overlay.update"
    AGENT_TASK = "agent.task"
    AGENT_RESULT = "agent.result"
    SIGNAL_HEAD_POSE = "signal.head_pose"
    MEMORY_READ_REQ = "memory.read_request"
    MEMORY_READ_RESP = "memory.read_response"
    SYSTEM_CALIBRATE = "system.calibrate"


@dataclass
class Event:
    topic: EventTopic
    payload: Any
    source: str = "unknown"
    timestamp: float = field(default_factory=time.monotonic)
    correlation_id: str | None = None


class EventBus:
    def __init__(self, queue_size=32):
        self.queue_size = queue_size
        self.subscribers = defaultdict(list)
        self.tasks = []
        self.running = False
        self.stats = defaultdict(int)
        self._wake = asyncio.Event()

    def subscribe(self, topic, handler):
        queue = asyncio.Queue(maxsize=self.queue_size)
        self.subscribers[topic].append((handler, queue))
        if self.running:
            self.tasks.append(asyncio.create_task(self._consume(handler, queue)))

    async def publish(self, topic, payload, source="unknown", correlation_id=None):
        event = Event(topic, payload, source, correlation_id=correlation_id)
        for _, queue in self.subscribers[topic]:
            if queue.full():
                queue.get_nowait()
                queue.task_done()
                self.stats["coalesced"] += 1
            queue.put_nowait(event)
        self.stats["published"] += 1

    async def _consume(self, handler, queue):
        while True:
            event = await queue.get()
            try:
                await handler(event)
                self.stats["dispatched"] += 1
            except Exception:
                logger.exception("Subscriber failed for %s", event.topic)
                self.stats["errors"] += 1
            finally:
                queue.task_done()

    async def run(self):
        self.running = True
        for entries in self.subscribers.values():
            for handler, queue in entries:
                self.tasks.append(asyncio.create_task(self._consume(handler, queue)))
        await self._wake.wait()
        self.running = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def stop(self):
        self._wake.set()

    def get_stats(self):
        return dict(self.stats)


bus = EventBus()
