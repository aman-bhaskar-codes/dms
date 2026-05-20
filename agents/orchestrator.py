"""
V5 Orchestrator — True ReAct Loop based on asyncio.PriorityQueue.
Replaces LangGraph polling with purely event-driven architecture.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from core.bus import EventBus, EventTopic
from memory.working_memory import WorkingMemory

logger = logging.getLogger("dms.agents.orchestrator")


@dataclass(order=True)
class AgentTask:
    """Prioritized task for the ReAct loop."""
    priority: int      # 0 = CRITICAL, 50 = NORMAL, 100 = LOW
    topic: EventTopic  = field(compare=False)
    payload: dict      = field(compare=False)


class Orchestrator:
    """
    Central brain. Subscribes to bus events, places them in priority queue,
    and dispatches to registered agents.
    """

    def __init__(self, bus: EventBus, memory: WorkingMemory):
        self._bus = bus
        self._memory = memory
        self._queue: asyncio.PriorityQueue[AgentTask] = asyncio.PriorityQueue()
        self._agents: dict[str, Any] = {}
        self._running = False
        
        # Core subscriptions
        self._bus.subscribe(EventTopic.FATIGUE_CRITICAL, self._handle_critical)
        self._bus.subscribe(EventTopic.FATIGUE_SCORE, self._handle_normal)
        self._bus.subscribe(EventTopic.VOICE_INTENT, self._handle_voice)

    def register_agent(self, name: str, agent: Any):
        self._agents[name] = agent
        logger.info(f"[Orchestrator] Registered agent: {name}")

    async def _handle_critical(self, payload: dict):
        # Priority 0
        await self._queue.put(AgentTask(priority=0, topic=EventTopic.FATIGUE_CRITICAL, payload=payload))

    async def _handle_voice(self, payload: dict):
        # Priority 10
        await self._queue.put(AgentTask(priority=10, topic=EventTopic.VOICE_INTENT, payload=payload))

    async def _handle_normal(self, payload: dict):
        # Priority 50
        # Don't flood the queue; only if empty or significant change
        if self._queue.qsize() < 10:
            await self._queue.put(AgentTask(priority=50, topic=EventTopic.FATIGUE_SCORE, payload=payload))

    async def run(self):
        self._running = True
        logger.info("[Orchestrator] ReAct loop started.")
        while self._running:
            try:
                task: AgentTask = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._process_task(task)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Orchestrator] Task processing error: {e}")

    async def _process_task(self, task: AgentTask):
        """Dispatch task to appropriate agent."""
        if task.topic == EventTopic.FATIGUE_CRITICAL:
            if "safety" in self._agents:
                await self._agents["safety"].handle_critical(task.payload)
        elif task.topic == EventTopic.FATIGUE_SCORE:
            if "safety" in self._agents:
                await self._agents["safety"].evaluate(task.payload)
        elif task.topic == EventTopic.VOICE_INTENT:
            # Route voice intents (e.g. "I'm tired" -> find POI)
            intent = task.payload.get("intent")
            if intent == "find_poi" and "rag" in self._agents:
                logger.info(f"[Orchestrator] Dispatching to RAG for POI search")
                await self._agents["rag"].search(task.payload.get("text"))

    async def shutdown(self):
        self._running = False
        logger.info("[Orchestrator] Shutdown complete.")
