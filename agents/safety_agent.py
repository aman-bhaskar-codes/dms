"""
V5 Safety Agent — Deterministic rules mapping fatigue scores to alerts.
"""
import logging
import time
from core.bus import EventBus, EventTopic
from memory.working_memory import WorkingMemory

logger = logging.getLogger("dms.agents.safety")


class SafetyAgent:
    """
    Evaluates fatigue states and fires alerts. 
    Maintains state to prevent alert spam.
    """
    def __init__(self, bus: EventBus, memory: WorkingMemory):
        self._bus = bus
        self._memory = memory
        self._last_alert_time = 0.0
        self._alert_cooldown = 5.0  # seconds

    async def evaluate(self, payload: dict):
        score = payload.get("score", 0.0)
        level = payload.get("level", "safe")
        
        if level in ["warning", "critical"]:
            await self._trigger_alert(level, score)

    async def handle_critical(self, payload: dict):
        score = payload.get("score", 100.0)
        await self._trigger_alert("critical", score, bypass_cooldown=True)

    async def _trigger_alert(self, level: str, score: float, bypass_cooldown: bool = False):
        now = time.time()
        if not bypass_cooldown and (now - self._last_alert_time) < self._alert_cooldown:
            return

        self._last_alert_time = now
        logger.warning(f"[SafetyAgent] Firing {level.upper()} alert! Score: {score}")

        await self._bus.publish(
            EventTopic.UI_OVERLAY_UPDATE,
            {"alert": level, "message": f"{level.upper()} FATIGUE DETECTED"},
            source="safety_agent"
        )
        
        # Audio alert based on severity
        if level == "critical":
            await self._bus.publish(EventTopic.VOICE_SPEAK, {"text": "Critical fatigue. Pull over immediately."}, source="safety")
        elif level == "warning":
            await self._bus.publish(EventTopic.VOICE_SPEAK, {"text": "You seem tired. Please take a break."}, source="safety")
