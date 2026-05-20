"""
Alert Engine V4 — handles alert state machine, deduplication, and TTS routing.
"""
import time
from config import settings
from alerts.tts_engine import TTSEngine
from alerts.telegram_notifier import TelegramNotifier
from memory.driver_memory import DriverMemory


class AlertEngine:
    def __init__(self, tts: TTSEngine, memory: DriverMemory, telegram: TelegramNotifier):
        self.tts = tts
        self.memory = memory
        self.telegram = telegram
        self._last_alert_time = 0.0
        self._active_alerts = {}

    async def trigger(self, event_type: str, severity: str, score: float = 0.0, details: dict = None):
        """
        event_type: 'fatigue', 'distraction', 'microsleep', 'yawn', 'phone'
        severity: 'mild', 'warning', 'critical'
        """
        now = time.time()
        
        # Per-event-type cooldowns to avoid database spamming and high-frequency alert loops
        # Critical: 5s, Warning: 10s, Mild/Other: 30s
        cooldown = 5.0 if severity == "critical" else (10.0 if severity == "warning" else 30.0)
        
        if event_type in self._active_alerts:
            last_t = self._active_alerts[event_type]
            if now - last_t < cooldown:
                return
                
        self._active_alerts[event_type] = now
        self._last_alert_time = now
        
        # Log to memory database
        await self.memory.log_event(event_type, severity, score, details or {})
        
        # Route to TTS
        msg = self._get_tts_message(event_type, severity)
        priority = 0 if severity == "critical" else (1 if severity == "warning" else 2)
        if msg:
            self.tts.speak(msg, priority=priority)
            
        # Route to Telegram if critical
        if severity == "critical":
            await self.telegram.send_alert(
                f"Event: {event_type.upper()}\nScore: {score:.1f}\nDetails: {details}"
            )

    def _get_tts_message(self, event_type: str, severity: str) -> str:
        if event_type == "fatigue":
            if severity == "critical": return settings.TTS_MESSAGES["fatigue_high"]
            if severity == "warning": return settings.TTS_MESSAGES["drowsy_warn"]
        elif event_type == "microsleep":
            return settings.TTS_MESSAGES["drowsy_crit"]
        elif event_type == "yawn" and severity == "warning":
            return settings.TTS_MESSAGES["yawn"]
        elif event_type == "distraction":
            return settings.TTS_MESSAGES["distracted"]
        elif event_type == "phone":
            return settings.TTS_MESSAGES["phone"]
        return ""
