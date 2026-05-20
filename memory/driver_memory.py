"""
Driver Memory manager for V4.
Handles runtime state, session context for AI, and profile stats.
"""
import time
from datetime import datetime
from typing import Dict, List
from memory.database import DatabaseManager


class DriverMemory:
    def __init__(self, db: DatabaseManager, driver_id: str = "default"):
        self.db = db
        self.driver_id = driver_id
        self.session_id = None
        self.alert_count = 0
        self.total_fatigue = 0.0
        self.frame_count = 0
        self.last_ai_tip_time = 0.0

    async def start_session(self):
        self.session_id = await self.db.start_session(self.driver_id)
        print(f"[Memory] Session started. ID: {self.session_id}")

    async def log_event(self, event_type: str, severity: str, score: float, details: dict):
        if self.session_id is None:
            return
        await self.db.log_event(self.session_id, event_type, severity, score, details)
        self.alert_count += 1
        print(f"[Memory] Logged event: {event_type} ({severity})")

    async def log_ai_tip(self, context: str, tip_text: str):
        if self.session_id is None:
            return
        await self.db.log_ai_tip(self.session_id, context, tip_text)

    def update_fatigue(self, score: float):
        self.total_fatigue += score
        self.frame_count += 1

    async def end_session(self):
        if self.session_id is None:
            return
        avg = self.total_fatigue / max(1, self.frame_count)
        await self.db.end_session(self.session_id, avg, self.alert_count)
        print(f"[Memory] Session ended. Avg Fatigue: {avg:.1f}")

    async def get_session_context(self, metrics: dict = None) -> dict:
        """Build full context for AI — specific, grounded, non-repetitive."""
        if not metrics: metrics = {}
        if not self.session_id:
            return {"summary": "No active session."}
        
        # Recent alert history (last 10 min)
        recent_alerts = await self.db.get_recent_alerts(
            self.session_id, window_minutes=10
        )
        alert_types_given = [a["event_type"] for a in recent_alerts]
        
        # Historical baseline for this driver
        history = await self.db.get_driver_history(self.driver_id, last_n=5)
        avg_fatigue_hist = sum(s.get("avg_fatigue", 0) for s in history) / max(len(history), 1)
        
        # Time context
        hour = datetime.now().hour
        time_risk = "high" if (hour < 6 or hour > 21) else "medium" if hour > 18 else "low"
        
        return {
            "current_fatigue": metrics.get("fatigue_score", 0),
            "fatigue_trend": metrics.get("fatigue_trend", 0),
            "drive_minutes": metrics.get("drive_minutes", 0),
            "alerts_given_already": alert_types_given,
            "historical_avg_fatigue": avg_fatigue_hist,
            "performing_vs_history": "worse" if metrics.get("fatigue_score", 0) > avg_fatigue_hist + 10 else "better",
            "time_of_day_risk": time_risk,
            "yawn_rate": metrics.get("yawn_rate", 0),
            "ear": metrics.get("ear", 0),
            "perclos_pct": metrics.get("perclos", 0) * 100,
            "summary": "Active driver session"
        }
