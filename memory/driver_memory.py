"""
Driver Memory manager for V4.
Handles runtime state, session context for AI, and profile stats.
"""
import time
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

    async def get_session_context(self) -> str:
        """Compile recent events into a text string for the AI."""
        if not self.session_id:
            return "No active session."
        
        events = await self.db.get_recent_events(self.session_id, limit=10)
        if not events:
            return "Driver has been driving steadily with no major incidents."
            
        context = "Recent events in this session:\n"
        for e in events:
            mins_ago = (time.time() - e['timestamp']) / 60
            context += f"- {mins_ago:.1f} min ago: {e['event_type']} (Severity: {e['severity']}, Fatigue: {e['fatigue_score']:.1f})\n"
        return context
