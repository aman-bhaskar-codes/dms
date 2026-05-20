"""
V5 Episodic Memory — Asynchronous SQLite storage via aiosqlite.
Logs session metrics and critical events.
"""
import aiosqlite
import json
import time
import logging
from typing import Optional

logger = logging.getLogger("dms.memory.episodic")


class EpisodicMemory:
    """Async database interface for long-term logging."""

    def __init__(self, db_path: str = "dms_v5.db"):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        self._conn = await aiosqlite.connect(self.db_path)
        await self._init_schema()

    async def _init_schema(self):
        if not self._conn:
            return
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                driver_id TEXT,
                start_time REAL,
                end_time REAL,
                max_fatigue REAL,
                alert_count INTEGER
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                event_type TEXT,
                payload TEXT,
                fatigue_score REAL
            );
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
        """)
        await self._conn.commit()

    async def log_session_start(self, session_id: str, driver_id: str):
        if not self._conn:
            return
        await self._conn.execute(
            "INSERT INTO sessions (id, driver_id, start_time, max_fatigue, alert_count) VALUES (?, ?, ?, 0.0, 0)",
            (session_id, driver_id, time.time())
        )
        await self._conn.commit()

    async def log_event(self, session_id: str, event_type: str, payload: dict, fatigue_score: float):
        if not self._conn:
            return
        await self._conn.execute(
            "INSERT INTO events (session_id, timestamp, event_type, payload, fatigue_score) VALUES (?, ?, ?, ?, ?)",
            (session_id, time.time(), event_type, json.dumps(payload), fatigue_score)
        )
        await self._conn.commit()
        
    async def get_session_events(self, session_id: str, limit: int = 50) -> list[dict]:
        if not self._conn:
            return []
        async with self._conn.execute(
            "SELECT timestamp, event_type, payload, fatigue_score FROM events WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "timestamp": r[0],
                    "event_type": r[1],
                    "payload": json.loads(r[2]),
                    "fatigue_score": r[3]
                }
                for r in rows
            ]

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None
