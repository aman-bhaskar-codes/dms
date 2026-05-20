"""
SQLite Database for DMS V4
Handles driver memory, session events, and logs.
Replaces ChromaDB from V3 for a simpler, zero-dependency storage.
"""
import aiosqlite
import json
import time
from typing import Dict, List, Optional
from config import settings


class DatabaseSchema:
    CREATE_SESSIONS = """
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        driver_id TEXT,
        start_time REAL,
        end_time REAL,
        avg_fatigue REAL,
        total_alerts INTEGER
    )
    """

    CREATE_EVENTS = """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        timestamp REAL,
        event_type TEXT,
        severity TEXT,
        fatigue_score REAL,
        details TEXT,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """

    CREATE_TIPS = """
    CREATE TABLE IF NOT EXISTS ai_tips (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        timestamp REAL,
        context TEXT,
        tip_text TEXT,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """


class DatabaseManager:
    def __init__(self):
        self.db_path = settings.db_path

    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(DatabaseSchema.CREATE_SESSIONS)
            await db.execute(DatabaseSchema.CREATE_EVENTS)
            await db.execute(DatabaseSchema.CREATE_TIPS)
            await db.commit()
        print(f"[DB] SQLite initialized at {self.db_path}")

    async def start_session(self, driver_id: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO sessions (driver_id, start_time) VALUES (?, ?)",
                (driver_id, time.time())
            )
            await db.commit()
            return cursor.lastrowid

    async def end_session(self, session_id: int, avg_fatigue: float, alerts: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE sessions 
                SET end_time = ?, avg_fatigue = ?, total_alerts = ?
                WHERE id = ?
                """,
                (time.time(), avg_fatigue, alerts, session_id)
            )
            await db.commit()

    async def log_event(self, session_id: int, event_type: str, severity: str,
                        fatigue_score: float, details: dict):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO events (session_id, timestamp, event_type, severity, fatigue_score, details)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, time.time(), event_type, severity, fatigue_score, json.dumps(details))
            )
            await db.commit()

    async def log_ai_tip(self, session_id: int, context: str, tip_text: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO ai_tips (session_id, timestamp, context, tip_text) VALUES (?, ?, ?, ?)",
                (session_id, time.time(), context, tip_text)
            )
            await db.commit()

    async def get_recent_events(self, session_id: int, limit: int = 20) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM events 
                WHERE session_id = ? 
                ORDER BY timestamp DESC LIMIT ?
                """,
                (session_id, limit)
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    async def get_recent_ai_tips(self, session_id: int, limit: int = 5) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM ai_tips 
                WHERE session_id = ? 
                ORDER BY timestamp DESC LIMIT ?
                """,
                (session_id, limit)
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
            
    async def get_session_summary(self, session_id: int) -> Dict:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT COUNT(*) as alert_count FROM events WHERE session_id = ? AND severity IN ('medium', 'high', 'critical')",
                (session_id,)
            )
            row = await cursor.fetchone()
            return {"total_alerts": row["alert_count"] if row else 0}

    async def get_recent_alerts(self, session_id: int, window_minutes: int = 10) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cutoff = time.time() - (window_minutes * 60)
            cursor = await db.execute(
                """
                SELECT * FROM events 
                WHERE session_id = ? AND timestamp > ? AND severity IN ('medium', 'high', 'critical')
                ORDER BY timestamp DESC
                """,
                (session_id, cutoff)
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
            
    async def get_driver_history(self, driver_id: str, last_n: int = 5) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM sessions 
                WHERE driver_id = ? AND end_time IS NOT NULL
                ORDER BY end_time DESC LIMIT ?
                """,
                (driver_id, last_n)
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
