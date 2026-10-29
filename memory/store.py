"""Versioned, local-only session storage. JSON payloads are explicitly allowlisted."""

import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4
import aiosqlite


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.db = None
        self.lock = asyncio.Lock()

    async def connect(self, recover_sessions=True):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            # SQLite backup API includes committed WAL content in the backup.
            import sqlite3

            with sqlite3.connect(self.path) as source:
                version = source.execute("PRAGMA user_version").fetchone()[0]
                if version > 1:
                    raise ValueError("Database was created by a newer Sentinel version")
                if version < 1:
                    with sqlite3.connect(str(self.path) + ".pre-v1.bak") as target:
                        source.backup(target)
        self.db = await aiosqlite.connect(self.path)
        await self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS sentinel_profiles (id TEXT PRIMARY KEY, name TEXT NOT NULL);
            INSERT OR IGNORE INTO sentinel_profiles VALUES('default','Local driver');
            CREATE TABLE IF NOT EXISTS sentinel_sessions (
                id TEXT PRIMARY KEY, profile TEXT NOT NULL, mode TEXT NOT NULL,
                start REAL NOT NULL, end REAL, interrupted INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS sentinel_metrics (
                session_id TEXT REFERENCES sentinel_sessions(id) ON DELETE CASCADE,
                captured REAL, payload TEXT NOT NULL,
                PRIMARY KEY(session_id, captured));
            CREATE TABLE IF NOT EXISTS sentinel_events (
                id TEXT PRIMARY KEY, session_id TEXT REFERENCES sentinel_sessions(id) ON DELETE CASCADE,
                timestamp REAL, kind TEXT, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sentinel_calibrations (
                id TEXT PRIMARY KEY, profile TEXT, created REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS sentinel_usage (
                day TEXT PRIMARY KEY, count INTEGER, last_request REAL);
            PRAGMA user_version=1;
        """)
        if recover_sessions:
            await self.db.execute(
                "UPDATE sentinel_sessions SET end=?, interrupted=1 WHERE end IS NULL",
                (time.time(),),
            )
        await self.db.commit()

    async def close(self):
        if self.db:
            await self.db.close()
            self.db = None

    async def start_session(self, profile="default", mode="live"):
        session_id = str(uuid4())
        async with self.lock:
            await self.db.execute(
                "INSERT INTO sentinel_sessions(id,profile,mode,start) VALUES(?,?,?,?)",
                (session_id, profile, mode, time.time()),
            )
            await self.db.commit()
        return session_id

    async def end_session(self, session_id):
        async with self.lock:
            await self.db.execute(
                "UPDATE sentinel_sessions SET end=? WHERE id=? AND end IS NULL",
                (time.time(), session_id),
            )
            await self.db.commit()

    async def record_snapshot(self, snapshot):
        await self.record_batch([("snapshot", snapshot)])

    async def record_event(self, session_id, event, kind="alert"):
        await self.record_batch([("event", (session_id, event, kind))])

    async def record_batch(self, records):
        async with self.lock:
            try:
                for kind, value in records:
                    if kind == "snapshot":
                        await self.db.execute(
                            "INSERT OR REPLACE INTO sentinel_metrics VALUES(?,?,?)",
                            (value.session_id, value.captured_at, value.model_dump_json()),
                        )
                    else:
                        sid, event, event_kind = value
                        payload = (
                            event.model_dump(mode="json") if hasattr(event, "model_dump") else event
                        )
                        await self.db.execute(
                            "INSERT OR IGNORE INTO sentinel_events VALUES(?,?,?,?,?)",
                            (
                                payload.get("id", str(uuid4())),
                                sid,
                                payload.get("timestamp", time.time()),
                                event_kind,
                                json.dumps(payload),
                            ),
                        )
                await self.db.commit()
            except BaseException:
                await self.db.rollback()
                raise

    async def sessions(self, profile="default"):
        async with self.lock:
            cursor = await self.db.execute(
                "SELECT id,profile,mode,start,end,interrupted FROM sentinel_sessions "
                "WHERE profile=? ORDER BY start DESC LIMIT 200",
                (profile,),
            )
            rows = await cursor.fetchall()
        return [
            dict(zip(("id", "profile", "mode", "start", "end", "interrupted"), r)) for r in rows
        ]

    async def report(self, session_id, profile="default"):
        async with self.lock:
            cursor = await self.db.execute(
                "SELECT id,profile,mode,start,end,interrupted FROM sentinel_sessions WHERE id=? AND profile=?",
                (session_id, profile),
            )
            row = await cursor.fetchone()
            if row is None:
                raise KeyError(session_id)
            cursor = await self.db.execute(
                "SELECT payload FROM sentinel_metrics WHERE session_id=? ORDER BY captured",
                (session_id,),
            )
            metrics = [json.loads(r[0]) for r in await cursor.fetchall()]
            cursor = await self.db.execute(
                "SELECT kind,payload FROM sentinel_events WHERE session_id=? ORDER BY timestamp",
                (session_id,),
            )
            events = [dict(kind=r[0], **json.loads(r[1])) for r in await cursor.fetchall()]
        session = dict(zip(("id", "profile", "mode", "start", "end", "interrupted"), row))
        valid = sum(m["state"] == "monitoring" for m in metrics)
        scores = [m["score"] for m in metrics if m["score"] is not None]
        return dict(
            session=session,
            metrics=metrics,
            events=events,
            summary=dict(
                duration_s=max(0, (session["end"] or time.time()) - session["start"]),
                samples=len(metrics),
                tracking_coverage=valid / len(metrics) if metrics else None,
                alert_count=sum(e.get("transition") == "triggered" for e in events),
                max_score=max(scores) if scores else None,
            ),
        )

    async def delete_session(self, session_id, profile="default"):
        async with self.lock:
            cursor = await self.db.execute(
                "DELETE FROM sentinel_sessions WHERE id=? AND profile=? AND end IS NOT NULL",
                (session_id, profile),
            )
            await self.db.commit()
            return cursor.rowcount > 0

    async def save_calibration(self, data, profile="default"):
        async with self.lock:
            await self.db.execute(
                "INSERT INTO sentinel_calibrations VALUES(?,?,?,?)",
                (str(uuid4()), profile, time.time(), json.dumps(data)),
            )
            await self.db.commit()

    async def calibration(self, profile="default"):
        async with self.lock:
            cursor = await self.db.execute(
                "SELECT payload FROM sentinel_calibrations WHERE profile=? ORDER BY created DESC LIMIT 1",
                (profile,),
            )
            row = await cursor.fetchone()
        return json.loads(row[0]) if row else None

    async def reserve_usage(self, day, cap, interval, now):
        async with self.lock:
            cursor = await self.db.execute(
                "SELECT count,last_request FROM sentinel_usage WHERE day=?", (day,)
            )
            row = await cursor.fetchone()
            count, last = row if row else (0, -float("inf"))
            if count >= cap or now - last < interval:
                return False
            await self.db.execute(
                "INSERT OR REPLACE INTO sentinel_usage VALUES(?,?,?)", (day, count + 1, now)
            )
            await self.db.commit()
            return True

    async def usage(self, day):
        async with self.lock:
            cursor = await self.db.execute("SELECT count FROM sentinel_usage WHERE day=?", (day,))
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def cleanup(self, days):
        async with self.lock:
            await self.db.execute(
                "DELETE FROM sentinel_sessions WHERE end IS NOT NULL AND end<?",
                (time.time() - days * 86400,),
            )
            await self.db.commit()

    async def import_legacy(self, path):
        """Import V4/V5 sessions and allowlisted event fields without altering the source."""
        source_path = Path(path).resolve()
        if source_path == self.path.resolve():
            raise ValueError("Legacy source must differ from destination")
        import sqlite3

        with sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True) as source:
            source.row_factory = sqlite3.Row
            rows = source.execute(
                "SELECT id,driver_id,start_time,end_time FROM sessions"
            ).fetchall()
            has_events = source.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            events = (
                [dict(row) for row in source.execute("SELECT * FROM events").fetchall()]
                if has_events
                else []
            )
        imported_ids = {str(row["id"]) for row in rows}
        async with self.lock:
            try:
                for row in rows:
                    await self.db.execute(
                        "INSERT OR IGNORE INTO sentinel_sessions VALUES(?,?,?,?,?,?)",
                        (
                            "legacy-" + str(row["id"]),
                            "default",
                            "legacy",
                            row["start_time"],
                            row["end_time"] or row["start_time"],
                            int(row["end_time"] is None),
                        ),
                    )
                for event in events:
                    sid = str(event.get("session_id"))
                    if sid not in imported_ids:
                        continue
                    severity = event.get("severity")
                    if severity is None and event.get("payload"):
                        try:
                            payload = json.loads(event["payload"])
                            severity = payload.get("level") if isinstance(payload, dict) else None
                        except (TypeError, ValueError):
                            pass
                    severity = {"high": "critical", "medium": "warning", "danger": "critical"}.get(
                        severity, severity
                    )
                    identity = "legacy-event-" + sid + "-" + str(event["id"])
                    safe = dict(
                        id=identity,
                        timestamp=event.get("timestamp", 0),
                        cause=str(event.get("event_type", "legacy"))[:120],
                        transition="triggered" if severity in ("warning", "critical") else "legacy",
                    )
                    if severity in ("warning", "critical"):
                        safe["severity"] = severity
                    await self.db.execute(
                        "INSERT OR IGNORE INTO sentinel_events VALUES(?,?,?,?,?)",
                        (identity, "legacy-" + sid, safe["timestamp"], "legacy", json.dumps(safe)),
                    )
                await self.db.commit()
            except BaseException:
                await self.db.rollback()
                raise
        return len(rows)
