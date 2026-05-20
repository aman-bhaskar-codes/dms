"""
Session Logger V2 — SQLite + JSON Export

V2 additions:
  - Fatigue score logging
  - Heart rate (rPPG) logging
  - Blink dynamics metrics
  - JSON session export for external analysis
  - Structured query helpers
"""

import sqlite3
import json
import time
import os
from typing import Dict, Optional
import config


class SessionLogger:
    def __init__(self):
        os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
        os.makedirs(config.EXPORT_DIR, exist_ok=True)

        self.conn       = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        self.session_id : Optional[int] = None
        self.frame_num  = 0
        self._create_schema()
        print(f"[Logger V2] DB: {config.DB_PATH}")

    def _create_schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at       REAL,
            ended_at         REAL,
            duration_s       REAL,
            driver_profile   TEXT,
            total_frames     INTEGER,
            total_alerts     INTEGER,
            avg_ear          REAL,
            avg_perclos      REAL,
            avg_fatigue      REAL,
            peak_fatigue     REAL,
            avg_hr           REAL,
            yawn_count       INTEGER,
            drowsy_events    INTEGER,
            distract_events  INTEGER,
            phone_events     INTEGER,
            jerk_events      INTEGER
        );

        CREATE TABLE IF NOT EXISTS frames (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     INTEGER,
            timestamp      REAL,
            frame_num      INTEGER,
            ear            REAL,
            mar            REAL,
            pitch          REAL,
            yaw            REAL,
            roll           REAL,
            perclos        REAL,
            fatigue_score  REAL,
            gaze_x         REAL,
            gaze_y         REAL,
            hr_bpm         REAL,
            slow_blink_r   REAL,
            ear_state      TEXT,
            head_state     TEXT,
            fatigue_level  TEXT,
            phone          INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   INTEGER,
            timestamp    REAL,
            event_type   TEXT,
            severity     TEXT,
            details      TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_frames_sid ON frames(session_id);
        CREATE INDEX IF NOT EXISTS idx_events_sid ON events(session_id);
        CREATE INDEX IF NOT EXISTS idx_frames_ts  ON frames(timestamp);
        """)
        self.conn.commit()

    def start_session(self, driver_profile: str = "default") -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (started_at, driver_profile, total_alerts,"
            " yawn_count, drowsy_events, distract_events, phone_events, jerk_events)"
            " VALUES (?, ?, 0, 0, 0, 0, 0, 0)",
            (time.time(), driver_profile)
        )
        self.conn.commit()
        self.session_id = cur.lastrowid
        self.frame_num  = 0
        print(f"[Logger V2] Session {self.session_id} started. Profile: {driver_profile}")
        return self.session_id

    def log_frame(self, metrics: Dict):
        if self.session_id is None:
            return
        self.frame_num += 1
        self.conn.execute("""
            INSERT INTO frames (session_id, timestamp, frame_num,
              ear, mar, pitch, yaw, roll, perclos, fatigue_score,
              gaze_x, gaze_y, hr_bpm, slow_blink_r,
              ear_state, head_state, fatigue_level, phone)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            self.session_id, time.time(), self.frame_num,
            metrics.get("ear",           0.0),
            metrics.get("mar",           0.0),
            metrics.get("pitch",         0.0),
            metrics.get("yaw",           0.0),
            metrics.get("roll",          0.0),
            metrics.get("perclos",       0.0),
            metrics.get("fatigue_score", 0.0),
            metrics.get("gaze_x",        0.0),
            metrics.get("gaze_y",        0.0),
            metrics.get("hr_bpm",        0.0),
            metrics.get("slow_blink_r",  0.0),
            metrics.get("ear_state",     "normal"),
            metrics.get("head_state",    "normal"),
            metrics.get("fatigue_level", "normal"),
            1 if metrics.get("phone_detected", False) else 0,
        ))
        self.conn.commit()

    def log_event(self, event_type: str, severity: str = "INFO", details: str = ""):
        if self.session_id is None:
            return
        self.conn.execute(
            "INSERT INTO events (session_id, timestamp, event_type, severity, details)"
            " VALUES (?,?,?,?,?)",
            (self.session_id, time.time(), event_type, severity, details)
        )
        self.conn.commit()

    def end_session(self, summary: Dict):
        if self.session_id is None:
            return
        now = time.time()
        started = self.conn.execute(
            "SELECT started_at FROM sessions WHERE id=?", (self.session_id,)
        ).fetchone()[0]

        self.conn.execute("""
            UPDATE sessions SET
              ended_at=?, duration_s=?, total_frames=?,
              total_alerts=?, avg_ear=?, avg_perclos=?,
              avg_fatigue=?, peak_fatigue=?, avg_hr=?,
              yawn_count=?, drowsy_events=?, distract_events=?,
              phone_events=?, jerk_events=?
            WHERE id=?
        """, (
            now, now - started, self.frame_num,
            summary.get("total_alerts",   0),
            summary.get("avg_ear",        0.0),
            summary.get("avg_perclos",    0.0),
            summary.get("avg_fatigue",    0.0),
            summary.get("peak_fatigue",   0.0),
            summary.get("avg_hr",         0.0),
            summary.get("yawns",          0),
            summary.get("drowsy_events",  0),
            summary.get("distract_ev",    0),
            summary.get("phone_ev",       0),
            summary.get("jerk_ev",        0),
            self.session_id,
        ))
        self.conn.commit()

        # JSON export
        if config.EXPORT_JSON:
            self._export_json(summary, now - started)

        print(f"[Logger V2] Session {self.session_id} closed. {now-started:.0f}s")

    def _export_json(self, summary: Dict, duration: float):
        """Export session summary to JSON."""
        data = {
            "session_id": self.session_id,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(duration, 1),
            **summary
        }
        path = os.path.join(config.EXPORT_DIR,
                            f"session_{self.session_id}_{int(time.time())}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Logger V2] JSON exported: {path}")

    def get_session_data(self, session_id: int) -> Dict:
        session = self.conn.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        events = self.conn.execute(
            "SELECT timestamp, event_type, severity FROM events"
            " WHERE session_id=? ORDER BY timestamp", (session_id,)
        ).fetchall()
        stats = self.conn.execute("""
            SELECT AVG(ear), MIN(ear), AVG(perclos), MAX(perclos),
                   AVG(ABS(yaw)), AVG(ABS(pitch)),
                   AVG(fatigue_score), MAX(fatigue_score),
                   AVG(NULLIF(hr_bpm, 0))
            FROM frames WHERE session_id=?
        """, (session_id,)).fetchone()
        return {"session": session, "events": events, "frame_stats": stats}

    def close(self):
        self.conn.close()
        print("[Logger V2] Closed.")
