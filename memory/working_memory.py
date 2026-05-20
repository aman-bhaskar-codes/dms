"""
V5 Working Memory — Thread-safe ring buffer and state tracking.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import threading


@dataclass
class DriverState:
    """The single source of truth for current driver state."""
    # Perception
    fatigue_score: float = 0.0
    fatigue_level: str = "safe"
    fatigue_prediction_3min: float = 0.0
    ear: float = 0.30
    perclos: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    rppg_hr: float = 70.0
    gaze_quality: float = 1.0

    # Session
    session_id: str = ""
    driver_id: str = ""
    session_start: float = field(default_factory=time.time)
    drive_duration_min: float = 0.0
    total_alerts: int = 0
    microsleep_count: int = 0
    yawn_count: int = 0

    # Context
    time_since_last_break_min: float = 0.0
    estimated_time_to_destination_min: Optional[float] = None


class WorkingMemory:
    """
    Thread-safe ring-buffer + current state.
    O(1) read, O(1) write. No locks on reads thanks to GIL + atomic assignment.
    """

    HISTORY_DEPTH = 1800  # 60s @ 30fps

    def __init__(self) -> None:
        self._state = DriverState()
        self._score_history: deque[tuple[float, float]] = deque(
            maxlen=self.HISTORY_DEPTH
        )
        self._alert_history: deque[dict] = deque(maxlen=100)
        self._lock = threading.Lock()

    def update_state(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._state, k):
                    setattr(self._state, k, v)
            self._score_history.append(
                (time.time(), self._state.fatigue_score)
            )

    def snapshot(self) -> DriverState:
        """Return a copy of current state (safe across threads)."""
        with self._lock:
            import copy
            return copy.copy(self._state)

    def score_trend(self, window_sec: int = 60) -> float:
        """Returns slope of fatigue score over window (fatigue units/min)."""
        now = time.time()
        cutoff = now - window_sec
        recent = [(t, s) for t, s in self._score_history if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        times = [r[0] for r in recent]
        scores = [r[1] for r in recent]
        # Linear regression slope
        n = len(times)
        t_mean = sum(times) / n
        s_mean = sum(scores) / n
        num = sum((t - t_mean) * (s - s_mean) for t, s in zip(times, scores))
        den = sum((t - t_mean) ** 2 for t in times) + 1e-9
        return (num / den) * 60  # per minute
