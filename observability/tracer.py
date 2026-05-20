"""
DMS V5 Observability — Langfuse traces + per-frame timing + signal audit.
Every detector call, every AI call, every alert — traced and queryable.
"""
from __future__ import annotations
import time
import functools
import asyncio
from typing import Callable, Any, Optional
from contextlib import contextmanager, asynccontextmanager
from collections import deque
import statistics

try:
    from langfuse import Langfuse
    _langfuse_available = True
except ImportError:
    _langfuse_available = False


class FrameTimer:
    """Measures per-component latency in the frame processing loop."""
    
    def __init__(self, window: int = 300):
        self._timings: dict[str, deque] = {}
        self._window = window
    
    @contextmanager
    def measure(self, component: str):
        """Usage: with timer.measure("ear_detector"): ..."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if component not in self._timings:
                self._timings[component] = deque(maxlen=self._window)
            self._timings[component].append(elapsed_ms)
    
    def get_stats(self) -> dict:
        """Returns mean/max/p95 latency per component."""
        stats = {}
        for name, times in self._timings.items():
            if times:
                t = list(times)
                stats[name] = {
                    "mean_ms": round(statistics.mean(t), 2),
                    "max_ms":  round(max(t), 2),
                    "p95_ms":  round(sorted(t)[int(len(t) * 0.95)], 2),
                }
        return stats
    
    def total_frame_ms(self) -> float:
        """Sum of all component means — is it under 33ms?"""
        return sum(
            statistics.mean(t) for t in self._timings.values() if t
        )


class SignalAuditor:
    """Tracks contribution of each signal to fatigue score over time."""
    
    def __init__(self, window: int = 300):
        self._contributions: dict[str, deque] = {}
        self._window = window
    
    def record(self, signals: dict):
        """Record per-signal normalized contributions this frame."""
        for name, value in signals.items():
            if name not in self._contributions:
                self._contributions[name] = deque(maxlen=self._window)
            self._contributions[name].append(float(value))
    
    def get_audit(self) -> dict:
        """Returns mean contribution per signal — reveals which are firing."""
        return {
            name: round(statistics.mean(vals), 3)
            for name, vals in self._contributions.items()
            if vals
        }
    
    def get_dominant_signal(self) -> str:
        """Which signal is contributing most to fatigue right now?"""
        audit = self.get_audit()
        if not audit:
            return "none"
        return max(audit, key=audit.get)


class AlertAuditor:
    """Tracks every alert — for evaluation and no-repeat logic."""
    
    def __init__(self):
        self._log: list = []
        self._counts: dict = {}
    
    def record(self, alert_type: str, level: str, fatigue_score: float,
               driver_responded: bool = False):
        entry = {
            "type": alert_type, "level": level,
            "fatigue": fatigue_score,
            "timestamp": time.time(),
            "responded": driver_responded,
        }
        self._log.append(entry)
        self._counts[alert_type] = self._counts.get(alert_type, 0) + 1
    
    def already_alerted_recently(self, alert_type: str, window_sec: float = 120) -> bool:
        """Check if this alert type was given in the last N seconds."""
        cutoff = time.time() - window_sec
        return any(
            a["type"] == alert_type and a["timestamp"] > cutoff
            for a in self._log
        )
    
    def get_summary(self) -> dict:
        return {
            "total": len(self._log),
            "by_type": self._counts.copy(),
            "last_alert": self._log[-1] if self._log else None,
        }


class LangfuseTracer:
    """Optional Langfuse integration for AI call tracing."""
    
    def __init__(self, public_key: str = "", secret_key: str = "", host: str = ""):
        self._langfuse = None
        if _langfuse_available and public_key:
            try:
                self._langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host or "http://localhost:3000",
                )
                print("[Langfuse] Connected to observability server")
            except Exception as e:
                print(f"[Langfuse] Not available: {e}")
    
    def trace_ai_call(self, name: str, prompt: str, response: str,
                       model: str, latency_ms: float, metadata: dict = None):
        """Trace a single Ollama/Claude call to Langfuse."""
        if self._langfuse is None:
            return
        try:
            trace = self._langfuse.trace(name=f"dms.{name}")
            trace.generation(
                name=name,
                model=model,
                input=prompt[:500],  # Truncate for privacy
                output=response,
                metadata={
                    "latency_ms": latency_ms,
                    "dms_context": metadata or {}
                }
            )
        except Exception:
            pass  # Observability never crashes the main system
    
    def flush(self):
        if self._langfuse:
            self._langfuse.flush()


# ── Global singletons ──────────────────────────────────────────────────────────
frame_timer    = FrameTimer()
signal_auditor = SignalAuditor()
alert_auditor  = AlertAuditor()
langfuse_tracer = LangfuseTracer()  # Noop until configured
