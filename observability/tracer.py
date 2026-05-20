"""
Tracer V5 — Observability and performance auditing
Tracks component latencies and records signal drops.
"""
import time
import logging
from typing import Dict, Any

logger = logging.getLogger("dms.tracer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

class Tracer:
    def __init__(self):
        self._latencies: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}
        
    def start(self, component: str):
        self._start_times[component] = time.perf_counter()
        
    def end(self, component: str):
        if component in self._start_times:
            elapsed = (time.perf_counter() - self._start_times[component]) * 1000
            self._latencies[component] = elapsed
            
    def get_metrics(self) -> Dict[str, float]:
        return self._latencies.copy()

    def audit_signals(self, gaze_data: dict, head_data: dict, rppg_data: dict, perclos: float):
        # Log warnings if confidence is low or signals are dead
        if gaze_data.get("confidence", 1.0) < 0.5:
            logger.warning(f"[Audit] Low gaze confidence: {gaze_data}")
        if rppg_data.get("hr_state") in ("no_roi", "invalid_roi"):
            logger.warning(f"[Audit] rPPG ROI invalid: {rppg_data}")
        if head_data.get("jerk", False):
            logger.warning("[Audit] Sudden head jerk detected")

# Global tracer instance
tracer = Tracer()
