"""
Driver Calibration V2 — 30s baseline capture, per-driver profile save/load.
Run once per driver. Updates EAR/MAR thresholds to their face geometry.
"""
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from config import settings


@dataclass
class DriverProfile:
    driver_id: str = "default"
    ear_threshold: float = settings.ear_threshold
    ear_open_baseline: float = settings.ear_open_baseline
    mar_threshold: float = settings.mar_threshold
    calibration_date: str = ""
    session_count: int = 0
    total_drive_minutes: float = 0.0
    avg_fatigue_score: float = 0.0
    sample_count: int = 0
    ear_samples: List[float] = field(default_factory=list)
    mar_samples: List[float] = field(default_factory=list)


class DriverCalibration:
    def __init__(self, driver_id: str = "default"):
        self.driver_id = driver_id
        self.profile = self._load_or_create(driver_id)
        self._collecting = False
        self._start_time = 0.0
        self._ear_buf: List[float] = []
        self._mar_buf: List[float] = []
        print(f"[Calibration] Profile '{driver_id}' loaded.")

    def _load_or_create(self, driver_id: str) -> DriverProfile:
        path = Path(settings.profile_dir) / f"{driver_id}.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return DriverProfile(**{k: v for k, v in data.items()
                                        if k in DriverProfile.__dataclass_fields__})
            except Exception:
                pass
        return DriverProfile(driver_id=driver_id)

    def save(self):
        path = Path(settings.profile_dir) / f"{self.driver_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self.profile), f, indent=2)

    def start_calibration(self):
        self._collecting = True
        self._start_time = time.time()
        self._ear_buf.clear()
        self._mar_buf.clear()
        print(f"[Calibration] Starting {settings.calibration_duration_sec}s baseline...")

    def feed(self, ear: float, mar: float) -> bool:
        """Returns True when calibration complete."""
        if not self._collecting:
            return False
        if ear > 0.1:
            self._ear_buf.append(ear)
        if mar < 0.9:
            self._mar_buf.append(mar)

        elapsed = time.time() - self._start_time
        if elapsed >= settings.calibration_duration_sec:
            self._finalize()
            return True
        return False

    def _finalize(self):
        if len(self._ear_buf) >= 10:
            base = float(np.percentile(self._ear_buf, 75))
            self.profile.ear_open_baseline = base
            self.profile.ear_threshold = base * 0.78  # 78% of open baseline
            print(f"  EAR baseline: {base:.3f} → threshold: {self.profile.ear_threshold:.3f}")
        if len(self._mar_buf) >= 10:
            base = float(np.percentile(self._mar_buf, 90))
            self.profile.mar_threshold = min(0.85, max(0.45, base * 1.5))
            print(f"  MAR threshold: {self.profile.mar_threshold:.3f}")

        self.profile.calibration_date = time.strftime("%Y-%m-%d %H:%M")
        self._collecting = False
        self.save()
        print("[Calibration] Complete.")
