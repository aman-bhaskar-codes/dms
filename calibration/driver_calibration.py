"""
Driver Calibration Module — Adaptive Per-Driver Baseline
NEW in V2.

Captures 30 seconds of normal driving to establish:
  - Personal EAR open baseline (varies ±15% between people)
  - Personal MAR neutral baseline
  - Head pose neutral position (some drivers naturally tilt)
  - Derives personalized alert thresholds

Profile is saved to JSON and auto-loaded on next session.

Usage:
  cal = DriverCalibration(profile_name="driver1")
  if not cal.load():
      cal.start()   # Enters calibration mode
      # ... call cal.update(ear, mar, pitch, yaw, roll) each frame
      # ... cal.is_complete() returns True after 30s
      cal.save()
  ear_threshold = cal.get_ear_threshold()
"""

import json
import os
import time
import numpy as np
from collections import deque
import config


class DriverCalibration:
    def __init__(self, profile_name: str = config.DEFAULT_DRIVER_PROFILE):
        self.profile_name  = profile_name
        self.profile_path  = os.path.join(
            config.CALIBRATION_PROFILE_DIR, f"{profile_name}.json")
        os.makedirs(config.CALIBRATION_PROFILE_DIR, exist_ok=True)

        self.calibrating      = False
        self.complete         = False
        self._start_time      = 0.0
        self._duration        = config.CALIBRATION_DURATION_SEC

        # Collection buffers
        self._ear_samples  = deque()
        self._mar_samples  = deque()
        self._pitch_samples= deque()
        self._yaw_samples  = deque()

        # Calibrated values
        self.ear_baseline    = config.EAR_OPEN_BASELINE
        self.ear_threshold   = config.EAR_THRESHOLD
        self.mar_baseline    = 0.20
        self.mar_threshold   = config.MAR_THRESHOLD
        self.pitch_neutral   = 0.0
        self.yaw_neutral     = 0.0
        self.profile_loaded  = False

        print(f"[Calibration] Profile: {profile_name}  "
              f"Path: {self.profile_path}")

    def load(self) -> bool:
        """Load saved profile. Returns True if successful."""
        if not os.path.exists(self.profile_path):
            print(f"[Calibration] No profile found at {self.profile_path}.")
            return False
        try:
            with open(self.profile_path, 'r') as f:
                p = json.load(f)
            self.ear_baseline  = p.get("ear_baseline",  config.EAR_OPEN_BASELINE)
            self.ear_threshold = p.get("ear_threshold", config.EAR_THRESHOLD)
            self.mar_baseline  = p.get("mar_baseline",  0.20)
            self.mar_threshold = p.get("mar_threshold", config.MAR_THRESHOLD)
            self.pitch_neutral = p.get("pitch_neutral", 0.0)
            self.yaw_neutral   = p.get("yaw_neutral",   0.0)
            self.complete       = True
            self.profile_loaded = True
            print(f"[Calibration] Loaded profile '{self.profile_name}'.")
            print(f"  EAR baseline={self.ear_baseline:.3f}  "
                  f"threshold={self.ear_threshold:.3f}")
            return True
        except Exception as e:
            print(f"[Calibration] Load failed: {e}")
            return False

    def save(self):
        """Save calibration profile to JSON."""
        profile = {
            "profile_name":  self.profile_name,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ear_baseline":  round(self.ear_baseline,  4),
            "ear_threshold": round(self.ear_threshold, 4),
            "mar_baseline":  round(self.mar_baseline,  4),
            "mar_threshold": round(self.mar_threshold, 4),
            "pitch_neutral": round(self.pitch_neutral, 2),
            "yaw_neutral":   round(self.yaw_neutral,   2),
            "samples":       len(self._ear_samples),
        }
        with open(self.profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        print(f"[Calibration] Profile saved to {self.profile_path}")

    def start(self):
        """Begin calibration capture."""
        self._ear_samples.clear()
        self._mar_samples.clear()
        self._pitch_samples.clear()
        self._yaw_samples.clear()
        self.calibrating = True
        self.complete    = False
        self._start_time = time.time()
        print(f"[Calibration] Starting {self._duration}s calibration... "
              "Look straight ahead, keep eyes open normally.")

    def update(self, ear: float, mar: float,
               pitch: float, yaw: float) -> float:
        """
        Feed a frame of data during calibration.
        Returns progress fraction 0.0–1.0.
        """
        if not self.calibrating:
            return 1.0 if self.complete else 0.0

        elapsed = time.time() - self._start_time

        # Only collect samples when driver appears alert (EAR > 0.22)
        if ear > 0.22:
            self._ear_samples.append(ear)
            self._mar_samples.append(mar)
            self._pitch_samples.append(pitch)
            self._yaw_samples.append(yaw)

        progress = min(1.0, elapsed / self._duration)

        if elapsed >= self._duration:
            self._finalize()

        return progress

    def _finalize(self):
        """Compute calibrated thresholds from collected samples."""
        if len(self._ear_samples) < 10:
            print("[Calibration] Not enough samples. Using defaults.")
            self.complete   = True
            self.calibrating = False
            return

        ears = np.array(list(self._ear_samples))
        mars = np.array(list(self._mar_samples))
        pitches = np.array(list(self._pitch_samples))
        yaws    = np.array(list(self._yaw_samples))

        # EAR: baseline = 75th percentile (typical open-eye value)
        # Threshold = 70% of baseline (PERCLOS standard)
        self.ear_baseline  = float(np.percentile(ears, 75))
        self.ear_threshold = round(self.ear_baseline * 0.78, 3)
        self.ear_threshold = max(0.18, min(0.30, self.ear_threshold))

        # MAR: baseline = 75th percentile
        self.mar_baseline  = float(np.percentile(mars, 75))
        self.mar_threshold = round(self.mar_baseline + 0.35, 3)
        self.mar_threshold = max(0.45, min(0.75, self.mar_threshold))

        # Neutral head pose
        self.pitch_neutral = float(np.median(pitches))
        self.yaw_neutral   = float(np.median(yaws))

        self.complete    = True
        self.calibrating = False

        print("[Calibration] Complete!")
        print(f"  EAR: baseline={self.ear_baseline:.3f}  "
              f"threshold={self.ear_threshold:.3f}")
        print(f"  MAR: baseline={self.mar_baseline:.3f}  "
              f"threshold={self.mar_threshold:.3f}")
        print(f"  Head neutral: pitch={self.pitch_neutral:.1f}°  "
              f"yaw={self.yaw_neutral:.1f}°")

    @property
    def progress(self) -> float:
        if not self.calibrating:
            return 1.0 if self.complete else 0.0
        return min(1.0, (time.time() - self._start_time) / self._duration)

    @property
    def is_complete(self) -> bool:
        return self.complete

    @property
    def status_message(self) -> str:
        if self.complete:
            return "Calibrated ✓"
        if self.calibrating:
            pct = int(self.progress * 100)
            return f"Calibrating... {pct}% — Look ahead, eyes open"
        return "Not calibrated"
