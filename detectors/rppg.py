"""
rPPG V2 — Remote photoplethysmography (heart rate from camera)
Green channel FFT on forehead ROI. No hardware required.
Accuracy: ±10 BPM under good lighting.
"""
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt
from config import settings


class RPPGEstimator:
    def __init__(self):
        fps = settings.target_fps
        self._window_size = settings.rppg_window_sec * fps
        self._green_signal: deque = deque(maxlen=self._window_size)
        self.hr_bpm = 0.0
        self.hr_valid = False
        self._frame_count = 0

    def update(self, frame_bgr: np.ndarray, forehead_pts: np.ndarray) -> dict:
        if not settings.rppg_enabled or forehead_pts is None:
            return {"hr_bpm": 0.0, "hr_valid": False, "hr_state": "disabled"}

        # Extract forehead ROI mean green channel
        pts = forehead_pts.astype(int)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 + 5 or y2 <= y1 + 5:  # minimum 5px ROI
            return {"hr_bpm": self.hr_bpm, "hr_valid": False, "hr_state": "no_roi"}

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0 or np.any(np.isnan(roi)):
            return {"hr_bpm": self.hr_bpm, "hr_valid": False, "hr_state": "invalid_roi"}
            
        green_mean = float(np.mean(roi[:, :, 1]))  # Green channel
        self._green_signal.append(green_mean)
        self._frame_count += 1

        # Compute HR every 5 seconds
        if (self._frame_count % (settings.target_fps * 5) == 0
                and len(self._green_signal) >= self._window_size // 2):
            self.hr_bpm, self.hr_valid = self._compute_hr()

        hr_state = "normal"
        if self.hr_valid:
            if self.hr_bpm > settings.rppg_hr_stress_thresh:
                hr_state = "stress"
            elif self.hr_bpm < settings.rppg_hr_low_thresh:
                hr_state = "low"

        return {"hr_bpm": self.hr_bpm, "hr_valid": self.hr_valid, "hr_state": hr_state}

    def _compute_hr(self) -> tuple:
        signal = np.array(self._green_signal, dtype=np.float64)
        signal -= np.mean(signal)

        # Bandpass 0.7–3.0 Hz (42–180 BPM)
        fps = settings.target_fps
        try:
            b, a = butter(3, [0.7 / (fps / 2), 3.0 / (fps / 2)], btype='band')
            filtered = filtfilt(b, a, signal)
        except Exception:
            return 0.0, False

        fft = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)

        mask = (freqs >= 0.7) & (freqs <= 3.0)
        if not np.any(mask):
            return 0.0, False

        dominant = freqs[mask][np.argmax(fft[mask])]
        hr = dominant * 60.0
        valid = (settings.rppg_hr_min <= hr <= settings.rppg_hr_max
                 and np.max(fft[mask]) > np.mean(fft[mask]) * 2.0)
        return float(hr), valid
