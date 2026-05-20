"""
rPPG — Remote Photoplethysmography (Camera-Based Heart Rate)
NEW in V2.

Algorithm:
  1. Extract forehead/cheek ROI from facial landmarks
  2. Track mean green channel (G) over rolling 10-second window
  3. Detrend signal with polynomial fit to remove slow illumination drift
  4. Bandpass filter: 0.7–3.0 Hz (= 42–180 BPM)
  5. FFT → find dominant frequency → BPM

Why green channel?
  Blood absorbs green light (550nm) most strongly. The cardiac pulse
  causes ~1% intensity variation in skin pixels — imperceptible to eye
  but detectable by camera over 300 frames.

Accuracy: ±5–8 BPM under stable lighting, no physical activity.
Useful signal: elevated HR (stress, fear) or very low HR (microsleep).

No additional hardware required.
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple
import time
import config

try:
    from scipy import signal as sp_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _bandpass_filter(data: np.ndarray, fps: float,
                     low_hz: float, high_hz: float) -> np.ndarray:
    """Butterworth bandpass filter, or simple FFT mask if scipy unavailable."""
    if SCIPY_AVAILABLE and len(data) >= 6:
        nyq  = fps / 2.0
        low  = max(low_hz / nyq, 0.01)
        high = min(high_hz / nyq, 0.99)
        try:
            b, a = sp_signal.butter(2, [low, high], btype='band')
            return sp_signal.filtfilt(b, a, data)
        except Exception:
            pass
    # Fallback: zero-phase moving average (rough)
    k = max(1, int(fps / high_hz))
    return np.convolve(data, np.ones(k) / k, mode='same')


class RPPGEstimator:
    """
    Real-time rPPG heart rate estimator using green channel from face ROI.
    """

    def __init__(self, fps: int = config.TARGET_FPS):
        self.enabled    = config.RPPG_ENABLED
        self.fps        = fps
        self.window_sec = config.RPPG_WINDOW_SEC
        self.buf_size   = fps * self.window_sec

        # Raw green signal buffer
        self._green_buf = deque(maxlen=self.buf_size)
        self._time_buf  = deque(maxlen=self.buf_size)

        self.hr_bpm     = 0.0
        self.hr_confidence = 0.0   # 0..1 — spectral peak prominence
        self.state      = "unknown"  # "normal" | "elevated" | "low" | "unknown"

        self._last_hr_update = 0.0
        self._hr_history = deque(maxlen=10)  # for smoothing

        if not self.enabled:
            print("[rPPG] Disabled in config.")
        else:
            print(f"[rPPG] Enabled. Window={self.window_sec}s  "
                  f"Range={config.RPPG_HR_MIN}–{config.RPPG_HR_MAX} BPM")
            if not SCIPY_AVAILABLE:
                print("  [rPPG] scipy not found — using fallback filter (lower accuracy)")

    def update(self, frame_bgr: np.ndarray,
               forehead_pts: Optional[np.ndarray]) -> Tuple[float, float, str]:
        """
        Extract green channel from forehead ROI and update HR estimate.

        Args:
            frame_bgr:    Full BGR frame
            forehead_pts: (N, 2) pixel coords of forehead landmark convex hull

        Returns:
            (hr_bpm, confidence, state)
        """
        if not self.enabled:
            return 0.0, 0.0, "disabled"

        green = self._extract_green(frame_bgr, forehead_pts)
        if green is None:
            return self.hr_bpm, self.hr_confidence, self.state

        self._green_buf.append(green)
        self._time_buf.append(time.time())

        # Only compute HR every second (save CPU)
        now = time.time()
        if (now - self._last_hr_update < 1.0
                or len(self._green_buf) < self.buf_size // 2):
            return self.hr_bpm, self.hr_confidence, self.state

        self._last_hr_update = now
        self._estimate_hr()

        return self.hr_bpm, self.hr_confidence, self.state

    def _extract_green(self, frame_bgr: np.ndarray,
                        pts: Optional[np.ndarray]) -> Optional[float]:
        """Extract mean green channel from forehead ROI."""
        if pts is None or len(pts) < 3:
            # Fallback: use central strip of frame (approximate forehead)
            h, w = frame_bgr.shape[:2]
            roi = frame_bgr[int(h * 0.05):int(h * 0.25),
                            int(w * 0.30):int(w * 0.70)]
        else:
            # Build convex hull mask from landmark points
            hull_pts = pts.astype(np.int32)
            hull     = cv2_convex_hull(hull_pts)
            if hull is None:
                return None
            mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            import cv2
            cv2.fillConvexPoly(mask, hull, 255)
            roi = frame_bgr[mask > 0]
            if len(roi) < 10:
                return None
            return float(np.mean(roi[:, 1]))  # green channel

        if roi.size == 0:
            return None
        return float(np.mean(roi[:, :, 1]))  # green channel

    def _estimate_hr(self):
        """Run FFT on green signal buffer to estimate HR."""
        signal_raw = np.array(list(self._green_buf), dtype=np.float64)
        n = len(signal_raw)
        if n < 30:
            return

        # Detrend: remove slow drift with linear fit
        x = np.arange(n)
        poly = np.polyfit(x, signal_raw, 2)
        trend = np.polyval(poly, x)
        detrended = signal_raw - trend

        # Normalize
        std = detrended.std()
        if std < 1e-6:
            return
        detrended /= std

        # Actual fps from timestamps
        times = list(self._time_buf)
        if len(times) >= 2:
            actual_fps = len(times) / (times[-1] - times[0] + 1e-6)
        else:
            actual_fps = self.fps

        # Bandpass filter
        filtered = _bandpass_filter(
            detrended, actual_fps,
            config.RPPG_HR_MIN / 60.0,
            config.RPPG_HR_MAX / 60.0,
        )

        # FFT
        fft_vals = np.abs(np.fft.rfft(filtered))
        freqs    = np.fft.rfftfreq(n, d=1.0 / actual_fps)

        # Restrict to valid HR band
        lo = config.RPPG_HR_MIN / 60.0
        hi = config.RPPG_HR_MAX / 60.0
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return

        fft_band = fft_vals[mask]
        freq_band = freqs[mask]

        peak_idx = np.argmax(fft_band)
        peak_freq = freq_band[peak_idx]
        peak_val  = fft_band[peak_idx]

        # Confidence: ratio of peak power to band power
        confidence = float(peak_val / (np.sum(fft_band) + 1e-6))
        confidence = min(1.0, confidence * 3.0)  # scale to 0..1

        hr = peak_freq * 60.0
        if config.RPPG_HR_MIN <= hr <= config.RPPG_HR_MAX and confidence > 0.15:
            self._hr_history.append(hr)
            self.hr_bpm = float(np.median(list(self._hr_history)))
            self.hr_confidence = confidence
        else:
            self.hr_confidence *= 0.9  # Decay confidence if no good peak

        # State classification
        if self.hr_confidence > 0.2:
            if self.hr_bpm >= config.RPPG_HR_STRESS_THRESH:
                self.state = "elevated"
            elif self.hr_bpm <= config.RPPG_HR_LOW_THRESH:
                self.state = "low"
            else:
                self.state = "normal"
        else:
            self.state = "unknown"

    def fatigue_components(self) -> dict:
        """Return rPPG-based fatigue signal."""
        if self.hr_confidence < 0.2 or self.hr_bpm <= 0:
            return {"hr_score": 0.0}
        # Low HR can indicate approaching microsleep
        if self.hr_bpm < config.RPPG_HR_LOW_THRESH:
            score = min(1.0, (config.RPPG_HR_LOW_THRESH - self.hr_bpm) / 20.0)
        else:
            score = 0.0
        return {"hr_score": score * self.hr_confidence}

    def reset(self):
        self._green_buf.clear()
        self._time_buf.clear()
        self.hr_bpm = 0.0
        self.hr_confidence = 0.0
        self.state = "unknown"


def cv2_convex_hull(pts: np.ndarray):
    """Thin wrapper to handle OpenCV convex hull for rPPG ROI."""
    import cv2
    try:
        hull = cv2.convexHull(pts)
        if hull is not None and len(hull) >= 3:
            return hull.reshape(-1, 2)
    except Exception:
        pass
    return None
