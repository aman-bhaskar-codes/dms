"""
Ensemble rPPG: CHROM + POS algorithms averaged for noise robustness.
CHROM: de Haan & Jeanne (2013) - best for illumination changes
POS:   Wang et al. (2017)      - best for motion artifacts
"""

import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, welch
from dataclasses import dataclass


@dataclass
class RPPGResult:
    hr_bpm: float
    hrv_sdnn_ms: float
    stress_index: float
    signal_quality: float   # 0-1, based on SNR
    algorithm_used: str     # "ensemble" | "chrom" | "pos"


class RPPGEnsemble:
    FS = 30          # camera fps
    WINDOW_S = 10    # seconds of signal
    WINDOW_FRAMES = FS * WINDOW_S

    def __init__(self):
        # RGB trace buffers
        self._R = deque(maxlen=self.WINDOW_FRAMES)
        self._G = deque(maxlen=self.WINDOW_FRAMES)
        self._B = deque(maxlen=self.WINDOW_FRAMES)
        self._ibi_history = deque(maxlen=20)  # inter-beat intervals

    def update(self, forehead_roi: np.ndarray) -> RPPGResult:
        """
        forehead_roi: (H, W, 3) BGR numpy array
        Returns RPPGResult or None if window not full.
        """
        # Mean spatial pooling
        r = float(forehead_roi[:, :, 2].mean())
        g = float(forehead_roi[:, :, 1].mean())
        b = float(forehead_roi[:, :, 0].mean())

        self._R.append(r); self._G.append(g); self._B.append(b)

        if len(self._R) < self.WINDOW_FRAMES:
            return RPPGResult(0, 0, 0, 0, "warming_up")

        R = np.array(self._R); G = np.array(self._G); B = np.array(self._B)

        # ── CHROM algorithm ──────────────────────────────────────
        Rn = R / (R.mean() + 1e-8); Gn = G / (G.mean() + 1e-8); Bn = B / (B.mean() + 1e-8)
        Xs = 3*Rn - 2*Gn
        Ys = 1.5*Rn + Gn - 1.5*Bn
        alpha = Xs.std() / (Ys.std() + 1e-8)
        chrom_signal = Xs - alpha * Ys

        # ── POS algorithm ────────────────────────────────────────
        C = np.stack([Rn, Gn, Bn], axis=0)
        mean_color = C.mean(axis=1, keepdims=True)
        Cn = C / (mean_color + 1e-8)
        P = np.array([[0, 1, -1], [-2, 1, 1]])
        S = P @ Cn
        std1, std2 = S[0].std() + 1e-8, S[1].std() + 1e-8
        pos_signal = S[0] / std1 + S[1] / std2

        # ── Bandpass 0.7–4.0 Hz (42–240 bpm) ───────────────────
        b_coef, a_coef = butter(3, [0.7/(self.FS/2), 4.0/(self.FS/2)], btype="band")
        chrom_f = filtfilt(b_coef, a_coef, chrom_signal)
        pos_f   = filtfilt(b_coef, a_coef, pos_signal)

        # ── HR from ensemble ────────────────────────────────────
        ensemble = (chrom_f / (chrom_f.std() + 1e-8) +
                    pos_f  / (pos_f.std()   + 1e-8)) / 2.0

        freqs, psd = welch(ensemble, fs=self.FS, nperseg=min(256, len(ensemble)))
        mask = (freqs >= 0.7) & (freqs <= 4.0)
        if mask.sum() == 0:
            return RPPGResult(72, 40, 0.3, 0.1, "ensemble")

        peak_freq = freqs[mask][psd[mask].argmax()]
        hr_bpm = peak_freq * 60.0

        # ── Signal quality (SNR proxy) ───────────────────────────
        peak_power = psd[mask].max()
        noise_power = psd[~mask].mean() + 1e-8
        snr = 10 * np.log10(peak_power / noise_power)
        quality = float(np.clip((snr - 5) / 20, 0, 1))

        # ── HRV: std of IBI ─────────────────────────────────────
        # IBI estimated from HR (placeholder — upgrade with peak detection)
        ibi_ms = (60.0 / (hr_bpm + 1e-8)) * 1000
        self._ibi_history.append(ibi_ms)
        hrv_sdnn = float(np.std(self._ibi_history)) if len(self._ibi_history) > 5 else 40.0

        # ── Stress index (low HRV + high HR = high stress) ──────
        norm_hr = np.clip((hr_bpm - 60) / 60, 0, 1)
        norm_hrv_inv = np.clip(1.0 - hrv_sdnn / 80, 0, 1)
        stress = float(0.5 * norm_hr + 0.5 * norm_hrv_inv)

        return RPPGResult(
            hr_bpm=round(hr_bpm, 1),
            hrv_sdnn_ms=round(hrv_sdnn, 1),
            stress_index=round(stress, 3),
            signal_quality=round(quality, 3),
            algorithm_used="ensemble"
        )
