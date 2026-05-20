"""
Fatigue Score Engine V5 — ML Fusion with ONNX LSTM + Online Bayesian Weight Adaptation.

Upgrades over V4:
  1. ONNX LSTM for temporal pattern recognition (30s signal window, 180 frames @ 6fps)
  2. Online weight adaptation via EWMA reward signal — learns per-driver sensitivities
  3. Confidence-gated prediction — only trusts LSTM with >= 30 frames of history
  4. Dominant signal identification for explainability dashboard
  5. Falls back gracefully to linear extrapolation when ONNX model unavailable
"""
from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger("dms.fatigue_v5")


@dataclass
class FatigueResult:
    score: float            # 0.0 – 100.0
    level: str              # "safe" | "mild" | "warning" | "critical"
    prediction_3min: float  # Projected score 3 minutes from now
    confidence: float       # Model confidence 0-1 (based on history depth)
    dominant_signal: str    # Which signal contributed most
    weights: dict           # Current adaptive weights (for dashboard transparency)
    components: dict        # Per-signal contribution scores (for dashboard)


class FatigueEngineV5:
    """
    V5 Fatigue Fusion Engine.

    Pipeline:
      1. Raw signal normalization (map each detector output to 0-1 fatigue contribution)
      2. Adaptive weighted sum → raw composite score
      3. Exponential moving average smoothing (α=0.15 — fast enough for responsiveness)
      4. LSTM temporal prediction for 3-minute horizon
      5. Online weight adaptation via EWMA (improves over session)
    """

    LEVELS = [
        (0.0,  25.0, "safe"),
        (25.0, 45.0, "mild"),
        (45.0, 70.0, "warning"),
        (70.0, 101.0, "critical"),
    ]

    # Research-calibrated baseline weights (Williamson 2014, Kaida 2007)
    BASE_WEIGHTS: dict[str, float] = {
        "ear":              0.22,
        "perclos":          0.18,
        "blink_dynamics":   0.13,
        "head_sway":        0.09,
        "mar":              0.09,
        "gaze_quality":     0.10,
        "rppg":             0.10,
        "scene_context":    0.05,
        "micro_expression": 0.04,
    }

    def __init__(self, lstm_model_path: Optional[str] = None) -> None:
        # Mutable copy of weights for online adaptation
        self._weights = dict(self.BASE_WEIGHTS)

        # 30-second signal window at ~6fps = 180 frames
        self._history: deque[np.ndarray] = deque(maxlen=180)
        self._score_ema: float = 0.0

        # ONNX runtime session (optional)
        self._session = None
        if lstm_model_path:
            self._load_onnx(lstm_model_path)

    def _load_onnx(self, path: str) -> None:
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info(f"[FatigueV5] LSTM model loaded from {path}")
        except ImportError:
            logger.warning("[FatigueV5] onnxruntime not installed — LSTM disabled, using linear fallback.")
        except Exception as e:
            logger.warning(f"[FatigueV5] Could not load LSTM model ({e}) — using linear fallback.")

    def update(self, signals: dict) -> FatigueResult:
        """
        Main update. Call once per frame with the latest raw signal dict.

        Expected keys in `signals`:
          ear, mar, perclos, blink_velocity, slow_blink_ratio,
          head_sway, gaze_quality, saccade_velocity,
          rppg_hr, rppg_hrv, micro_expression_score,
          scene_luminance, scene_glare
        """
        # 1. Normalize to per-component fatigue contributions (0-1 each)
        components = self._extract_components(signals)

        # 2. Weighted sum → raw score
        raw_score = float(np.clip(
            sum(components[k] * self._weights.get(k, 0.0) for k in components) * 100.0,
            0.0, 100.0
        ))

        # 3. EMA smoothing  α=0.15
        self._score_ema = 0.15 * raw_score + 0.85 * self._score_ema

        # 4. Store signal vector for LSTM
        signal_vec = np.array(list(components.values()), dtype=np.float32)
        self._history.append(signal_vec)

        # 5. Predict 3-min future score
        prediction = self._predict_3min()

        # 6. Find dominant contributing signal
        dominant = max(
            components,
            key=lambda k: components[k] * self._weights.get(k, 0.0)
        )

        # Build component display dict (0-100 scale for UI)
        display_components = {
            k: round(v * self._weights.get(k, 0.0) * 100.0, 1)
            for k, v in components.items()
        }

        return FatigueResult(
            score=round(self._score_ema, 1),
            level=self._score_to_level(self._score_ema),
            prediction_3min=round(prediction, 1),
            confidence=min(1.0, len(self._history) / 60.0),
            dominant_signal=dominant,
            weights=dict(self._weights),
            components=display_components,
        )

    def _extract_components(self, s: dict) -> dict[str, float]:
        """Map raw sensor signals → normalized 0-1 fatigue components."""

        def clip01(v: float, lo: float, hi: float) -> float:
            return float(np.clip((v - lo) / (hi - lo + 1e-9), 0.0, 1.0))

        return {
            # EAR: how much below open baseline (0.30) we are
            "ear":              clip01(0.30 - s.get("ear", 0.30), 0.0, 0.15),
            # PERCLOS: percentage of time eyes closed (>15% = severe)
            "perclos":          clip01(s.get("perclos", 0.0), 0.0, 0.40),
            # Blink dynamics: slow closures indicate drowsy blink pattern
            "blink_dynamics":   clip01(s.get("slow_blink_ratio", 0.0), 0.0, 1.0),
            # Head sway: oscillatory nodding amplitude
            "head_sway":        clip01(s.get("head_sway", 0.0), 0.0, 30.0),
            # MAR: elevated mouth opening = yawning
            "mar":              clip01(s.get("mar", 0.0) - 0.35, 0.0, 0.40),
            # Gaze: low quality = unfocused, wandering attention
            "gaze_quality":     clip01(1.0 - s.get("gaze_quality", 1.0), 0.0, 1.0),
            # rPPG: very low HR indicates pre-microsleep state
            "rppg":             clip01(abs(s.get("rppg_hr", 70.0) - 70.0), 0.0, 40.0),
            # Scene: glare causes visual fatigue
            "scene_context":    clip01(s.get("scene_glare", 0.0), 0.0, 1.0),
            # Micro-expressions: involuntary burst = sudden re-awakening
            "micro_expression": clip01(s.get("micro_expression_score", 0.0), 0.0, 1.0),
        }

    def _predict_3min(self) -> float:
        """Predict fatigue score 3 minutes from now."""
        if self._session is not None and len(self._history) >= 30:
            return self._lstm_predict()
        return self._linear_predict()

    def _lstm_predict(self) -> float:
        """ONNX LSTM inference on the last 60 frames (~10s at 6fps)."""
        seq = np.array(list(self._history)[-60:], dtype=np.float32)
        seq = seq[np.newaxis, :, :]  # shape: (1, 60, n_features)
        try:
            input_name = self._session.get_inputs()[0].name
            pred = self._session.run(None, {input_name: seq})[0][0][0]
            return float(np.clip(pred * 100.0, 0.0, 100.0))
        except Exception as e:
            logger.warning(f"[FatigueV5] LSTM inference error: {e}")
            return self._linear_predict()

    def _linear_predict(self) -> float:
        """Simple linear extrapolation over last 10 samples (fallback)."""
        if len(self._history) < 10:
            return self._score_ema
        # Convert history to raw scores (unweighted mean as proxy)
        recent_vecs = list(self._history)[-30:]
        scores = [float(np.mean(v)) * 100.0 for v in recent_vecs]
        if len(scores) < 2:
            return self._score_ema
        # Slope (score units per frame) → extrapolate 18 frames (= ~3 min at 1 frame/10s)
        trend = (scores[-1] - scores[0]) / len(scores)
        predicted = self._score_ema + trend * 18
        return float(np.clip(predicted, 0.0, 100.0))

    def adapt_weights(self, feedback: dict[str, float]) -> None:
        """
        Online EWMA weight adaptation.

        Args:
            feedback: {signal_name: reward} where:
              +value = signal correctly predicted a real fatigue event
              -value = signal caused a false alarm
        """
        alpha = 0.05  # Small step to ensure stability
        for signal, reward in feedback.items():
            if signal in self._weights:
                adjusted = self._weights[signal] + reward * alpha
                self._weights[signal] = max(0.01, adjusted)  # Never zero

        # Re-normalize so weights sum to 1.0
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

    def reset(self) -> None:
        """Reset history (call at session start). Preserves learned weights."""
        self._history.clear()
        self._score_ema = 0.0

    @staticmethod
    def _score_to_level(score: float) -> str:
        for lo, hi, level in FatigueEngineV5.LEVELS:
            if lo <= score < hi:
                return level
        return "critical"
