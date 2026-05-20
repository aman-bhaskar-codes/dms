"""
Vectorized Kalman Bank — processes all 478 MediaPipe landmarks simultaneously.
Uses diagonal covariance matrix math (element-wise operations) for speed.
~10x faster than 478 individual KalmanFilter1D instances.
"""
from __future__ import annotations

import numpy as np


class VectorizedKalmanBank:
    """
    Applies a 1D Kalman filter to every landmark coordinate simultaneously
    via NumPy vectorized operations.

    Process model: constant position (no motion model).
    Measurement model: direct observation with Gaussian noise.

    Args:
        n_landmarks: Number of MediaPipe landmarks (478 for FaceMesh).
        dims:        Coordinate dimensions per landmark (3 for x,y,z).
        process_noise:    Q — how much we expect the signal to wander per frame.
        measurement_noise: R — expected sensor/detector noise variance.
    """

    def __init__(
        self,
        n_landmarks: int = 478,
        dims: int = 3,
        process_noise: float = 0.005,
        measurement_noise: float = 0.05,
    ) -> None:
        n = n_landmarks * dims
        self._n = n
        self._n_landmarks = n_landmarks
        self._dims = dims

        # State vector (flattened landmark positions)
        self.x = np.zeros(n, dtype=np.float64)
        # Diagonal covariance (stored as vector — fully diagonal)
        self.P = np.ones(n, dtype=np.float64)
        # Fixed noise hyperparameters
        self.Q = np.full(n, process_noise, dtype=np.float64)
        self.R = np.full(n, measurement_noise, dtype=np.float64)

        self._initialized = False

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Args:
            landmarks: shape (n_landmarks, dims), float32 or float64.

        Returns:
            Filtered landmarks as float32 array, same shape as input.
        """
        z = landmarks.flatten().astype(np.float64)

        if not self._initialized:
            self.x = z.copy()
            self._initialized = True
            return landmarks  # Pass through on first frame

        # Predict step
        P_pred = self.P + self.Q                        # (n,)

        # Update step
        K = P_pred / (P_pred + self.R)                  # Kalman gain (n,)
        self.x = self.x + K * (z - self.x)              # State update
        self.P = (1.0 - K) * P_pred                     # Covariance update

        return self.x.reshape(self._n_landmarks, self._dims).astype(np.float32)

    def reset(self) -> None:
        """Reset filter state (e.g., after face is lost and re-acquired)."""
        self.x = np.zeros(self._n, dtype=np.float64)
        self.P = np.ones(self._n, dtype=np.float64)
        self._initialized = False
