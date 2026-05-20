"""
Gaze Tracker V2 — Fixation Detection + Attention Heatmap

V2 additions:
  - Fixation detection: stable gaze for 15+ frames = proper visual fixation
  - Attention heatmap: 80×60 accumulator showing gaze history
  - Heatmap entropy: low entropy = driver focused ahead; high = scattered
  - Saccade detection: rapid gaze jumps
"""

import numpy as np
from collections import deque
from typing import Tuple
import cv2
import config


class GazeTracker:
    def __init__(self):
        self.gaze_x    = 0.0
        self.gaze_y    = 0.0
        self.direction = "center"
        self.counter_away = 0

        # Smoothing
        self.gaze_x_hist = deque(maxlen=8)
        self.gaze_y_hist = deque(maxlen=8)

        # V2: Fixation tracking
        self._fix_x    = 0.0
        self._fix_y    = 0.0
        self._fix_dur  = 0             # frames at current fixation
        self.fixation  = False         # True = stable gaze
        self.fixation_quality = 0.0   # 0..1

        # V2: Attention heatmap
        self._heatmap  = np.zeros((config.HEATMAP_HEIGHT, config.HEATMAP_WIDTH),
                                   dtype=np.float32)
        self._heatmap_decay = config.HEATMAP_DECAY

        # V2: Saccade detection
        self._saccade_analyzer = SaccadeAnalyzer()
        self._prev_gx  = 0.0
        self._prev_gy  = 0.0

        print(f"[GazeTracker V2] thresh={config.GAZE_THRESH}  "
              f"heatmap={config.HEATMAP_WIDTH}×{config.HEATMAP_HEIGHT}")

    def update(
        self,
        left_iris_pts:   np.ndarray,
        right_iris_pts:  np.ndarray,
        left_eye_corners:  np.ndarray,
        right_eye_corners: np.ndarray,
    ) -> Tuple[float, float, str, bool]:

        def eye_gaze(iris_pts, corners):
            iris_c  = np.mean(iris_pts, axis=0)
            mid     = np.mean(corners, axis=0)
            width   = np.linalg.norm(corners[0] - corners[1])
            return (iris_c - mid) / (width + 1e-6)

        lg = eye_gaze(left_iris_pts,  left_eye_corners)
        rg = eye_gaze(right_iris_pts, right_eye_corners)
        gaze = (lg + rg) / 2.0

        self.gaze_x_hist.append(float(gaze[0]))
        self.gaze_y_hist.append(float(gaze[1]))
        sx = float(np.mean(self.gaze_x_hist))
        sy = float(np.mean(self.gaze_y_hist))
        self.gaze_x, self.gaze_y = sx, sy

        # Direction classification
        thresh = config.GAZE_THRESH
        if abs(sx) < thresh and abs(sy) < thresh:
            direction = "center"
        elif abs(sx) >= abs(sy):
            direction = "right" if sx > 0 else "left"
        else:
            direction = "down" if sy > 0 else "up"
        self.direction = direction

        # Saccade detection
        self._saccade_analyzer.add_gaze(sx, sy)
        self._prev_gx, self._prev_gy = sx, sy

        # Fixation detection
        dist_from_fix = np.sqrt((sx - self._fix_x)**2 + (sy - self._fix_y)**2)
        if dist_from_fix < 0.08:
            self._fix_dur += 1
        else:
            self._fix_x, self._fix_y = sx, sy
            self._fix_dur = 0

        self.fixation = self._fix_dur >= config.FIXATION_MIN_FRAMES
        self.fixation_quality = min(1.0, self._fix_dur / (config.FIXATION_MIN_FRAMES * 3))

        # Update heatmap
        self._update_heatmap(sx, sy)

        # Distraction counter
        is_distracted = False
        if direction != "center":
            self.counter_away += 1
            if self.counter_away >= config.GAZE_CONSEC_FRAMES:
                is_distracted = True
        else:
            self.counter_away = max(0, self.counter_away - 2)

        return sx, sy, direction, is_distracted

    def _update_heatmap(self, gx: float, gy: float):
        """Map normalized gaze coords to heatmap and accumulate."""
        # Decay
        self._heatmap *= self._heatmap_decay
        # Map gaze [-0.5..0.5] to heatmap pixel
        px = int(np.clip((gx + 0.5) * config.HEATMAP_WIDTH,  0, config.HEATMAP_WIDTH  - 1))
        py = int(np.clip((gy + 0.5) * config.HEATMAP_HEIGHT, 0, config.HEATMAP_HEIGHT - 1))
        # Gaussian splat
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = px + dx, py + dy
                if 0 <= nx < config.HEATMAP_WIDTH and 0 <= ny < config.HEATMAP_HEIGHT:
                    self._heatmap[ny, nx] += np.exp(-(dx*dx + dy*dy) / 2.0)
        np.clip(self._heatmap, 0, 255, out=self._heatmap)

    @property
    def heatmap_entropy(self) -> float:
        """
        Entropy of normalized attention heatmap (0..1).
        Low = focused ahead. High = scattered gaze = distraction.
        """
        h = self._heatmap.copy()
        total = h.sum()
        if total < 1e-6:
            return 0.0
        p = h / total
        p = p[p > 1e-10]
        return float(-np.sum(p * np.log2(p)) / np.log2(h.size))

    def get_heatmap_image(self, width: int = 80, height: int = 60) -> np.ndarray:
        """Return heatmap as BGR color image for display."""
        norm = cv2.normalize(self._heatmap, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        if width != config.HEATMAP_WIDTH or height != config.HEATMAP_HEIGHT:
            colored = cv2.resize(colored, (width, height))
        return colored

    def fatigue_components(self) -> dict:
        saccade_stats = self._saccade_analyzer.analyze()
        return {
            "gaze_entropy":   self.heatmap_entropy,
            "away_fraction":  min(1.0, self.counter_away / config.GAZE_CONSEC_FRAMES),
            "low_fixation":   1.0 - self.fixation_quality,
            "slow_saccade_pct": saccade_stats.get('slow_saccade_pct', 0.0),
        }

    def draw_gaze_indicator(self, frame: np.ndarray,
                             center: Tuple[int, int], radius: int = 40) -> np.ndarray:
        cx, cy = center
        color = config.COLOR_NORMAL if self.direction == "center" else config.COLOR_WARN
        cv2.circle(frame, (cx, cy), radius, color, 1)
        dot_x = int(cx + self.gaze_x * radius * 0.7)
        dot_y = int(cy + self.gaze_y * radius * 0.7)
        cv2.circle(frame, (dot_x, dot_y), 6, config.COLOR_IRIS, -1)
        cv2.line(frame, (cx - radius, cy), (cx + radius, cy), (60,60,60), 1)
        cv2.line(frame, (cx, cy - radius), (cx, cy + radius), (60,60,60), 1)
        return frame

    def reset(self):
        self.counter_away = 0
        self.gaze_x_hist.clear()
        self.gaze_y_hist.clear()
        self._heatmap[:] = 0
        self._saccade_analyzer = SaccadeAnalyzer()

class SaccadeAnalyzer:
    """
    Computes saccade velocity from successive gaze positions.
    Drowsy saccades are slower (< 200°/s vs normal 400-600°/s).
    """
    def __init__(self):
        self.gaze_history = deque(maxlen=config.TARGET_FPS)
        
    def add_gaze(self, gaze_x: float, gaze_y: float):
        self.gaze_history.append((gaze_x, gaze_y))

    def analyze(self) -> dict:
        if len(self.gaze_history) < 2:
            return {'mean_velocity': 0.0, 'slow_saccade_pct': 0.0}
            
        velocities = []
        for i in range(1, len(self.gaze_history)):
            dx = self.gaze_history[i][0] - self.gaze_history[i-1][0]
            dy = self.gaze_history[i][1] - self.gaze_history[i-1][1]
            dist = np.sqrt(dx**2 + dy**2)
            # Rough conversion from normalized gaze to degrees
            # (Assuming -0.5 to 0.5 covers roughly 60 degrees field of view)
            deg_per_frame = dist * 60.0
            vel_deg_per_sec = deg_per_frame * config.TARGET_FPS
            if vel_deg_per_sec > 30.0:  # Only count actual saccades, not drift
                velocities.append(vel_deg_per_sec)
                
        if not velocities:
            return {'mean_velocity': 0.0, 'slow_saccade_pct': 0.0}
            
        mean_vel = np.mean(velocities)
        slow_pct = np.mean([1 if v < config.SACCADE_SLOW_THRESH else 0 for v in velocities])
        
        return {
            'mean_velocity': mean_vel,
            'slow_saccade_pct': slow_pct
        }
