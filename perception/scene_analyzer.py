import numpy as np
import cv2

class SceneAnalyzer:
    """
    Analyzes environment to contextually adjust detector weights.
    Uses frame ROI analysis — no external model needed.
    """
    def __init__(self):
        pass

    def analyze(self, frame: np.ndarray) -> dict:
        if frame is None or frame.size == 0:
            return {
                'ambient_brightness': 0.0,
                'contrast': 0.0,
                'blue_channel_high': False,
                'glare_detected': False,
                'lighting_class': 'day'
            }

        # Handle grayscale vs color
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blue_mean = frame[:,:,0].mean()
        else:
            gray = frame
            blue_mean = gray.mean()

        ambient_brightness = float(gray.mean())
        contrast = float(gray.std())
        
        return {
            'ambient_brightness': ambient_brightness,
            'contrast': contrast,
            'blue_channel_high': bool(blue_mean > 120),
            'glare_detected': self._detect_glare(gray),
            'lighting_class': self._classify_lighting(ambient_brightness, contrast),
        }

    def _detect_glare(self, gray: np.ndarray) -> bool:
        # Check if more than 2% of pixels are near max brightness (250+)
        saturated_pixels = np.sum(gray > 250)
        total_pixels = gray.size
        return (saturated_pixels / total_pixels) > 0.02

    def _classify_lighting(self, brightness: float, contrast: float) -> str:
        if brightness < 60:
            if contrast > 50:
                return 'tunnel'
            else:
                return 'night'
        elif brightness > 180:
            return 'day' # Bright daytime
        else:
            return 'day' # Default to day for normal lighting

    def get_weight_adjustments(self, scene: dict) -> dict:
        """Returns per-signal weight multipliers based on scene."""
        if scene.get('lighting_class') == 'night':
            return {'ear': 1.2, 'perclos': 1.3, 'rppg': 0.7}
        if scene.get('glare_detected'):
            return {'gaze': 0.6, 'ear': 1.1}
        return {}  # default weights
