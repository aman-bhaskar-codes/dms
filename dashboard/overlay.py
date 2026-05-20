"""
OpenCV HUD Overlay V4.
Renders minimal telemetry onto the frame for testing/fallback if web UI isn't used.
"""
import cv2
import numpy as np
from typing import Dict
from config import settings


class HUDOverlay:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        h, w = frame.shape[:2]
        out = frame.copy()

        # Score box (top right)
        score = metrics.get("fatigue_score", 0.0)
        level = metrics.get("fatigue_level", "normal")
        color = self._get_color(level)
        
        cv2.rectangle(out, (w - 200, 10), (w - 10, 80), (0, 0, 0), -1)
        cv2.putText(out, f"FATIGUE: {score:.1f}", (w - 190, 40), 
                    self.font, 0.7, color, 2)
        cv2.putText(out, level.upper(), (w - 190, 70), 
                    self.font, 0.6, color, 2)

        # Heart Rate (top left)
        hr = metrics.get("hr_bpm", 0.0)
        hr_valid = metrics.get("hr_valid", False)
        if hr_valid:
            cv2.putText(out, f"HR: {hr:.0f} BPM", (20, 40), 
                        self.font, 0.7, (0, 255, 0), 2)

        # Status text
        y = h - 60
        status_items = [
            f"EAR: {metrics.get('ear', 0.0):.2f}",
            f"MAR: {metrics.get('mar', 0.0):.2f}",
            f"PERCLOS: {metrics.get('perclos', 0.0):.2f}",
        ]
        
        cv2.rectangle(out, (10, y - 25), (w - 10, h - 10), (0, 0, 0), -1)
        text = " | ".join(status_items)
        cv2.putText(out, text, (20, y), self.font, 0.5, (255, 255, 255), 1)

        # Draw YOLO detections
        detections = metrics.get("detections", [])
        for det in detections:
            bbox = det["bbox"]
            name = det["class_name"]
            conf = det["confidence"]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(out, f"{name} {conf:.2f}", (x1, y1 - 10), 
                        self.font, 0.5, (0, 165, 255), 2)

        return out

    def _get_color(self, level: str) -> tuple:
        if level == "critical": return (0, 0, 255)    # Red
        if level == "warning": return (0, 165, 255)   # Orange
        if level == "mild": return (0, 255, 255)      # Yellow
        return (0, 255, 0)                            # Green
