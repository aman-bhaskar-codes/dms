"""YOLOv8 Object Detector — background thread, doesn't block main loop"""
import threading
import time
import numpy as np
from typing import List, Dict, Optional
from config import settings


class ObjectDetector:
    def __init__(self):
        self._model = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_detections: List[Dict] = []
        self._running = False

        if settings.yolo_enabled:
            self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(settings.yolo_model)
            print(f"[YOLO] Model loaded: {settings.yolo_model}")
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}. YOLO disabled.")
            self._model = None

    def start(self):
        if self._model is None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def _inference_loop(self):
        while self._running:
            with self._lock:
                frame = self._latest_frame.copy() if self._latest_frame is not None else None
            if frame is not None:
                results = self._model(frame, verbose=False,
                                      conf=settings.yolo_confidence)
                detections = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id in settings.YOLO_CLASSES:
                            detections.append({
                                "class_id": cls_id,
                                "class_name": settings.YOLO_CLASSES[cls_id],
                                "confidence": float(box.conf[0]),
                                "bbox": box.xyxy[0].tolist(),
                            })
                with self._lock:
                    self._latest_detections = detections
            time.sleep(0.1)  # 10 fps inference — not blocking camera loop

    def submit_frame(self, frame: np.ndarray):
        if self._model is None:
            return
        with self._lock:
            self._latest_frame = frame

    def get_detections(self) -> List[Dict]:
        with self._lock:
            return self._latest_detections.copy()

    def stop(self):
        self._running = False
