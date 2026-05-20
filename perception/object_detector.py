"""
Object Detector V2 — YOLOv8 in Dedicated Background Thread

V2: Moves YOLO inference to a worker thread so the main camera loop
never blocks. The thread always processes the latest available frame.

Main loop posts frames via a queue (maxsize=1, drops old frames).
Results are read from a results queue — always uses latest inference.
"""

import numpy as np
import cv2
import threading
import queue
from typing import List, Dict, Tuple
import config

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ObjectDetector:
    def __init__(self):
        self.enabled         = config.YOLO_ENABLED and YOLO_AVAILABLE
        self.model           = None
        self.phone_detected  = False
        self.phone_conf      = 0.0
        self.last_detections: List[Dict] = []

        # Threading
        self._frame_q  = queue.Queue(maxsize=1)
        self._result_q = queue.Queue(maxsize=2)
        self._running  = False
        self._thread   = None

        if self.enabled:
            print("[ObjectDetector V2] Loading YOLOv8 (background thread)...")
            try:
                self.model = YOLO(config.YOLO_MODEL)
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                self.model(dummy, verbose=False)
                self._start_thread()
                print("[ObjectDetector V2] YOLOv8 ready in background thread.")
            except Exception as e:
                print(f"[ObjectDetector V2] Load failed: {e}")
                self.enabled = False
        else:
            print("[ObjectDetector V2] Disabled.")

    def _start_thread(self):
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Background thread: continuously reads frames and runs inference."""
        while self._running:
            try:
                frame = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                results = self.model(
                    frame, verbose=False,
                    conf=config.YOLO_CONFIDENCE,
                    classes=list(config.YOLO_CLASSES.keys()),
                )
                detections = []
                phone_found = False
                phone_conf  = 0.0

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf   = float(box.conf[0])
                        bbox   = box.xyxy[0].tolist()
                        if cls_id in config.YOLO_CLASSES:
                            detections.append({
                                "class": config.YOLO_CLASSES[cls_id],
                                "conf":  conf,
                                "bbox":  bbox,
                                "class_id": cls_id,
                            })
                            if cls_id == 67:
                                phone_found = True
                                phone_conf  = max(phone_conf, conf)

                # Drop old result, put new one
                try:
                    self._result_q.get_nowait()
                except queue.Empty:
                    pass
                self._result_q.put((phone_found, phone_conf, detections))

            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"[ObjectDetector V2] Worker error: {e}")

    def update(self, frame_bgr: np.ndarray) -> Tuple[bool, List[Dict]]:
        """Submit frame to worker thread and return latest results (non-blocking)."""
        if not self.enabled:
            return False, []

        # Submit frame (drop old if queue full)
        try:
            self._frame_q.put_nowait(frame_bgr.copy())
        except queue.Full:
            pass  # Worker is busy; results will come from previous frame

        # Read latest results (non-blocking)
        try:
            phone_found, phone_conf, detections = self._result_q.get_nowait()
            self.phone_detected  = phone_found
            self.phone_conf      = phone_conf
            self.last_detections = detections
        except queue.Empty:
            pass  # Use cached results

        return self.phone_detected, self.last_detections

    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        for det in self.last_detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label = f"{det['class']} {det['conf']:.2f}"
            color = config.COLOR_CRITICAL if det["class"] == "phone" else config.COLOR_WARN
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + len(label) * 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
        return frame

    def release(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.model = None
        print("[ObjectDetector V2] Released.")
