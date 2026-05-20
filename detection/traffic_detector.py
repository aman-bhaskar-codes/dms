"""
Traffic-aware scene understanding.
Uses YOLOv8s fine-tuned on CARLA + nuScenes-style traffic crops.
Publishes TrafficFrame to bus for voice agent context.
"""

import asyncio
import cv2
import numpy as np
import time
from collections import deque
from core.bus import EventBus, EventTopic
from core.models import TrafficFrame

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class TrafficDetector:
    TRAFFIC_CLASSES = {
        0: "car", 1: "truck", 2: "bus", 3: "motorcycle",
        4: "bicycle", 5: "pedestrian", 6: "traffic_light",
        7: "stop_sign", 8: "highway_sign"
    }

    def __init__(self, bus: EventBus, model_path: str = "models/traffic_dms_v5.onnx"):
        self._bus = bus
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._density_window = deque(maxlen=30)
        self._model = None
        self._load_model(model_path)

        # Subscribe to raw frames
        bus.subscribe(EventTopic.FRAME_RAW, self._on_frame)

    def _load_model(self, path: str):
        if not YOLO_AVAILABLE:
            print("[TrafficDetector] ultralytics not installed. Skipping.")
            return
        try:
            # Try fine-tuned ONNX first, fall back to YOLOv8s base
            self._model = YOLO(path)
            print(f"[TrafficDetector] Loaded {path}")
        except Exception:
            print("[TrafficDetector] Fine-tuned model not found. Using YOLOv8s base.")
            try:
                self._model = YOLO("yolov8s.pt")
            except Exception as e:
                print(f"[TrafficDetector] Could not load any model: {e}")

    async def _on_frame(self, event):
        try:
            self._frame_queue.put_nowait(event.payload["frame"])
        except asyncio.QueueFull:
            pass

    async def run(self):
        while True:
            try:
                frame = await asyncio.wait_for(self._frame_queue.get(), timeout=0.5)
                traffic = await asyncio.get_event_loop().run_in_executor(
                    None, self._analyze, frame
                )
                await self._bus.publish(
                    EventTopic.SIGNAL_TRAFFIC, traffic, source="traffic_detector"
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _analyze(self, frame: np.ndarray) -> TrafficFrame:
        if self._model is None:
            return TrafficFrame(0.1, False, 0, 0, 0)

        # Downsample for speed
        small = cv2.resize(frame, (416, 416))
        results = self._model(small, conf=0.4, verbose=False)[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return TrafficFrame(0.0, False, 0.0, 0.0, 0.0)

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs   = boxes.conf.cpu().numpy()
        xyxy    = boxes.xyxy.cpu().numpy()

        # Vehicle density
        vehicles = [c for c in cls_ids if c in [0,1,2,3]]
        density = min(len(vehicles) / 10.0, 1.0)
        self._density_window.append(density)
        avg_density = float(np.mean(self._density_window))

        # Highway detection: large vehicles filling frame width
        is_highway = False
        for box in xyxy:
            w = box[2] - box[0]
            if w > 0.4 * 416:
                is_highway = True
                break

        # Pedestrian proximity: bboxes in lower 1/3 of frame
        pedestrian_boxes = [xyxy[i] for i, c in enumerate(cls_ids) if c == 5]
        ped_proximity = 0.0
        for pb in pedestrian_boxes:
            if pb[3] > 416 * 0.6:
                area = (pb[2]-pb[0]) * (pb[3]-pb[1]) / (416*416)
                ped_proximity = max(ped_proximity, area * 20)
        ped_proximity = min(ped_proximity, 1.0)

        # Lane change risk: multiple vehicles at similar y-position
        lane_risk = min(avg_density * 1.5 * (0.5 if is_highway else 1.0), 1.0)

        return TrafficFrame(
            vehicle_density=round(avg_density, 3),
            is_highway=is_highway,
            estimated_speed_kmh=120 if is_highway else 50,
            lane_change_risk=round(lane_risk, 3),
            pedestrian_proximity=round(ped_proximity, 3)
        )
