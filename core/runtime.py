"""Supervised local application lifecycle and safety-first data flow."""

import asyncio
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from agents.safety_agent import SafetyPolicy
from ai.assistant import AssistantService
from alerts.alert_engine import AlertEngine
from core.bus import EventBus, EventTopic
from core.config import ROOT, Settings
from core.contracts import MonitoringSnapshot
from core.monitoring import Monitor
from memory.store import Store
from rag.knowledge import KnowledgeBase
from voice.local import LocalVoice


class Runtime:
    def __init__(self, config=None):
        self.config = config or Settings()
        self.store = Store(self.config.db_path)
        self.monitor = Monitor(self.config)
        self.policy = SafetyPolicy(self.config)
        self.audio = AlertEngine(self.config.audio_enabled)
        self.voice = LocalVoice(self.config.whisper_path)
        self.bus = EventBus()
        self.knowledge = KnowledgeBase(
            self.config.knowledge_dir,
            self.config.embedding_path,
            self.config.data_dir / "knowledge-index",
        )
        self.assistant = AssistantService(
            self.store,
            self.knowledge,
            self.config.gemini_api_key.get_secret_value(),
            self.config.gemini_model,
            self.config.gemini_daily_cap,
            self.config.gemini_interval_s,
        )
        self.instance_lock = None
        self.session_id = None
        self.snapshot = None
        self.last_alert = None
        self.component_errors = {}
        self.session_task = None
        self.bus_task = None
        self.recorder_task = None
        self.maintenance_task = None
        self.knowledge_task = None
        self.records = asyncio.Queue(maxsize=256)
        self.last_recorded = -float("inf")
        self.calibration_saved = False
        self.lock = asyncio.Lock()
        self.latencies = deque(maxlen=3600)
        self.alert_latencies = deque(maxlen=1000)
        self.frame_count = 0
        self.started_at = None
        self.preview = None
        self.preview_version = 0

    async def open(self):
        import fcntl

        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.instance_lock = (self.config.data_dir / ".runtime.lock").open("a")
        try:
            fcntl.flock(self.instance_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.instance_lock.close()
            self.instance_lock = None
            raise RuntimeError("DMS is already running with this data directory") from error
        try:
            await self.store.connect()
        except BaseException:
            self.instance_lock.close()
            self.instance_lock = None
            raise
        await self.store.cleanup(self.config.retention_days)
        self.knowledge.ingest()
        self.audio.start()
        self.bus_task = asyncio.create_task(self.bus.run())
        self.recorder_task = asyncio.create_task(self._record())
        self.maintenance_task = asyncio.create_task(self._maintenance())
        if self.config.embedding_path.is_dir():
            self.knowledge.status = "lexical; dense model loading"
            self.knowledge_task = asyncio.create_task(
                asyncio.to_thread(self.knowledge.enable_dense)
            )

    async def _maintenance(self):
        cleanup_at = time.monotonic()
        while True:
            await asyncio.sleep(0.5)
            if self.session_id and self.snapshot and self.snapshot.state != "unavailable":
                if time.time() - self.snapshot.processed_at > self.config.max_frame_age_s:
                    await self.unavailable("camera_or_inference_stalled")
            if time.monotonic() - cleanup_at > 86400:
                await self.store.cleanup(self.config.retention_days)
                cleanup_at = time.monotonic()

    async def _record(self):
        while True:
            batch = [await self.records.get()]
            while len(batch) < 32 and not self.records.empty():
                batch.append(self.records.get_nowait())
            try:
                await self.store.record_batch(batch)
            except Exception:
                self.component_errors["storage"] = (
                    "Session recording failed; stop and check free disk space"
                )
                if self.session_task:
                    self.session_task.cancel()
                self.snapshot = MonitoringSnapshot(
                    session_id=self.session_id,
                    frame_id=0,
                    captured_at=time.time(),
                    state="unavailable",
                    reasons=["storage_failure"],
                )
            finally:
                for _ in batch:
                    self.records.task_done()

    def _enqueue_record(self, kind, value):
        try:
            self.records.put_nowait((kind, value))
        except asyncio.QueueFull as error:
            raise RuntimeError("Session recorder backlog exceeded its bound") from error

    async def start(self, mode="live"):
        async with self.lock:
            if self.session_id:
                raise ValueError("A session is already running")
            self.monitor = Monitor(self.config)
            self.policy = SafetyPolicy(self.config)
            self.audio.enabled = self.config.audio_enabled
            self.audio.silence()
            self.last_alert = None
            self.preview = None
            self.component_errors.pop("monitoring", None)
            calibration = await self.store.calibration()
            if calibration:
                self.monitor.ear_threshold = calibration["ear_threshold"]
                self.monitor.baseline_ear = calibration["baseline_ear"]
            self.session_id = await self.store.start_session(mode=mode)
            self.monitor.session_id = self.session_id
            self.last_recorded = -float("inf")
            self.frame_count = 0
            self.latencies.clear()
            self.started_at = time.monotonic()
            self.snapshot = MonitoringSnapshot(
                session_id=self.session_id,
                frame_id=0,
                captured_at=time.time(),
                state="initializing",
                synthetic=mode == "replay",
            )
            self.session_task = asyncio.create_task(self._session(mode))
            return self.session_id

    async def _session(self, mode):
        try:
            if mode == "replay":
                await self._replay()
            else:
                await self._camera()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Exception messages can contain device paths, never log raw frame/user data.
            self.component_errors["monitoring"] = (
                type(error).__name__ + ": check camera permission and model setup"
            )
            await self.unavailable("capture_or_perception_failure")
            self._enqueue_record(
                "event",
                (
                    self.session_id,
                    {"component": "monitoring", "status": "unavailable", "timestamp": time.time()},
                    "health",
                ),
            )

    async def _replay(self):
        fixture = json.loads((ROOT / "fixtures/demo.json").read_text())
        base = time.time()
        start = time.monotonic()
        for index, item in enumerate(fixture["frames"]):
            await asyncio.sleep(max(0, start + item["t"] - time.monotonic()))
            await self.ingest(
                dict(item, frame_id=index + 1, captured_at=base + item["t"], synthetic=True)
            )
        await self.unavailable("synthetic_replay_complete")

    async def _camera(self):
        import cv2
        from perception.pipeline import FrameProcessor

        from core.setup import verify_asset

        if not verify_asset(self.config.model_path):
            raise FileNotFoundError("Run dms setup-models")
        loop = asyncio.get_running_loop()
        camera_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dms-camera")
        processor = FrameProcessor(self.config)
        worker = asyncio.create_task(processor.worker.run())
        cap = None
        capture_task = None
        try:
            cap = await loop.run_in_executor(
                camera_executor, cv2.VideoCapture, self.config.camera_index
            )

            def configure():
                if not cap.isOpened():
                    raise RuntimeError("Camera could not be opened")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
                cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)

            await loop.run_in_executor(camera_executor, configure)

            async def capture():
                frame_id = 0
                while True:
                    t0 = time.monotonic()
                    ok, frame = await loop.run_in_executor(camera_executor, cap.read)
                    if not ok:
                        raise RuntimeError("Camera disconnected")
                    frame_id += 1
                    captured = time.time()
                    if self.config.flip_horizontal:
                        frame = cv2.flip(frame, 1)
                    processor.worker.submit((frame, frame_id, captured))
                    await asyncio.sleep(
                        max(0.001, 1 / self.config.camera_fps - (time.monotonic() - t0))
                    )

            capture_task = asyncio.create_task(capture())

            async def consume():
                while True:
                    result = await processor.worker.results.get()
                    if result is not None:
                        raw, preview = result
                        await self.ingest(raw)
                        self.preview = preview
                        self.preview_version += 1

            consumer = asyncio.create_task(consume())
            try:
                done, _ = await asyncio.wait(
                    [worker, capture_task, consumer], return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    await task
            finally:
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)
        finally:
            if capture_task:
                capture_task.cancel()
                await asyncio.gather(capture_task, return_exceptions=True)
            # Complete any in-flight camera read before releasing the device on its owning thread.
            if cap is not None:
                await loop.run_in_executor(camera_executor, cap.release)
            camera_executor.shutdown(wait=True)
            await processor.worker.close()
            await asyncio.gather(worker, return_exceptions=True)

    async def ingest(self, raw):
        if time.time() - raw["captured_at"] > self.config.max_frame_age_s and not raw.get(
            "synthetic"
        ):
            await self.unavailable("stale_capture")
            return
        snapshot = self.monitor.update(raw)
        if snapshot is None:
            return
        self.snapshot = snapshot
        self.frame_count += 1
        self.latencies.append(max(0, (time.time() - snapshot.captured_at) * 1000))
        # Safety is evaluated and audio dispatched before optional queues or network consumers.
        started = time.perf_counter()
        alert = self.policy.evaluate(snapshot)
        if alert:
            self.last_alert = alert
            if alert.transition != "resolved":
                self.audio.speak(alert.message, critical=alert.severity == "critical")
            else:
                self.audio.silence()
            self.alert_latencies.append((time.perf_counter() - started) * 1000)
            self._enqueue_record("event", (self.session_id, alert, "alert"))
            await self.bus.publish(EventTopic.ALERT_TRIGGER, alert.model_dump(mode="json"))
        if snapshot.captured_at - self.last_recorded >= 1:
            self._enqueue_record("snapshot", snapshot)
            self.last_recorded = snapshot.captured_at
        if self.monitor.calibration.result and not self.calibration_saved:
            await self.store.save_calibration(self.monitor.calibration.result)
            self.calibration_saved = True
        await self.bus.publish(EventTopic.FATIGUE_SCORE, snapshot.model_dump(mode="json"))

    async def unavailable(self, reason):
        self.snapshot = MonitoringSnapshot(
            session_id=self.session_id,
            frame_id=(self.snapshot.frame_id + 1 if self.snapshot else 0),
            captured_at=time.time(),
            state="unavailable",
            reasons=[reason],
            synthetic=bool(self.snapshot and self.snapshot.synthetic),
        )
        if self.session_id:
            self._enqueue_record("snapshot", self.snapshot)
        await self.bus.publish(EventTopic.FATIGUE_SCORE, self.snapshot.model_dump(mode="json"))

    async def acknowledge(self):
        self.policy.acknowledge()
        self.audio.silence()
        if self.last_alert and self.session_id:
            event = self.last_alert.model_copy(
                update={
                    "id": __import__("uuid").uuid4().hex,
                    "transition": "acknowledged",
                    "acknowledged": True,
                    "timestamp": time.time(),
                }
            )
            self.last_alert = event
            self._enqueue_record("event", (self.session_id, event, "alert"))
            await self.bus.publish(EventTopic.ALERT_TRIGGER, event.model_dump(mode="json"))

    async def stop(self):
        async with self.lock:
            if self.session_task:
                self.session_task.cancel()
                await asyncio.gather(self.session_task, return_exceptions=True)
                self.session_task = None
            self.audio.silence()
            await self.records.join()
            if self.session_id:
                await self.store.end_session(self.session_id)
            self.session_id = None
            self.preview = None
            self.snapshot = None
            self.monitor = Monitor(self.config)
            self.last_alert = None

    async def close(self):
        if self.maintenance_task:
            self.maintenance_task.cancel()
            await asyncio.gather(self.maintenance_task, return_exceptions=True)
        await self.stop()
        await self.assistant.close()
        await self.audio.shutdown()
        self.bus.stop()
        if self.bus_task:
            await self.bus_task
        if self.recorder_task:
            self.recorder_task.cancel()
            await asyncio.gather(self.recorder_task, return_exceptions=True)
        if self.knowledge_task:
            await asyncio.gather(self.knowledge_task, return_exceptions=True)
        await self.store.close()
        if self.instance_lock:
            self.instance_lock.close()
            self.instance_lock = None

    def health(self):
        def p95(items):
            return sorted(items)[int(0.95 * (len(items) - 1))] if items else None

        return dict(
            version="0.6.0",
            session_id=self.session_id,
            monitoring=self.snapshot.state if self.snapshot else "idle",
            snapshot=self.snapshot.model_dump(mode="json") if self.snapshot else None,
            alert=self.last_alert.model_dump(mode="json") if self.last_alert else None,
            errors={
                **self.component_errors,
                **({"audio": self.audio.last_error} if self.audio.last_error else {}),
            },
            components=dict(
                gemini="configured" if self.assistant.provider else "not configured",
                knowledge=self.knowledge.status,
                voice="installed" if self.config.whisper_path.is_dir() else "not installed",
            ),
            metrics=dict(
                frames=self.frame_count,
                p95_capture_to_state_ms=p95(self.latencies),
                p95_alert_dispatch_ms=p95(self.alert_latencies),
                recorder_backlog=self.records.qsize(),
                bus=self.bus.get_stats(),
            ),
        )
