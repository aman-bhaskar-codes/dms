"""
DMS V5 Voice Pipeline.
Streaming VAD-gated Speech-to-Text -> Intent Classification -> RAG Query -> Text-to-Speech.
Features traffic-aware voice coaching and elite event-driven architecture.
"""

from __future__ import annotations
import asyncio
import logging
import queue
import threading
import time
from enum import Enum
import numpy as np
import torch

from core.bus import EventBus, EventTopic, Event
from core.config import settings
from core.models import TrafficFrame, FatigueFrame

logger = logging.getLogger("dms.voice")


class VoiceIntent(str, Enum):
    ASK_CURRENT       = "ask_current"        # "how am I doing?"
    ASK_HISTORY       = "ask_history"        # "how did I do last week?"
    ASK_TREND         = "ask_trend"          # "am I getting worse?"
    REPORT_TIRED      = "report_tired"       # "I'm feeling tired"
    REPORT_FINE       = "report_fine"        # "I'm fine, stop alerting"
    ADJUST_THRESHOLD  = "adjust_threshold"   # "you're being too sensitive"
    REQUEST_BREAK     = "request_break"      # "where can I stop?"
    TRAFFIC_STATUS    = "traffic_status"     # "how is the traffic ahead?"
    UNKNOWN           = "unknown"


class VoicePipeline:
    """
    Elite Event-Driven Voice Pipeline.
    Listens to the driver via microphone with Silero-VAD gating, transcribes using Whisper,
    classifies intent with a traffic-aware NLU system, queries the RAG engine via event bus,
    and speaks responses using TTS.
    """

    INTENT_PATTERNS = {
        VoiceIntent.ASK_HISTORY:  ["last time", "yesterday", "last week", "compare", "history", "before", "previous"],
        VoiceIntent.ASK_TREND:    ["getting worse", "trend", "rising", "improving", "trajectory"],
        VoiceIntent.ASK_CURRENT:  ["how am i", "my score", "fatigue", "how tired", "doing right now", "status"],
        VoiceIntent.REPORT_TIRED: ["feeling tired", "sleepy", "drowsy", "need a break", "exhausted"],
        VoiceIntent.REPORT_FINE:  ["i'm fine", "im fine", "stop", "false alarm", "i'm okay", "shut up"],
        VoiceIntent.ADJUST_THRESHOLD: ["too sensitive", "too many alerts", "adjust", "calibrate"],
        VoiceIntent.REQUEST_BREAK: ["rest stop", "where to stop", "pull over", "take a break"],
        VoiceIntent.TRAFFIC_STATUS: ["traffic", "highway", "cars ahead", "road status", "lane risk"]
    }

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Biometric & environmental context (constantly updated via bus)
        self._latest_fatigue: Optional[FatigueFrame] = None
        self._latest_traffic: Optional[TrafficFrame] = None
        
        # Models & engines (lazy loaded to prevent startup lag)
        self._vad_model = None
        self._whisper_model = None
        self._audio = None
        self._stream = None
        self._tts_engine = None
        self._tts_lock = threading.Lock()
        
        self._pending_queries: dict[str, float] = {}  # correlation_id -> timestamp

        # Subscribe to bus events
        self._bus.subscribe(EventTopic.SIGNAL_TRAFFIC, self._on_traffic)
        self._bus.subscribe(EventTopic.FATIGUE_SCORE, self._on_fatigue)
        self._bus.subscribe(EventTopic.MEMORY_READ_RESP, self._on_rag_response)
        self._bus.subscribe(EventTopic.VOICE_RESPONSE, self._on_voice_response)
        self._bus.subscribe(EventTopic.VOICE_TRANSCRIPT, self._on_transcript_received)

    async def _on_traffic(self, event: Event):
        self._latest_traffic = event.payload

    async def _on_fatigue(self, event: Event):
        self._latest_fatigue = event.payload

    async def _on_rag_response(self, event: Event):
        corr_id = event.correlation_id
        if corr_id in self._pending_queries:
            self._pending_queries.pop(corr_id, None)
            answer = event.payload.get("answer", "")
            if answer:
                await self._bus.publish(
                    EventTopic.VOICE_RESPONSE,
                    {"text": answer},
                    source="voice_pipeline",
                    correlation_id=corr_id
                )

    async def _on_voice_response(self, event: Event):
        text = event.payload.get("text", "")
        if text:
            # Run TTS speak in executor to prevent blocking the event bus dispatch loop
            asyncio.create_task(self._speak(text))

    async def _on_transcript_received(self, event: Event):
        if event.source != "voice_pipeline":
            # Direct incoming typed chat query from the dashboard
            text = event.payload.get("text", "")
            if text:
                logger.info(f"[Voice] Processing external typed query from dashboard: '{text}'")
                await self._process_speech(text)


    def _load_tts_engine(self):
        if self._tts_engine is not None:
            return
        
        engine_type = settings.tts_engine.lower()
        if engine_type == "coqui":
            try:
                from TTS.api import TTS
                # Coqui TTS is heavy but premium. Attempt offline XTTS or fast model
                self._tts_engine = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False)
                logger.info("[Voice] Coqui TTS engine initialized.")
                return
            except Exception as e:
                logger.warning(f"[Voice] Coqui TTS failed to load ({e}). Falling back to pyttsx3.")
        
        # Fallback to pyttsx3
        try:
            import pyttsx3
            # Initialize offline engine cleanly
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty('rate', 170)
            logger.info("[Voice] pyttsx3 TTS engine initialized.")
        except Exception as e:
            logger.error(f"[Voice] All TTS engines failed to load: {e}")

    def _load_speech_models(self):
        """Lazy load VAD & Whisper models."""
        if self._vad_model is None:
            logger.info("[Voice] Loading Silero-VAD model...")
            self._vad_model, utils = torch.hub.load(
                'snakers4/silero-vad', 'silero_vad', force_reload=False
            )
            self.get_speech_ts = utils[0]
            logger.info("[Voice] Silero-VAD model loaded.")

        if self._whisper_model is None:
            logger.info(f"[Voice] Loading Whisper model ({settings.whisper_model})...")
            from faster_whisper import WhisperModel
            self._whisper_model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type="int8"
            )
            logger.info("[Voice] Whisper model loaded.")

    async def _speak(self, text: str):
        """Speak text asynchronously, thread-safely notifying the bus."""
        await self._bus.publish(EventTopic.VOICE_SPEAKING, {"speaking": True}, source="voice_pipeline")
        
        # Call TTS in a separate thread so it doesn't block the asyncio event loop
        await asyncio.get_event_loop().run_in_executor(None, self._speak_sync, text)
        
        await self._bus.publish(EventTopic.VOICE_SPEAKING, {"speaking": False}, source="voice_pipeline")

    def _speak_sync(self, text: str):
        with self._tts_lock:
            self._load_tts_engine()
            if self._tts_engine is None:
                logger.error("[Voice] TTS engine unavailable. Cannot speak.")
                return

            logger.info(f"[Voice] Speaking: '{text}'")
            try:
                # Check engine type
                if hasattr(self._tts_engine, "tts_to_file"):
                    # Coqui TTS
                    import tempfile
                    import os
                    from playsound import playsound
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        temp_path = f.name
                    try:
                        self._tts_engine.tts_to_file(text=text, file_path=temp_path)
                        playsound(temp_path)
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                else:
                    # pyttsx3 engine
                    # Re-initialize to ensure stability across threads
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 170)
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                logger.error(f"[Voice] TTS playback error: {e}")

    def classify_intent(self, text: str) -> tuple[VoiceIntent, float]:
        """Traffic-aware rule-based intent classification."""
        text_lower = text.lower()
        best_intent = VoiceIntent.UNKNOWN
        best_count = 0
        best_ratio = 0.0

        for intent, patterns in self.INTENT_PATTERNS.items():
            count = sum(1 for p in patterns if p in text_lower)
            ratio = count / len(patterns)
            if count > best_count or (count == best_count and ratio > best_ratio):
                best_count = count
                best_ratio = ratio
                best_intent = intent

        # Context-dependent overrides (e.g. traffic concerns override to TRAFFIC_STATUS)
        traffic_words = {"risk", "cars", "lane", "highway", "fast", "speed"}
        if any(w in text_lower for w in traffic_words) and best_intent == VoiceIntent.UNKNOWN:
            best_intent = VoiceIntent.TRAFFIC_STATUS
            best_ratio = 0.5

        return best_intent, best_ratio

    def _listen_loop(self):
        """Microphone capture thread loop with Silero-VAD + Whisper."""
        import pyaudio
        
        try:
            self._load_speech_models()
        except Exception as e:
            logger.error(f"[Voice] Failed to load speech models: {e}. Voice capture is disabled.")
            return

        CHUNK = 512
        RATE = 16000
        BUFFER_SEC = 4
        
        try:
            self._audio = pyaudio.PyAudio()
            self._stream = self._audio.open(
                format=pyaudio.paInt16,
                channels=1, rate=RATE,
                input=True, frames_per_buffer=CHUNK
            )
        except Exception as e:
            logger.warning(f"[Voice] Microphone initialization failed: {e}. Voice capture disabled (offline simulation available).")
            return

        audio_buffer = []
        buffer_frames = RATE * BUFFER_SEC // CHUNK

        logger.info("[Voice] Microphone active. Listening for speech...")

        while self._running:
            try:
                data = self._stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.append(audio_chunk)

                if len(audio_buffer) >= buffer_frames:
                    audio_array = np.concatenate(audio_buffer)
                    audio_buffer = audio_buffer[buffer_frames // 2:]  # 50% overlap

                    # VAD check
                    tensor = torch.FloatTensor(audio_array)
                    speech_ts = self.get_speech_ts(
                        tensor, self._vad_model,
                        sampling_rate=RATE,
                        min_speech_duration_ms=250,
                        min_silence_duration_ms=400,
                    )

                    if not speech_ts:
                        continue  # Noise only, skip transcription

                    # Whisper transcription
                    segments, _ = self._whisper_model.transcribe(
                        audio_array,
                        language="en",
                        vad_filter=True,
                    )
                    text = " ".join(s.text.strip() for s in segments).strip()

                    if len(text) < 4:
                        continue

                    # Process transcript in threadsafe way
                    if self._loop and self._loop.is_running():
                        asyncio.run_coroutine_threadsafe(self._process_speech(text), self._loop)

            except Exception as e:
                logger.error(f"[Voice] Error in listening loop: {e}")
                time.sleep(0.2)

    async def _process_speech(self, text: str):
        """Asynchronously process speech transcript, classify, and trigger RAG context lookup."""
        logger.info(f"[Voice] Heard: '{text}'")
        await self._bus.publish(EventTopic.VOICE_TRANSCRIPT, {"text": text}, source="voice_pipeline")
        
        intent, confidence = self.classify_intent(text)
        logger.info(f"[Voice] Classified Intent: {intent.value} (conf: {confidence:.2f})")
        
        await self._bus.publish(
            EventTopic.VOICE_INTENT,
            {"text": text, "intent": intent.value, "confidence": confidence},
            source="voice_pipeline"
        )
        
        # Build live context (include traffic details if present!)
        live_ctx = {
            "fatigue_score": self._latest_fatigue.score if self._latest_fatigue else 0.0,
            "fatigue_level": (self._latest_fatigue.level.value if hasattr(self._latest_fatigue.level, "value") else self._latest_fatigue.level) if self._latest_fatigue else "safe",
            "vehicle_density": self._latest_traffic.vehicle_density if self._latest_traffic else 0.0,
            "is_highway": self._latest_traffic.is_highway if self._latest_traffic else False,
            "estimated_speed_kmh": self._latest_traffic.estimated_speed_kmh if self._latest_traffic else 50,
            "lane_change_risk": self._latest_traffic.lane_change_risk if self._latest_traffic else 0.0,
            "pedestrian_proximity": self._latest_traffic.pedestrian_proximity if self._latest_traffic else 0.0,
        }
        
        # Safety/Traffic-Aware modification of questions before querying the RAG system
        query_modifier = ""
        if live_ctx["vehicle_density"] > 0.6 or live_ctx["lane_change_risk"] > 0.6:
            query_modifier = " High traffic context: recommend staying focused on the lane."
        
        corr_id = f"voice_{int(time.time()*1000)}"
        self._pending_queries[corr_id] = time.time()
        
        await self._bus.publish(
            EventTopic.MEMORY_READ_REQ,
            {
                "question": f"{text}{query_modifier}",
                "live_context": live_ctx
            },
            source="voice_pipeline",
            correlation_id=corr_id
        )

    async def run(self):
        """Main async task loop keeping the voice pipeline responsive."""
        self._running = True
        self._loop = asyncio.get_running_loop()
        
        # Start PyAudio listening loop in a background thread
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        
        logger.info("[Voice] Voice pipeline started successfully.")
        
        while self._running:
            # Clean up stale pending queries
            now = time.time()
            stale_keys = [k for k, t in self._pending_queries.items() if now - t > 30.0]
            for k in stale_keys:
                self._pending_queries.pop(k, None)
                
            await asyncio.sleep(5)

    def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        if self._audio:
            try:
                self._audio.terminate()
            except Exception:
                pass
        logger.info("[Voice] Voice pipeline stopped.")
