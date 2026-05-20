"""
Voice Cognition Agent V5 — The working voice agent your repo claims to have.
VAD-gated Whisper STT → Intent classification → Rich context → Ollama → TTS

This is NOT a fire-and-forget prompt. It is a live, context-aware, memory-grounded
conversational agent that operates parallel to the camera loop.
"""
from __future__ import annotations
import asyncio
import threading
import queue
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
import pyaudio
import torch


class Intent(Enum):
    ASK_CURRENT       = "ask_current"        # "how am I doing?"
    ASK_HISTORY       = "ask_history"        # "how did I do last week?"
    ASK_TREND         = "ask_trend"          # "am I getting worse?"
    REPORT_TIRED      = "report_tired"       # "I'm feeling tired"
    REPORT_FINE       = "report_fine"        # "I'm fine, stop alerting"
    ADJUST_THRESHOLD  = "adjust_threshold"   # "you're being too sensitive"
    REQUEST_BREAK     = "request_break"      # "where can I stop?"
    UNKNOWN           = "unknown"


@dataclass
class VoiceQuery:
    text: str
    intent: Intent
    confidence: float
    timestamp: float
    live_metrics: dict


class VoiceCognitionAgent:
    """
    Runs in a background thread. Listens, classifies intent, 
    queries memory, generates grounded response, speaks it.
    """
    
    INTENT_PATTERNS = {
        Intent.ASK_CURRENT:  ["how am i", "my score", "fatigue", "how tired", "doing right now"],
        Intent.ASK_HISTORY:  ["last time", "yesterday", "last week", "compare", "history", "before"],
        Intent.ASK_TREND:    ["getting worse", "trend", "rising", "improving", "trajectory"],
        Intent.REPORT_TIRED: ["feeling tired", "sleepy", "drowsy", "need a break", "exhausted"],
        Intent.REPORT_FINE:  ["i'm fine", "im fine", "stop", "false alarm", "i'm okay"],
        Intent.ADJUST_THRESHOLD: ["too sensitive", "too many alerts", "adjust", "calibrate"],
        Intent.REQUEST_BREAK: ["rest stop", "where to stop", "pull over", "take a break"],
    }
    
    def __init__(self, get_metrics: Callable, get_memory_context: Callable,
                 ollama_client, tts_engine):
        self.get_metrics = get_metrics          # Callable → current metrics dict
        self.get_memory_context = get_memory_context  # Callable → rich context dict
        self.ollama = ollama_client
        self.tts = tts_engine
        
        self._query_queue: queue.Queue = queue.Queue(maxsize=3)
        self._response_history: list = []      # Last 5 exchanges for context
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # VAD + Whisper (lazy loaded to avoid startup delay)
        self._vad_model = None
        self._whisper_model = None
        self._audio = None
        self._stream = None
        
    def start(self):
        """Start background voice listening thread."""
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[VoiceAgent] Cognition agent started — listening for speech...")
    
    def _load_models(self):
        """Lazy load to avoid slowing down startup."""
        if self._vad_model is None:
            self._vad_model, utils = torch.hub.load(
                'snakers4/silero-vad', 'silero_vad', force_reload=False
            )
            (self.get_speech_ts, *_) = utils
            print("[VoiceAgent] VAD model loaded")
        
        if self._whisper_model is None:
            from faster_whisper import WhisperModel
            self._whisper_model = WhisperModel(
                "medium", device="cpu", compute_type="int8"
            )
            print("[VoiceAgent] Whisper medium loaded")
    
    def classify_intent(self, text: str) -> tuple[Intent, float]:
        """Rule-based intent classification — no LLM needed for routing."""
        text_lower = text.lower()
        best_intent = Intent.UNKNOWN
        best_score = 0.0
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if p in text_lower) / len(patterns)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return best_intent, best_score
    
    def _listen_loop(self):
        """Background audio capture + VAD + transcription."""
        self._load_models()
        
        CHUNK = 512
        RATE = 16000
        BUFFER_SEC = 4  # Process 4-second windows
        
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=1, rate=RATE,
            input=True, frames_per_buffer=CHUNK
        )
        
        audio_buffer = []
        buffer_frames = RATE * BUFFER_SEC // CHUNK
        
        print("[VoiceAgent] Microphone active. Listening...")
        
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
                        min_speech_duration_ms=300,
                        min_silence_duration_ms=400,
                    )
                    
                    if not speech_ts:
                        continue  # No speech, skip
                    
                    # Whisper transcription
                    segments, _ = self._whisper_model.transcribe(
                        audio_array,
                        language="en",
                        vad_filter=True,
                    )
                    text = " ".join(s.text.strip() for s in segments).strip()
                    
                    if len(text) < 5:
                        continue
                    
                    intent, conf = self.classify_intent(text)
                    query = VoiceQuery(
                        text=text,
                        intent=intent,
                        confidence=conf,
                        timestamp=time.time(),
                        live_metrics=self.get_metrics(),
                    )
                    
                    # Put in queue (non-blocking, drop if full)
                    try:
                        self._query_queue.put_nowait(query)
                    except queue.Full:
                        pass
                        
            except Exception as e:
                print(f"[VoiceAgent] Listen error: {e}")
                time.sleep(0.1)
    
    async def process_queries(self):
        """
        Coroutine — called from main async loop.
        Processes queued voice queries and generates spoken responses.
        """
        while not self._query_queue.empty():
            try:
                query: VoiceQuery = self._query_queue.get_nowait()
                await self._respond(query)
            except queue.Empty:
                break
    
    async def _respond(self, query: VoiceQuery):
        """Generate and speak a grounded, context-aware response."""
        print(f"[VoiceAgent] Query: '{query.text}' | Intent: {query.intent.value}")
        
        # Build grounded prompt based on intent
        context = await self.get_memory_context(query.live_metrics)
        prompt = self._build_response_prompt(query, context)
        
        # Get response from Ollama (with timeout + circuit breaker)
        response = await self.ollama.get_driving_tip(context, query.live_metrics,
                                                      override_prompt=prompt)
        
        # Track conversation (last 5 exchanges)
        self._response_history.append({
            "query": query.text,
            "response": response,
            "timestamp": query.timestamp,
            "intent": query.intent.value,
        })
        if len(self._response_history) > 5:
            self._response_history.pop(0)
        
        # Speak it
        priority = 3 if query.intent in (Intent.REPORT_TIRED, Intent.REQUEST_BREAK) else 2
        self.tts.speak(response, priority=priority)
        print(f"[VoiceAgent] Response: '{response}'")
    
    def _build_response_prompt(self, query: VoiceQuery, context: dict) -> str:
        """Build intent-specific prompt — NOT the generic tip prompt."""
        m = query.live_metrics
        history_str = "; ".join([
            f"Q: {h['query'][:30]} → A: {h['response'][:40]}"
            for h in self._response_history[-2:]
        ]) or "none"
        
        base = f"""You are a calm driver safety co-pilot responding to the driver in real-time.
Current metrics: fatigue={m.get('fatigue_score',0):.0f}/100, 
EAR={m.get('ear',0):.3f}, PERCLOS={m.get('perclos',0)*100:.1f}%,
drive={m.get('drive_minutes',0):.0f}min, trend={m.get('fatigue_trend',0):+.2f},
yawns={m.get('yawn_rate',0)}/10min, HR={m.get('hr_bpm',0):.0f}BPM.
History: {context.get('summary','first session')}.
Recent conversation: {history_str}.
Driver asked: "{query.text}"
"""
        
        intent_instructions = {
            Intent.ASK_CURRENT: "Answer with their exact current fatigue score and one specific observation.",
            Intent.ASK_HISTORY: f"Compare to history. Historical avg: {context.get('historical_avg_fatigue',0):.0f}. Say if better or worse and why.",
            Intent.ASK_TREND: f"Explain the trend direction ({context.get('fatigue_trend',0):+.2f}/frame). Predict where they'll be in 10 min.",
            Intent.REPORT_TIRED: "Acknowledge it. Give specific advice. Suggest a concrete action (stop, open window, etc).",
            Intent.REPORT_FINE: "Acknowledge. Reduce alert frequency mentally. Confirm you'll keep monitoring.",
            Intent.ADJUST_THRESHOLD: "Acknowledge sensitivity feedback. Explain what's being detected.",
            Intent.REQUEST_BREAK: "Suggest a break. Mention their current fatigue level.",
            Intent.UNKNOWN: "Answer their driving safety question directly.",
        }
        
        instruction = intent_instructions.get(query.intent, "Answer helpfully.")
        return f"{base}\nInstruction: {instruction}\nResponse (max 25 words, calm, spoken aloud):"
    
    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio:
            self._audio.terminate()
