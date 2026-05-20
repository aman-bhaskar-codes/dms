"""
Text-to-Speech Alert Engine — pyttsx3 Offline Voice Alerts
NEW in V2.

Uses pyttsx3 for 100% offline, zero-latency TTS.
No internet, no API keys, works on Windows/macOS/Linux.

Features:
  - Priority queue: critical alerts preempt lower-priority speech
  - Duplicate suppression: same message won't repeat within cooldown
  - Non-blocking: speaks in background thread
  - Rate/volume/voice configurable

Install: pip install pyttsx3
"""

import threading
import queue
import time
from typing import Dict
import config

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("[TTS] pyttsx3 not installed. Run: pip install pyttsx3")


class TTSEngine:
    def __init__(self):
        self.enabled   = config.TTS_ENABLED and PYTTSX3_AVAILABLE
        self._engine   = None
        self._queue    = queue.PriorityQueue()
        self._thread   = None
        self._running  = False
        self._speaking = False

        # Cooldown: track last time each message type was spoken
        self._last_spoken: Dict[str, float] = {}
        self._COOLDOWN = config.ALERT_COOLDOWN_SEC * 2   # TTS cooldown is 2× alert cooldown

        if self.enabled:
            self._start()
        else:
            print(f"[TTS] {'Enabled' if config.TTS_ENABLED else 'Disabled in config'}. "
                  f"pyttsx3: {'✓' if PYTTSX3_AVAILABLE else '✗ not installed'}")

    def _start(self):
        """Initialize TTS engine and start background thread."""
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate',   config.TTS_RATE)
            self._engine.setProperty('volume', config.TTS_VOLUME)

            voices = self._engine.getProperty('voices')
            if voices and config.TTS_VOICE_INDEX < len(voices):
                self._engine.setProperty('voice', voices[config.TTS_VOICE_INDEX].id)

            self._running = True
            self._thread  = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            print(f"[TTS] Voice alerts ready. Rate={config.TTS_RATE} "
                  f"Vol={config.TTS_VOLUME}")
        except Exception as e:
            print(f"[TTS] Init failed: {e}")
            self.enabled = False

    def _worker(self):
        """Background thread: speaks queued messages."""
        while self._running:
            try:
                # Priority queue returns (priority, timestamp, msg_key, text)
                # Lower priority number = higher priority
                priority, ts, msg_key, text = self._queue.get(timeout=0.5)
                self._speaking = True
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    if config.DEBUG_MODE:
                        print(f"[TTS] Speak error: {e}")
                self._speaking = False
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self._speaking = False
                if config.DEBUG_MODE:
                    print(f"[TTS] Worker error: {e}")

    def speak(self, message_key: str, priority: int = config.TTS_PRIORITY_INFO,
              force: bool = False):
        """
        Queue a TTS alert by message key (from config.TTS_MESSAGES).

        Args:
            message_key: Key in config.TTS_MESSAGES
            priority:    1=info, 2=warning, 3=critical (higher = higher priority)
            force:       Skip cooldown check
        """
        if not self.enabled:
            return

        text = config.TTS_MESSAGES.get(message_key)
        if not text:
            return

        now = time.time()
        if not force:
            last = self._last_spoken.get(message_key, 0.0)
            if now - last < self._COOLDOWN:
                return   # Cooldown active

        self._last_spoken[message_key] = now

        # Clear low-priority items from queue if this is high priority
        if priority >= config.TTS_PRIORITY_CRITICAL:
            # Clear the queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        # Negative priority so PriorityQueue returns highest first
        self._queue.put((-priority, now, message_key, text))

    def speak_custom(self, text: str, priority: int = config.TTS_PRIORITY_INFO):
        """Speak arbitrary text (from AI reporter)."""
        if not self.enabled or not text:
            return
        self._queue.put((-priority, time.time(), "_custom", text))

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def stop(self):
        self._running = False
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
