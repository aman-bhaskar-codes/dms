"""
TTS Engine for V4 — priority queue using pyttsx3.
"""
import threading
import pyttsx3
import queue
from typing import Tuple
from config import settings


class TTSEngine:
    def __init__(self):
        self._q = queue.PriorityQueue()
        self._running = False
        self._thread = None
        # Must initialize pyttsx3 in the same thread that runs it
        
    def _run_loop(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', settings.tts_rate)
        engine.setProperty('volume', settings.tts_volume)
        
        while self._running:
            try:
                priority, text = self._q.get(timeout=0.5)
                engine.say(text)
                engine.runAndWait()
                self._q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS] Error: {e}")

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[TTS] Engine started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def speak(self, text: str, priority: int = 2):
        """
        priority 0 = critical (microsleep)
        priority 1 = warning (yawn, distracted)
        priority 2 = info (tips)
        """
        if self._running:
            # Avoid huge backlogs
            if self._q.qsize() < 3:
                self._q.put((priority, text))
