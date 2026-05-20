import pyttsx3
import threading

class TTSEngine:
    """
    Text-to-Speech engine.
    Uses pyttsx3 for reliable offline operation on macOS/Windows/Linux.
    """
    def __init__(self, rate: int = 175):
        self.rate = rate
        # Test initialization
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.stop()
        except Exception as e:
            print(f"[TTS] Warning: Failed to init pyttsx3: {e}")

    def speak(self, text: str):
        """Speaks the given text asynchronously."""
        print(f"[TTS] Speaking: '{text}'")
        thread = threading.Thread(target=self._speak_sync, args=(text,), daemon=True)
        thread.start()
        
    def _speak_sync(self, text: str):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS] Error during speech: {e}")
