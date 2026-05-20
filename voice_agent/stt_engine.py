from faster_whisper import WhisperModel
import os

class STTEngine:
    """
    Speech-to-Text engine using faster-whisper.
    """
    def __init__(self, model_size: str = "tiny.en", device: str = "cpu", compute_type: str = "int8"):
        print(f"[STT] Loading Whisper model ({model_size}) on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_file_path: str) -> str:
        """Transcribes an audio file and returns the text."""
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
        segments, info = self.model.transcribe(audio_file_path, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        print(f"[STT] Transcribed: '{text}' (prob: {info.language_probability:.2f})")
        return text
