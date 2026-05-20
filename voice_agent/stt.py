"""
Whisper STT with silero-VAD gating.
Only transcribes actual speech, ignoring car noise.
"""
import torch
import numpy as np
from collections import deque

class VADWhisperSTT:
    """Whisper with silero-VAD gating. Only transcribes actual speech."""
    
    def __init__(self, model_size: str = "medium"):
        # Load VAD model (tiny, fast, offline)
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        (self.get_speech_ts, _, _, _, _) = utils
        
        # Load Whisper lazily or here
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        self._buffer: deque = deque(maxlen=480000)  # 30s at 16kHz
        self.SAMPLE_RATE = 16000
        self.MIN_SPEECH_DURATION = 0.3  # seconds — ignore noise bursts
        
    def process_chunk(self, audio_chunk: np.ndarray) -> str | None:
        """Returns transcribed text only if real speech detected."""
        self._buffer.extend(audio_chunk.tolist())
        audio_tensor = torch.FloatTensor(list(self._buffer))
        
        # VAD check — is there speech in this window?
        speech_timestamps = self.get_speech_ts(
            audio_tensor, self.vad_model,
            sampling_rate=self.SAMPLE_RATE,
            min_speech_duration_ms=int(self.MIN_SPEECH_DURATION * 1000),
            min_silence_duration_ms=300,
        )
        
        if not speech_timestamps:
            return None  # Just noise — don't even call Whisper
        
        # Only transcribe speech segments
        audio_array = np.array(list(self._buffer), dtype=np.float32)
        segments, _ = self.whisper.transcribe(
            audio_array,
            language="en",
            task="transcribe",
            vad_filter=True,    # Whisper's own VAD as second layer
            vad_parameters={"threshold": 0.5},
        )
        
        text = " ".join(seg.text.strip() for seg in segments)
        if len(text) < 3:  # Ignore noise-induced single chars
            return None
            
        self._buffer.clear()
        return text
