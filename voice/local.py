"""Explicit local push-to-talk; recordings are held in memory and never sent to Gemini."""

import asyncio
from pathlib import Path


class LocalVoice:
    def __init__(self, model_path):
        self.path = Path(model_path)
        self.busy = False
        self.model = None

    async def listen(self):
        if self.busy:
            raise ValueError("A recording is already in progress")
        if not self.path.is_dir():
            raise ValueError("Install the local Whisper model with sentinel setup-models --voice")
        self.busy = True
        try:
            return await asyncio.to_thread(self._record)
        finally:
            self.busy = False

    def _record(self):
        import sounddevice as sd
        from faster_whisper import WhisperModel

        if self.model is None:
            self.model = WhisperModel(
                str(self.path), device="cpu", compute_type="int8", local_files_only=True
            )
        audio = sd.rec(5 * 16000, samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        segments, _ = self.model.transcribe(audio[:, 0], language="en", beam_size=1)
        return " ".join(s.text for s in segments).strip()[:1000]


def local_intent(text, snapshot):
    text = text.lower()
    if any(word in text for word in ("tired", "sleep", "break", "fatigue")):
        return "If you feel tired, stop at a safe place and take a break. Do not rely on the monitoring score to decide whether you can drive."
    if "calibrat" in text:
        return "Open Setup and start calibration while stationary. Face the camera with your eyes naturally open."
    if "status" in text or "score" in text:
        if snapshot is None or snapshot.score is None:
            return "There is no valid current monitoring measurement."
        return f"Current monitoring state is {snapshot.state}; the demonstration score is {snapshot.score:.1f}."
    return "Local voice supports status, fatigue, breaks, and calibration. Use the general knowledge topic selector for cloud guidance."
