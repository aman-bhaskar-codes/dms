import numpy as np


def test_local_recording_enables_silence_filter(monkeypatch, tmp_path):
    import sounddevice
    from types import SimpleNamespace
    from voice.local import LocalVoice

    observed = {}

    class Model:
        def transcribe(self, audio, **kwargs):
            observed.update(kwargs)
            return iter([SimpleNamespace(text="Take a safe break.")]), None

    monkeypatch.setattr(sounddevice, "rec", lambda *a, **k: np.zeros((80000, 1), dtype=np.float32))
    monkeypatch.setattr(sounddevice, "wait", lambda: None)
    voice = LocalVoice(tmp_path)
    voice.model = Model()
    assert voice._record() == "Take a safe break."
    assert observed["vad_filter"] is True
