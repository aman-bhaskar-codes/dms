import asyncio
from alerts.alert_engine import AlertEngine


async def test_nonzero_audio_exit_is_reported_without_affecting_monitoring(monkeypatch):
    import alerts.alert_engine as audio

    class FailedAudio:
        returncode = 1

        async def wait(self):
            return 1

    async def spawn(*args, **kwargs):
        return FailedAudio()

    monkeypatch.setattr(audio.sys, "platform", "darwin")
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/say")
    monkeypatch.setattr(audio.asyncio, "create_subprocess_exec", spawn)
    engine = AlertEngine()
    engine.start()
    try:
        engine.speak("Test alert", critical=True)
        await asyncio.wait_for(engine.queue.join(), 1)
        assert "failed" in engine.last_error.lower()
    finally:
        await engine.shutdown()


def test_critical_audio_replaces_queued_coaching():
    engine = AlertEngine()
    engine.speak("Optional coaching")
    engine.speak("Critical fatigue", critical=True)
    assert engine.queue.qsize() == 1
    priority, _, text = engine.queue.get_nowait()
    assert priority == 0 and text == "Critical fatigue"
