import asyncio

import pytest

from core.contracts import MonitoringSnapshot
from core.monitoring import Monitor
from agents.safety_agent import SafetyPolicy
from core.bus import EventBus, EventTopic


def sample(t, **changes):
    values = dict(
        frame_id=int(t * 100) + 1,
        captured_at=t,
        face_detected=True,
        ear=0.3,
        mar=0.2,
        pitch=0.0,
        yaw=0.0,
    )
    values.update(changes)
    return values


def test_invalid_face_never_looks_safe_and_closure_resets():
    m = Monitor()
    m.update(sample(1, ear=0.1))
    missing = m.update(sample(2, face_detected=False))
    assert missing.state == "unavailable"
    assert missing.score is None and missing.measurements["ear"] is None
    recovered = m.update(sample(3, ear=0.1))
    assert recovered.measurements["eye_closed_s"] == 0


def test_elapsed_closure_irregular_frames_and_stale_results():
    m = Monitor()
    m.update(sample(1, ear=0.1))
    result = m.update(sample(2.6, ear=0.1))
    assert result.measurements["eye_closed_s"] == pytest.approx(1.6)
    assert "prolonged_eye_closure" in result.reasons
    assert m.update(sample(2)) is None
    assert "rppg_hr" not in result.measurements


def test_critical_dedup_ack_recovery_and_escalation():
    policy = SafetyPolicy()
    s = MonitoringSnapshot(
        frame_id=1,
        captured_at=1,
        state="monitoring",
        score=80,
        level="critical",
        reasons=["prolonged_eye_closure"],
    )
    alert = policy.evaluate(s, now=1)
    assert alert.severity == "critical"
    assert policy.evaluate(s, now=2) is None
    policy.acknowledge()
    assert policy.evaluate(s, now=20) is None
    safe = s.model_copy(update={"score": 0, "level": "safe", "reasons": []})
    assert policy.evaluate(safe, now=21) is None
    assert policy.evaluate(safe, now=24).transition == "resolved"
    assert policy.evaluate(s, now=25).severity == "critical"


def test_warning_requires_sustained_crossing():
    policy = SafetyPolicy()
    s = MonitoringSnapshot(frame_id=1, captured_at=1, state="monitoring", score=60, level="warning")
    assert policy.evaluate(s, now=0) is None
    assert policy.evaluate(s, now=0.5) is None
    assert policy.evaluate(s, now=1.1).severity == "warning"


async def test_slow_subscriber_does_not_block_other_subscribers():
    bus = EventBus(queue_size=2)
    started, fast = asyncio.Event(), asyncio.Event()

    async def slow(event):
        started.set()
        await asyncio.sleep(0.2)

    async def quick(event):
        fast.set()

    bus.subscribe(EventTopic.FATIGUE_SCORE, slow)
    bus.subscribe(EventTopic.FATIGUE_SCORE, quick)
    task = asyncio.create_task(bus.run())
    await bus.publish(EventTopic.FATIGUE_SCORE, {})
    await asyncio.wait_for(started.wait(), 0.1)
    await asyncio.wait_for(fast.wait(), 0.1)
    bus.stop()
    await task


def test_normal_state_never_emits_null_severity_alert():
    policy = SafetyPolicy()
    normal = MonitoringSnapshot(
        frame_id=1, captured_at=1, state="monitoring", score=0, level="safe"
    )
    assert policy.evaluate(normal, now=0) is None
    assert policy.evaluate(normal, now=100) is None


def test_invalid_measurements_and_missing_pose_are_explicit():
    m = Monitor()
    invalid = m.update(sample(1, ear=float("nan")))
    assert invalid.score is None and invalid.state == "unavailable"
    partial = m.update(sample(2, yaw=None, pitch=None))
    assert partial.state == "degraded"
    assert partial.measurements["yaw"] is None
    assert partial.validity["yaw"] is False


def test_perclos_accounts_for_elapsed_time_not_frame_count():
    m = Monitor()
    m.update(sample(1))
    m.update(sample(2, ear=0.1))
    result = m.update(sample(2.5, ear=0.1))
    # First second was open, then half a second closed.
    assert result.measurements["perclos"] == pytest.approx(1 / 3, abs=0.001)


def test_calibration_collects_only_valid_time_and_times_out():
    from core.monitoring import Calibration

    calibration = Calibration()
    calibration.start(0)
    for index in range(202):
        calibration.update(index / 10, 0.3, True)
    assert calibration.result["ear_threshold"] == pytest.approx(0.195)
    assert not calibration.active
    calibration.start(0)
    calibration.update(61, None, False)
    assert calibration.result is None and not calibration.active
