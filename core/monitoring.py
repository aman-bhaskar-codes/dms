"""Time-based signal fusion and calibration; independent of camera and network."""

import math
import statistics
from collections import deque
from core.config import Settings
from core.contracts import MonitoringSnapshot


class Calibration:
    def __init__(self):
        self.active = False
        self.started = 0.0
        self.valid_seconds = 0.0
        self.last = None
        self.samples = []
        self.result = None
        self.reason = ""

    def start(self, now):
        self.__init__()
        self.active = True
        self.started = now

    def update(self, t, ear, frontal):
        if not self.active:
            return
        if t - self.started >= 60:
            self.active = False
            self.reason = "Not enough valid samples. Improve lighting and retry."
            return
        if frontal and ear is not None and 0.20 <= ear <= 0.5:
            if self.last is not None:
                self.valid_seconds += min(t - self.last, 0.25)
            self.last = t
            self.samples.append(ear)
            if self.valid_seconds >= 20:
                baseline = statistics.median(self.samples)
                self.result = {"baseline_ear": baseline, "ear_threshold": baseline * 0.65}
                self.active = False
                self.reason = "Calibration complete"
        else:
            self.last = None

    def status(self):
        return dict(
            active=self.active,
            valid_seconds=round(self.valid_seconds, 1),
            required_seconds=20,
            reason=self.reason,
        )


class Monitor:
    KEYS = ("ear", "mar", "pitch", "yaw", "eye_closed_s", "perclos")

    def __init__(self, config=None):
        self.config = config or Settings()
        self.last_time = None
        self.last_id = -1
        self.closed_since = None
        self.previous_closed = None
        self.window = deque()
        self.calibration = Calibration()
        self.ear_threshold = self.config.microsleep_ear_threshold
        self.baseline_ear = 0.30
        self.session_id = None

    def update(self, raw):
        t = float(raw["captured_at"])
        frame_id = raw["frame_id"]
        if self.last_time is not None and (t <= self.last_time or frame_id <= self.last_id):
            return None
        gap = t - self.last_time if self.last_time is not None else 0
        self.last_time, self.last_id = t, frame_id
        raw = dict(raw)
        for key, limit in (("ear", 1.5), ("mar", 3), ("pitch", 180), ("yaw", 180)):
            value = raw.get(key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or abs(value) > limit
            ):
                raw[key] = None
        valid = bool(raw.get("face_detected")) and all(
            raw.get(k) is not None for k in ("ear", "mar")
        )
        pose_valid = raw.get("yaw") is not None and raw.get("pitch") is not None
        self.calibration.update(
            t,
            raw.get("ear") if valid else None,
            valid and pose_valid and abs(raw["yaw"]) < 15 and abs(raw["pitch"]) < 15,
        )
        if self.calibration.result:
            self.ear_threshold = self.calibration.result["ear_threshold"]
            self.baseline_ear = self.calibration.result["baseline_ear"]
        if not valid:
            self.previous_closed = None
            self.closed_since = None
            self.window.clear()
            return MonitoringSnapshot(
                session_id=self.session_id,
                frame_id=frame_id,
                captured_at=t,
                state="calibrating" if self.calibration.active else "unavailable",
                measurements=dict.fromkeys(self.KEYS),
                validity=dict.fromkeys(self.KEYS, False),
                reasons=[raw.get("reason", "face_not_detected")],
                synthetic=raw.get("synthetic", False),
                calibration=self.calibration.status(),
            )
        ear, mar = float(raw["ear"]), float(raw["mar"])
        closed = ear < self.ear_threshold
        if gap > 2:
            self.previous_closed = None
            self.closed_since = None
            self.window.clear()
        if closed:
            if self.closed_since is None:
                self.closed_since = t
        else:
            self.closed_since = None
        duration = t - self.closed_since if self.closed_since is not None else 0.0
        if self.previous_closed is not None:
            self.window.append((t, gap, self.previous_closed))
        self.previous_closed = closed
        while self.window and self.window[0][0] < t - 60:
            self.window.popleft()
        total = sum(x[1] for x in self.window)
        perclos = sum(x[1] for x in self.window if x[2]) / total if total else 0
        # Fixed demonstration weights. Missing/experimental signals never enter fusion.
        score = min(
            100,
            35 * min(1, max(0, (self.baseline_ear - ear) / (self.baseline_ear * 0.5)))
            + 35 * min(1, perclos / 0.4)
            + 15 * min(1, max(0, (mar - 0.35) / 0.4))
            + (15 * min(1, abs(raw["yaw"]) / 45) if pose_valid else 0),
        )
        reasons = [] if pose_valid else ["head_pose_unavailable"]
        if duration >= self.config.eye_closure_s:
            reasons.append("prolonged_eye_closure")
        if pose_valid and abs(raw["yaw"]) >= 30:
            reasons.append("head_turned")
        if mar > 0.65:
            reasons.append("mouth_open")
        level = (
            "critical"
            if score >= self.config.fatigue_crit_threshold
            else "warning"
            if score >= self.config.fatigue_warn_threshold
            else "mild"
            if score >= 25
            else "safe"
        )
        return MonitoringSnapshot(
            session_id=self.session_id,
            frame_id=frame_id,
            captured_at=t,
            state="calibrating"
            if self.calibration.active
            else "monitoring"
            if pose_valid
            else "degraded",
            measurements=dict(
                ear=ear,
                mar=mar,
                pitch=raw["pitch"],
                yaw=raw["yaw"],
                eye_closed_s=round(duration, 3),
                perclos=round(perclos, 3),
            ),
            validity={key: key not in ("pitch", "yaw") or pose_valid for key in self.KEYS},
            score=round(score, 1),
            level=level,
            reasons=reasons,
            synthetic=raw.get("synthetic", False),
            calibration=self.calibration.status(),
        )
