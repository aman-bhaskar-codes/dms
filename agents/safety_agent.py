"""Deterministic transition policy; no LLM or optional dependency participates."""

import time
from typing import Literal
from core.config import Settings
from core.contracts import AlertEvent, MonitoringSnapshot


class SafetyPolicy:
    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or Settings()
        self.active: Literal["warning", "critical"] | None = None
        self.pending: Literal["warning", "critical"] | None = None
        self.pending_since = 0.0
        self.recovery_since: float | None = None
        self.last_alert = -float("inf")
        self.acknowledged = False

    def acknowledge(self):
        self.acknowledged = True

    def evaluate(self, snapshot: MonitoringSnapshot, now: float | None = None) -> AlertEvent | None:
        now = time.monotonic() if now is None else now
        if snapshot.state not in ("monitoring", "degraded") or snapshot.score is None:
            self.pending = None
            self.recovery_since = None
            return None
        immediate = "prolonged_eye_closure" in snapshot.reasons
        target: Literal["warning", "critical"] | None = (
            "critical"
            if immediate or snapshot.score >= self.config.fatigue_crit_threshold
            else "warning"
            if snapshot.score >= self.config.fatigue_warn_threshold
            else None
        )
        if self.active:
            threshold = (
                self.config.fatigue_crit_threshold
                if self.active == "critical"
                else self.config.fatigue_warn_threshold
            ) - 5
            if snapshot.score < threshold and not immediate:
                if self.recovery_since is None:
                    self.recovery_since = now
                if now - self.recovery_since >= 3:
                    old = self.active
                    self.active = self.pending = None
                    self.acknowledged = False
                    return AlertEvent(
                        session_id=snapshot.session_id,
                        severity=old,
                        cause="recovered",
                        transition="resolved",
                    )
            else:
                self.recovery_since = None
        if target and (self.active is None or target == "critical" and self.active == "warning"):
            if self.pending != target:
                self.pending, self.pending_since = target, now
            if immediate or now - self.pending_since >= 1:
                self.active, self.pending = target, None
                self.acknowledged = False
                self.last_alert = now
                return AlertEvent(
                    session_id=snapshot.session_id,
                    severity=target,
                    cause="prolonged_eye_closure"
                    if immediate
                    else snapshot.reasons[0]
                    if snapshot.reasons
                    else "sustained_score",
                    transition="triggered",
                )
        elif target is not None and target == self.active and not self.acknowledged:
            if now - self.last_alert >= self.config.alert_cooldown_s:
                self.last_alert = now
                return AlertEvent(
                    session_id=snapshot.session_id,
                    severity=target,
                    cause="sustained_condition",
                    transition="reminder",
                )
        else:
            self.pending = None
        return None
