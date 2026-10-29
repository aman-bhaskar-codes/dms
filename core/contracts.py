"""Validated public telemetry contracts. Never include raw media in snapshots."""

import math
import time
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MonitoringSnapshot(Contract):
    version: int = 1
    session_id: str | None = None
    frame_id: int = Field(ge=0)
    captured_at: float
    processed_at: float = Field(default_factory=time.time)
    state: Literal["initializing", "calibrating", "monitoring", "degraded", "unavailable"]
    validity: dict[str, bool] = Field(default_factory=dict)
    measurements: dict[str, float | None] = Field(default_factory=dict)
    score: float | None = Field(None, ge=0, le=100)
    level: Literal["unknown", "safe", "mild", "warning", "critical"] = "unknown"
    reasons: list[str] = Field(default_factory=list)
    synthetic: bool = False
    calibration: dict = Field(default_factory=dict)

    @field_validator("measurements")
    @classmethod
    def finite_measurements(cls, value):
        if any(v is not None and not math.isfinite(v) for v in value.values()):
            raise ValueError("Measurements must be finite or null")
        return value


class AlertEvent(Contract):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    severity: Literal["warning", "critical"]
    cause: str
    transition: Literal["triggered", "reminder", "resolved", "acknowledged"]
    timestamp: float = Field(default_factory=time.time)
    acknowledged: bool = False
    message: str = "Stop at a safe place and take a break."


class ComponentHealth(Contract):
    component: str
    status: Literal["ready", "starting", "unavailable", "degraded"]
    last_success: float | None = None
    reason: str = ""
