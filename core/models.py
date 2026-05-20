"""
Single source of truth for all data shapes.
Using Pydantic v2 for validation + serialization.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
import numpy as np
import time


class FatigueLevel(str, Enum):
    SAFE     = "safe"
    MILD     = "mild"
    WARNING  = "warning"
    CRITICAL = "critical"
    MICROSLEEP = "microsleep"


class AlertSeverity(str, Enum):
    INFO    = "info"
    WARNING = "warning"
    DANGER  = "danger"
    CRITICAL = "critical"


@dataclass
class EARFrame:
    ear: float
    ear_left: float
    ear_right: float
    blink_count: int
    blink_velocity: float
    slow_blink_ratio: float
    state: str   # "normal" | "tired" | "critical"
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class PERCLOSFrame:
    perclos: float       # 0.0 – 1.0
    trend_slope: float   # positive = worsening
    predicted_3min: float
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class HeadPoseFrame:
    yaw: float
    pitch: float
    roll: float
    sway_amplitude: float
    nod_frequency: float
    state: str   # "forward" | "nodding" | "distracted"
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class GazeFrame:
    gaze_x: float
    gaze_y: float
    fixation_duration: float
    saccade_velocity: float
    attention_score: float   # 0-1
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class RPPGFrame:
    heart_rate_bpm: float
    hrv_sdnn: float
    stress_index: float   # 0-1
    signal_quality: float # 0-1
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class TrafficFrame:
    """NEW V5: Traffic context from scene understanding."""
    vehicle_density: float    # 0-1 (sparse → heavy)
    is_highway: bool
    estimated_speed_kmh: float
    lane_change_risk: float   # 0-1
    pedestrian_proximity: float  # 0-1
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class FatigueFrame:
    """The unified output of the fusion engine."""
    score: float              # 0-100
    level: FatigueLevel
    predicted_3min: float     # projected score in 3 minutes
    component_scores: Dict[str, float]  # per-signal contributions
    ear: Optional[EARFrame] = None
    perclos: Optional[PERCLOSFrame] = None
    head_pose: Optional[HeadPoseFrame] = None
    gaze: Optional[GazeFrame] = None
    rppg: Optional[RPPGFrame] = None
    traffic: Optional[TrafficFrame] = None
    timestamp: float = field(default_factory=time.monotonic)


# Pydantic models for API serialization
class FatigueFrameSchema(BaseModel):
    score: float
    level: str
    predicted_3min: float
    component_scores: Dict[str, float]
    heart_rate_bpm: Optional[float] = None
    timestamp: float

    class Config:
        from_attributes = True


class AlertSchema(BaseModel):
    alert_type: str
    severity: str
    score: float
    message: str
    timestamp: float


class DriverProfileSchema(BaseModel):
    driver_id: str
    baseline_ear: float
    baseline_mar: float
    ear_threshold_warn: float
    ear_threshold_crit: float
    total_sessions: int
    average_fatigue: float
