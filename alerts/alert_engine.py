"""
Alert Engine V2 — Escalation Ladder + Priority Queue

V2 additions:
  - Escalation ladder: repeated alerts within a time window auto-escalate
  - Per-alert severity levels with escalation tracking
  - Integration with TTS engine
  - Fatigue-score-driven alert tier
  - "Suggest break" logic after long drive time
"""

import time
import numpy as np
from enum import IntEnum
from typing import Optional, Dict
from collections import defaultdict, Counter
import config

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False


class AlertLevel(IntEnum):
    NORMAL   = 0
    MILD     = 1
    ALERT    = 2
    WARNING  = 3
    CRITICAL = 4


def _gen_beep(freq, dur_ms, vol=0.8, wave="sine"):
    if not PYGAME_AVAILABLE:
        return None
    sr   = 44100
    n    = int(sr * dur_ms / 1000)
    t    = np.linspace(0, dur_ms / 1000, n, endpoint=False)
    if wave == "square":
        w = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave == "sawtooth":
        w = 2 * (t * freq - np.floor(t * freq + 0.5))
    else:
        w = np.sin(2 * np.pi * freq * t)
    fade = min(int(sr * 0.05), n // 4)
    if fade > 0:
        w[-fade:] *= np.linspace(1.0, 0.0, fade)
    w = (w * vol * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([w, w]))


class AlertEngine:
    ALERT_DEFS = {
        "drowsy_warn":  {"level": AlertLevel.WARNING,  "freq": 880,  "dur": 200, "tts": "drowsy_warn",  "msg": "⚠ DROWSY — WARNING"},
        "drowsy_crit":  {"level": AlertLevel.CRITICAL, "freq": 440,  "dur": 500, "tts": "drowsy_crit",  "msg": "🚨 DROWSY — CRITICAL"},
        "yawn":         {"level": AlertLevel.ALERT,    "freq": 660,  "dur": 150, "tts": "yawn",         "msg": "😮 YAWN DETECTED"},
        "head_nod":     {"level": AlertLevel.WARNING,  "freq": 770,  "dur": 300, "tts": "head_nod",     "msg": "⚠ HEAD NOD"},
        "distracted":   {"level": AlertLevel.WARNING,  "freq": 990,  "dur": 250, "tts": "distracted",   "msg": "👀 EYES OFF ROAD"},
        "gaze_away":    {"level": AlertLevel.ALERT,    "freq": 1100, "dur": 150, "tts": None,            "msg": "👁 GAZE AWAY"},
        "phone":        {"level": AlertLevel.CRITICAL, "freq": 440,  "dur": 600, "tts": "phone",        "msg": "📱 PHONE DETECTED"},
        "perclos_warn": {"level": AlertLevel.WARNING,  "freq": 880,  "dur": 350, "tts": "drowsy_warn",  "msg": "⚠ HIGH PERCLOS"},
        "perclos_crit": {"level": AlertLevel.CRITICAL, "freq": 330,  "dur": 700, "tts": "perclos_crit", "msg": "🚨 DROWSY DRIVING"},
        "fatigue_high": {"level": AlertLevel.CRITICAL, "freq": 440,  "dur": 600, "tts": "fatigue_high", "msg": "⚡ FATIGUE CRITICAL"},
        "hr_stress":    {"level": AlertLevel.ALERT,    "freq": 550,  "dur": 200, "tts": "hr_stress",    "msg": "💓 HR ELEVATED"},
        "no_face":      {"level": AlertLevel.WARNING,  "freq": 550,  "dur": 400, "tts": None,            "msg": "❓ DRIVER NOT DETECTED"},
        "break_suggest":{"level": AlertLevel.MILD,     "freq": 440,  "dur": 150, "tts": "break_suggest","msg": "☕ TAKE A BREAK"},
        "head_jerk":    {"level": AlertLevel.CRITICAL, "freq": 440,  "dur": 400, "tts": "head_nod",     "msg": "⚡ MICROSLEEP SUSPECTED"},
    }

    def __init__(self, tts_engine=None):
        self.current_level   = AlertLevel.NORMAL
        self.active_alerts   : Dict[str, float] = {}
        self.alert_history   : list = []
        self.current_message = ""
        self.message_until   = 0.0
        self._good_frames    = 0
        self._tts            = tts_engine

        # Escalation tracking: count recent triggers per type
        self._escalation_counts: Dict[str, list] = defaultdict(list)

        # Pre-generate sounds
        self._sounds: Dict = {}
        self._pregenerate_sounds()

        # Drive time tracking for break suggestions
        self._session_start = time.time()

        print(f"[AlertEngine V2] Escalation: {'ON' if config.ALERT_ESCALATION_ENABLED else 'OFF'}")

    def _pregenerate_sounds(self):
        if not PYGAME_AVAILABLE:
            return
        for key, spec in self.ALERT_DEFS.items():
            wave  = "square" if spec["level"] == AlertLevel.CRITICAL else "sine"
            rpts  = 3 if spec["level"] == AlertLevel.CRITICAL else 1
            sound = _gen_beep(spec["freq"], spec["dur"] * rpts, wave=wave)
            self._sounds[key] = sound
        print(f"[AlertEngine V2] {len(self._sounds)} audio alerts generated.")

    def trigger(self, alert_type: str, force: bool = False) -> bool:
        if alert_type not in self.ALERT_DEFS:
            return False

        now  = time.time()
        last = self.active_alerts.get(alert_type, 0.0)

        if not force and (now - last) < config.ALERT_COOLDOWN_SEC:
            return False

        spec = self.ALERT_DEFS[alert_type]

        # Escalation: track recent triggers
        if config.ALERT_ESCALATION_ENABLED:
            times = self._escalation_counts[alert_type]
            times.append(now)
            # Keep only last 60 seconds
            times[:] = [t for t in times if now - t < 60.0]
            if len(times) >= config.ALERT_ESCALATION_COUNT:
                # Escalate to next level
                spec = dict(spec)  # copy
                new_lvl = min(int(AlertLevel.CRITICAL), int(spec["level"]) + 1)
                spec["level"] = AlertLevel(new_lvl)

        # Play sound
        if PYGAME_AVAILABLE and config.ALERT_SOUND_ENABLED:
            s = self._sounds.get(alert_type)
            if s:
                s.play()

        # TTS
        if self._tts and spec.get("tts"):
            tts_priority = {
                AlertLevel.CRITICAL: config.TTS_PRIORITY_CRITICAL,
                AlertLevel.WARNING:  config.TTS_PRIORITY_WARNING,
            }.get(spec["level"], config.TTS_PRIORITY_INFO)
            self._tts.speak(spec["tts"], priority=tts_priority)

        # State update
        self.active_alerts[alert_type] = now
        self.alert_history.append((now, alert_type, spec["level"].name))
        self.current_message = spec["msg"]
        self.message_until   = now + 3.0
        self._good_frames    = 0

        if spec["level"] > self.current_level:
            self.current_level = spec["level"]

        return True

    def trigger_fatigue(self, fatigue_level: str):
        """Trigger appropriate alert based on fatigue score level."""
        if fatigue_level == "critical":
            self.trigger("fatigue_high")
        elif fatigue_level == "warning":
            self.trigger("drowsy_warn")

    def check_break_suggestion(self):
        """Suggest a break if driving time exceeds threshold."""
        elapsed_min = (time.time() - self._session_start) / 60.0
        if elapsed_min >= config.BREAK_SUGGEST_MIN:
            # Repeat every BREAK_SUGGEST_REPEAT minutes
            last = self.active_alerts.get("break_suggest", 0.0)
            interval = config.BREAK_SUGGEST_REPEAT * 60
            if time.time() - last >= interval:
                self.trigger("break_suggest")

    def update_good_frame(self):
        self._good_frames += 1
        if self._good_frames >= config.HYSTERESIS_FRAMES:
            if self.current_level > AlertLevel.NORMAL:
                self.current_level = AlertLevel(int(self.current_level) - 1)
                self._good_frames  = 0

    @property
    def display_message(self) -> Optional[str]:
        if time.time() < self.message_until:
            return self.current_message
        return None

    @property
    def level_color(self) -> tuple:
        return {
            AlertLevel.NORMAL:   config.COLOR_NORMAL,
            AlertLevel.MILD:     config.COLOR_ACCENT,
            AlertLevel.ALERT:    config.COLOR_WARN,
            AlertLevel.WARNING:  config.COLOR_WARN,
            AlertLevel.CRITICAL: config.COLOR_CRITICAL,
        }.get(self.current_level, config.COLOR_INFO)

    def session_summary(self) -> Dict:
        counts = Counter(a[1] for a in self.alert_history)
        return {
            "total_alerts": len(self.alert_history),
            "alert_counts": dict(counts),
            "peak_level":   max((a[2] for a in self.alert_history), default="NORMAL"),
        }

    def reset_level(self):
        self.current_level = AlertLevel.NORMAL
        self._good_frames  = 0
