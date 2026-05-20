"""
Dashboard Overlay V2 — Modern HUD

Complete redesign with:
  - Circular fatigue score gauge (big, prominent)
  - EAR waveform strip chart (last 150 frames)
  - Gaze attention heatmap thumbnail
  - Head pose compass rose
  - Heart rate display with pulse icon
  - Color-coded component breakdown
  - Animated alert banner with pulsing border
  - PERCLOS trend indicator (↑ rising / → stable / ↓ falling)
  - Smooth gradient panel rendering
"""

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import time
import math
from typing import Dict
import config
from alerts.alert_engine import AlertLevel
from collections import deque


# ── Drawing Utilities ────────────────────────────────────────────────────────


def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness == -1:  # filled
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
        cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
        cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
        cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
        cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)


def draw_arc_gauge(
    img,
    center,
    radius,
    value,
    max_val,
    color,
    bg_color=(50, 50, 60),
    start_angle=-210,
    sweep=240,
    thickness=10,
):
    """
    Draw a circular arc gauge.
    start_angle: degrees (0=right, 90=down in OpenCV)
    sweep: total sweep degrees
    """
    cx, cy = center
    # Background arc
    cv2.ellipse(
        img,
        (cx, cy),
        (radius, radius),
        0,
        -start_angle,
        -(start_angle - sweep),
        bg_color,
        thickness,
        cv2.LINE_AA,
    )
    # Value arc
    val_sweep = int(sweep * min(1.0, max(0.0, value / max_val)))
    if val_sweep > 0:
        cv2.ellipse(
            img,
            (cx, cy),
            (radius, radius),
            0,
            -start_angle,
            -(start_angle - val_sweep),
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_text_centered(img, text, center, font, scale, color, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = center[0] - tw // 2
    y = center[1] + th // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ── Main Dashboard Class ─────────────────────────────────────────────────────


class DashboardOverlay:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_MONO = cv2.FONT_HERSHEY_PLAIN
    FS = 0.38  # font scale small
    FM = 0.52  # font scale medium
    FL = 0.70  # font scale large

    def __init__(
        self, frame_w: int = config.FRAME_WIDTH, frame_h: int = config.FRAME_HEIGHT
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.panel_w = config.DASHBOARD_WIDTH
        self.total_w = frame_w + self.panel_w

        # Session
        self.session_start = time.time()
        self.frame_count = 0

        # FPS tracking
        self._fps_t = time.time()
        self._fps_n = 0
        self.fps = 0.0

        # EAR waveform buffer
        self._ear_wave = deque(maxlen=150)

        # Alert pulse animation
        self._pulse_phase = 0.0

        print(f"[Overlay V2] Modern HUD initialized. Panel: {self.panel_w}px")

    def render(
        self,
        frame: np.ndarray,
        metrics: Dict,
        alert_engine=None,
        fatigue_engine=None,
        calibration=None,
    ) -> np.ndarray:
        """
        Render full V2 dashboard.

        metrics keys: ear, mar, pitch, yaw, roll, perclos, perclos_trend,
                      gaze_x, gaze_y, gaze_dir, ear_state, head_state,
                      perclos_state, phone_detected, blinks, yawns,
                      drowsy_ev, distract_ev, face_found,
                      fatigue_score, fatigue_level, fatigue_components,
                      hr_bpm, hr_confidence, hr_state,
                      heatmap_img (optional np.ndarray 80×60)
        """
        self.frame_count += 1
        self._update_fps()

        if config.SHOW_HEAD_AXES and "head_pose_obj" in metrics:
            metrics["head_pose_obj"].draw_axes(frame)

        # Update EAR waveform
        self._ear_wave.append(metrics.get("ear", config.EAR_OPEN_BASELINE))

        # ── Create panel ──────────────────────────────────────────────────────
        panel = np.full(
            (self.frame_h, self.panel_w, 3), config.PANEL_BG, dtype=np.uint8
        )

        alert_level = alert_engine.current_level if alert_engine else AlertLevel.NORMAL
        status_color = alert_engine.level_color if alert_engine else config.COLOR_NORMAL
        fatigue_score = metrics.get("fatigue_score", 0.0)
        fatigue_level = metrics.get("fatigue_level", "normal")

        y = 0

        # ── Top header ────────────────────────────────────────────────────────
        y = self._draw_header(panel, y, status_color, alert_level)

        # ── Fatigue gauge ─────────────────────────────────────────────────────
        if config.SHOW_FATIGUE_GAUGE:
            y = self._draw_fatigue_gauge(
                panel, y, fatigue_score, fatigue_level, metrics, fatigue_engine
            )

        # ── EAR + PERCLOS bars ────────────────────────────────────────────────
        y = self._draw_metric_bar(
            panel,
            y,
            "EAR",
            metrics.get("ear", 0.0),
            0.0,
            0.45,
            config.EAR_THRESHOLD,
            config.COLOR_EAR,
            invert=True,
        )
        perclos_val = metrics.get("perclos", 0.0) * 100
        trend_str = metrics.get("perclos_trend", "→")
        y = self._draw_metric_bar(
            panel,
            y,
            f"PERCLOS {trend_str}",
            perclos_val,
            0.0,
            30.0,
            config.PERCLOS_ALERT_THRESH * 100,
            config.COLOR_ACCENT,
            invert=False,
            fmt="{:.1f}%",
        )

        # ── Heart rate ────────────────────────────────────────────────────────
        y = self._draw_heart_rate(panel, y, metrics)

        # ── EAR Waveform ──────────────────────────────────────────────────────
        if config.SHOW_EAR_WAVEFORM:
            y = self._draw_ear_waveform(panel, y)

        # ── Head pose ─────────────────────────────────────────────────────────
        y = self._draw_head_pose(panel, y, metrics)

        # ── Gaze + heatmap ────────────────────────────────────────────────────
        y = self._draw_gaze(panel, y, metrics)

        # ── Events ───────────────────────────────────────────────────────────
        y = self._draw_events(panel, y, metrics)

        # ── Phone alert ───────────────────────────────────────────────────────
        if metrics.get("phone_detected", False):
            y = self._draw_phone_banner(panel, y)

        # ── Calibration overlay on panel ──────────────────────────────────────
        if calibration and calibration.calibrating:
            self._draw_calibration_overlay(panel, calibration)

        # ── Alert banner on camera frame ──────────────────────────────────────
        if alert_engine:
            msg = alert_engine.display_message
            if msg:
                self._draw_alert_banner(frame, msg, status_color)

        # No-face indicator
        if not metrics.get("face_found", True):
            self._draw_no_face(frame)

        # ── Compose final frame ───────────────────────────────────────────────
        if frame.shape[0] != self.frame_h or frame.shape[1] != self.frame_w:
            frame = cv2.resize(frame, (self.frame_w, self.frame_h))

        dashboard = np.concatenate([frame, panel], axis=1)
        cv2.line(
            dashboard, (self.frame_w, 0), (self.frame_w, self.frame_h), (45, 45, 60), 1
        )
        return dashboard

    # ── Sub-rendering methods ─────────────────────────────────────────────────

    def _draw_header(self, panel, y, status_color, alert_level) -> int:
        # Header bar
        cv2.rectangle(panel, (0, 0), (self.panel_w, 44), config.PANEL_HEADER, -1)
        cv2.putText(
            panel,
            "DMS V2",
            (10, 17),
            self.FONT,
            self.FM,
            (220, 220, 230),
            1,
            cv2.LINE_AA,
        )

        elapsed = time.time() - self.session_start
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        cv2.putText(
            panel,
            f"{h:02d}:{m:02d}:{s:02d}  {self.fps:.0f}fps",
            (10, 36),
            self.FONT,
            self.FS,
            (120, 120, 140),
            1,
            cv2.LINE_AA,
        )

        # Status strip
        level_names = {0: "NORMAL", 1: "MILD", 2: "ALERT", 3: "WARNING", 4: "CRITICAL"}
        lname = level_names.get(int(alert_level), "NORMAL")
        draw_rounded_rect(
            panel, (8, 46), (self.panel_w - 8, 70), status_color, radius=5
        )
        draw_text_centered(
            panel,
            f"◆ {lname}",
            (self.panel_w // 2, 58),
            self.FONT,
            self.FS,
            (10, 10, 10),
            2,
        )
        return 76

    def _draw_fatigue_gauge(
        self, panel, y, score, level, metrics, fatigue_engine
    ) -> int:
        """Large circular fatigue score gauge."""
        cx = self.panel_w // 2
        cy = y + 65
        R = 52

        # Color based on score
        if score < config.FATIGUE_MILD:
            gauge_color = config.COLOR_FATIGUE_LOW
        elif score < config.FATIGUE_WARN:
            gauge_color = config.COLOR_ACCENT
        elif score < config.FATIGUE_CRITICAL:
            gauge_color = config.COLOR_FATIGUE_MID
        else:
            gauge_color = config.COLOR_FATIGUE_HIGH

        # Background track
        draw_arc_gauge(panel, (cx, cy), R, 100, 100, (50, 50, 60), thickness=10)
        # Score arc
        draw_arc_gauge(panel, (cx, cy), R, score, 100, gauge_color, thickness=10)

        # Score number
        draw_text_centered(
            panel, f"{int(score)}", (cx, cy - 5), self.FONT, self.FL, gauge_color, 2
        )
        draw_text_centered(
            panel, "FATIGUE", (cx, cy + 16), self.FONT, 0.30, (150, 150, 170), 1
        )

        # Trend
        if fatigue_engine:
            trend = fatigue_engine.trend_str
            draw_text_centered(
                panel, trend, (cx, cy + 30), self.FONT, self.FS, (160, 160, 180), 1
            )

        # Component dots row
        if fatigue_engine and hasattr(fatigue_engine, "components"):
            comps = fatigue_engine.components
            keys = list(comps.keys())
            dot_y = cy + 48
            spacing = self.panel_w // (len(keys) + 1)
            for i, k in enumerate(keys):
                v = comps[k] / 100.0
                dx = spacing * (i + 1)
                c = (
                    config.COLOR_FATIGUE_HIGH
                    if v > 0.6
                    else (config.COLOR_ACCENT if v > 0.3 else config.COLOR_FATIGUE_LOW)
                )
                cv2.circle(panel, (dx, dot_y), 5, c, -1)
                cv2.putText(
                    panel,
                    k[0],
                    (dx - 4, dot_y + 16),
                    self.FONT,
                    0.28,
                    (140, 140, 160),
                    1,
                )

        return cy + 62

    def _draw_metric_bar(
        self, panel, y, label, value, lo, hi, threshold, color, invert, fmt="{:.3f}"
    ) -> int:
        bx, bw, bh = 10, self.panel_w - 20, 9
        # Label
        cv2.putText(panel, label, (bx, y), self.FONT, self.FS, (160, 160, 180), 1)
        cv2.putText(
            panel,
            fmt.format(value),
            (self.panel_w - 60, y),
            self.FONT,
            self.FS,
            color,
            1,
        )
        y += 13
        # Background
        draw_rounded_rect(panel, (bx, y), (bx + bw, y + bh), (45, 45, 55), 3)
        # Fill
        ratio = np.clip((value - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        fw = max(1, int(bw * ratio))
        thresh_ratio = (threshold - lo) / (hi - lo + 1e-6)
        bad = (ratio < thresh_ratio) if invert else (ratio > thresh_ratio)
        bar_color = config.COLOR_CRITICAL if bad else config.COLOR_NORMAL
        draw_rounded_rect(panel, (bx, y), (bx + fw, y + bh), bar_color, 3)
        # Threshold mark
        tx = bx + int(bw * thresh_ratio)
        cv2.line(panel, (tx, y - 2), (tx, y + bh + 2), (220, 220, 100), 2)
        return y + bh + 10

    def _draw_heart_rate(self, panel, y, metrics) -> int:
        hr = metrics.get("hr_bpm", 0.0)
        conf = metrics.get("hr_confidence", 0.0)
        state = metrics.get("hr_state", "unknown")
        y += 2

        if hr > 0 and conf > 0.2:
            # Pulse symbol
            hr_color = (
                config.COLOR_HR
                if state == "normal"
                else config.COLOR_CRITICAL
                if state == "elevated"
                else config.COLOR_WARN
            )
            cv2.putText(
                panel, "♥", (10, y + 12), self.FONT, self.FM, hr_color, 1, cv2.LINE_AA
            )
            cv2.putText(
                panel,
                f"HR: {int(hr)} bpm",
                (32, y + 12),
                self.FONT,
                self.FM,
                hr_color,
                1,
                cv2.LINE_AA,
            )
            # Confidence dot
            conf_r = max(2, int(conf * 8))
            cv2.circle(panel, (self.panel_w - 15, y + 8), conf_r, (100, 200, 100), -1)
        else:
            cv2.putText(
                panel,
                "♥ HR: measuring...",
                (10, y + 12),
                self.FONT,
                self.FS,
                (90, 90, 100),
                1,
            )
        return y + 22

    def _draw_ear_waveform(self, panel, y) -> int:
        """Mini EAR strip chart (150 frames)."""
        y += 4
        cv2.putText(
            panel, "EAR WAVEFORM", (10, y), self.FONT, self.FS, (140, 140, 160), 1
        )
        y += 13

        ww, wh = self.panel_w - 20, 32
        # Background
        draw_rounded_rect(panel, (10, y), (10 + ww, y + wh), (30, 30, 40), 4)

        ear_vals = list(self._ear_wave)
        if len(ear_vals) >= 2:
            # Draw threshold line
            ty = y + wh - int(config.EAR_THRESHOLD / 0.45 * wh)
            cv2.line(panel, (10, ty), (10 + ww, ty), (180, 180, 60), 1)

            # Draw waveform
            step = ww / max(1, len(ear_vals) - 1)
            pts = []
            for i, v in enumerate(ear_vals):
                px = int(10 + i * step)
                py = int(y + wh - (v / 0.45) * wh)
                py = max(y + 1, min(y + wh - 1, py))
                pts.append((px, py))

            for i in range(len(pts) - 1):
                ear_v = ear_vals[i]
                lc = (
                    config.COLOR_CRITICAL
                    if ear_v < config.EAR_THRESHOLD
                    else config.COLOR_EAR
                )
                cv2.line(panel, pts[i], pts[i + 1], lc, 1, cv2.LINE_AA)

        return y + wh + 8

    def _draw_head_pose(self, panel, y, metrics) -> int:
        y += 4
        cv2.putText(panel, "HEAD POSE", (10, y), self.FONT, self.FS, (140, 140, 160), 1)
        y += 14

        pitch = metrics.get("pitch", 0.0)
        yaw = metrics.get("yaw", 0.0)
        roll = metrics.get("roll", 0.0)
        state = metrics.get("head_state", "normal")

        state_color = (
            config.COLOR_CRITICAL
            if state in ("distracted", "nod", "jerk", "sway")
            else config.COLOR_NORMAL
        )

        for label, val, thresh in [
            (f"P {pitch:+.1f}°", pitch, config.PITCH_DOWN_THRESH),
            (f"Y {yaw:+.1f}°", yaw, config.YAW_THRESH),
            (f"R {roll:+.1f}°", roll, config.ROLL_THRESH),
        ]:
            c = config.COLOR_CRITICAL if abs(val) > thresh else config.COLOR_INFO
            cv2.putText(panel, label, (12, y), self.FONT, self.FS, c, 1)
            y += 14

        # Compass visualization (mini bird's-eye view of yaw)
        cx, cy_c = self.panel_w - 28, y - 21
        cv2.circle(panel, (cx, cy_c), 16, (45, 45, 60), -1)
        cv2.circle(panel, (cx, cy_c), 16, (70, 70, 90), 1)
        # Yaw arrow
        angle_rad = math.radians(-yaw)  # negative = screen clockwise
        dx = int(12 * math.sin(angle_rad))
        dy = int(-12 * math.cos(angle_rad))
        cv2.arrowedLine(
            panel, (cx, cy_c), (cx + dx, cy_c + dy), state_color, 2, tipLength=0.4
        )

        # State label
        cv2.putText(
            panel, f"[{state.upper()}]", (10, y), self.FONT, self.FS, state_color, 1
        )
        return y + 14

    def _draw_gaze(self, panel, y, metrics) -> int:
        y += 4
        gaze_dir = metrics.get("gaze_dir", "center")
        gc = config.COLOR_WARN if gaze_dir != "center" else config.COLOR_NORMAL
        cv2.putText(
            panel, f"GAZE: {gaze_dir.upper()}", (10, y), self.FONT, self.FS, gc, 1
        )
        y += 14

        # Gaze indicator circle
        cir_x = 34
        gaze_x = metrics.get("gaze_x", 0.0)
        gaze_y = metrics.get("gaze_y", 0.0)
        R = 22
        cv2.circle(panel, (cir_x, y + R), R, (50, 50, 65), -1)
        cv2.circle(panel, (cir_x, y + R), R, gc, 1)
        dot_x = int(cir_x + gaze_x * R * 0.7)
        dot_y = int(y + R + gaze_y * R * 0.7)
        cv2.circle(panel, (dot_x, dot_y), 5, config.COLOR_IRIS, -1)
        cv2.line(panel, (cir_x - R, y + R), (cir_x + R, y + R), (50, 50, 60), 1)
        cv2.line(panel, (cir_x, y), (cir_x, y + 2 * R), (50, 50, 60), 1)

        # Heatmap thumbnail (if available)
        heatmap = metrics.get("heatmap_img")
        if heatmap is not None and config.SHOW_HEATMAP:
            hm_w, hm_h = 80, 48
            hy = int(y)
            if hy + hm_h > self.frame_h:
                hm_h = max(0, self.frame_h - hy)
            if hm_h > 0:
                hm = cv2.resize(heatmap, (hm_w, hm_h))
                hx = self.panel_w - hm_w - 8
                panel[hy : hy + hm_h, hx : hx + hm_w] = hm
                cv2.rectangle(panel, (hx, hy), (hx + hm_w, hy + hm_h), (70, 70, 90), 1)
                cv2.putText(
                    panel,
                    "ATTN",
                    (hx + 2, hy + hm_h - 3),
                    self.FONT,
                    0.28,
                    (180, 180, 200),
                    1,
                )

        return y + R * 2 + 8

    def _draw_events(self, panel, y, metrics) -> int:
        y += 4
        cv2.putText(
            panel, "SESSION EVENTS", (10, y), self.FONT, self.FS, (130, 130, 150), 1
        )
        y += 14

        events = [
            ("Blinks", metrics.get("blinks", 0), config.COLOR_INFO),
            ("Yawns", metrics.get("yawns", 0), config.COLOR_ACCENT),
            ("Drowsy", metrics.get("drowsy_ev", 0), config.COLOR_CRITICAL),
            ("Distract", metrics.get("distract_ev", 0), config.COLOR_WARN),
        ]
        col_w = (self.panel_w - 12) // 2
        for i, (label, val, color) in enumerate(events):
            cx_e = 10 + (i % 2) * col_w
            draw_rounded_rect(
                panel, (cx_e, y), (cx_e + col_w - 4, y + 20), (30, 30, 40), 3
            )
            cv2.putText(
                panel,
                f"{label}",
                (cx_e + 4, y + 9),
                self.FONT,
                self.FS,
                (140, 140, 160),
                1,
            )
            cv2.putText(
                panel,
                str(val),
                (cx_e + col_w - 22, y + 13),
                self.FONT,
                self.FM,
                color,
                1,
            )
            if (i + 1) % 2 == 0:
                y += 24
        return y + 4

    def _draw_phone_banner(self, panel, y) -> int:
        self._pulse_phase = (self._pulse_phase + 0.15) % (2 * math.pi)
        alpha = int(180 + 75 * math.sin(self._pulse_phase))
        color = (0, 0, min(255, alpha))
        draw_rounded_rect(panel, (5, y), (self.panel_w - 5, y + 22), color, 4)
        draw_text_centered(
            panel,
            "📱 PHONE DETECTED!",
            (self.panel_w // 2, y + 11),
            self.FONT,
            self.FS,
            (255, 255, 255),
            2,
        )
        return y + 28

    def _draw_alert_banner(self, frame, msg, color):
        h, w = frame.shape[:2]
        bh = 48
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bh), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        (tw, th), _ = cv2.getTextSize(msg, self.FONT, self.FL, 2)
        cv2.putText(
            frame,
            msg,
            ((w - tw) // 2, h - bh // 2 + th // 2),
            self.FONT,
            self.FL,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_no_face(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            "NO DRIVER DETECTED",
            (w // 2 - 130, h // 2),
            self.FONT,
            self.FL,
            config.COLOR_CRITICAL,
            2,
            cv2.LINE_AA,
        )

    def _draw_calibration_overlay(self, panel, calibration):
        """Show calibration progress bar overlaid on panel."""
        progress = calibration.progress
        bar_y = self.frame_h - 50
        bw = self.panel_w - 20
        # Dark overlay
        cv2.rectangle(
            panel, (0, bar_y - 20), (self.panel_w, self.frame_h), (20, 20, 30), -1
        )
        draw_text_centered(
            panel,
            calibration.status_message,
            (self.panel_w // 2, bar_y - 8),
            self.FONT,
            self.FS,
            config.COLOR_ACCENT,
            1,
        )
        # Progress bar
        draw_rounded_rect(panel, (10, bar_y), (10 + bw, bar_y + 16), (40, 40, 50), 4)
        fw = int(bw * progress)
        if fw > 4:
            draw_rounded_rect(
                panel, (10, bar_y), (10 + fw, bar_y + 16), config.COLOR_ACCENT, 4
            )
        pct = f"{int(progress * 100)}%"
        draw_text_centered(
            panel,
            pct,
            (self.panel_w // 2, bar_y + 8),
            self.FONT,
            self.FS,
            (220, 220, 230),
            1,
        )

    def _update_fps(self):
        self._fps_n += 1
        elapsed = time.time() - self._fps_t
        if elapsed >= 1.0:
            self.fps = self._fps_n / elapsed
            self._fps_n = 0
            self._fps_t = time.time()
