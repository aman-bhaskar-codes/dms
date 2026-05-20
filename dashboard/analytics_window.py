"""
Analytics Window V2 — Live Matplotlib Dashboard
NEW in V2.

Opens a separate window with 6 real-time subplots:
  1. EAR over time (with blink markers)
  2. PERCLOS over time (with 15% line)
  3. Fatigue score over time (with level bands)
  4. Head yaw over time (with ±30° bands)
  5. Yawn / drowsy event timeline
  6. Heart rate over time

Runs in a background thread. Updates every 2 seconds.
Toggle with 'A' key.

Requires: matplotlib (pip install matplotlib)
"""

import threading
import time
from collections import deque
from typing import Dict
import config

try:
    import matplotlib
    matplotlib.use('TkAgg')          # or 'Qt5Agg' depending on system
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("[Analytics] matplotlib not available. Run: pip install matplotlib")


class AnalyticsWindow:
    """
    Background thread that maintains a live matplotlib analytics dashboard.
    Shares data via thread-safe deques.
    """

    HISTORY_SEC = 300   # 5 minutes of history

    def __init__(self, fps: int = config.TARGET_FPS):
        self.enabled   = config.ANALYTICS_ENABLED and MPL_AVAILABLE
        self.fps       = fps
        maxlen         = self.HISTORY_SEC * fps

        # Data buffers (thread-safe deques)
        self.t_buf       = deque(maxlen=maxlen)
        self.ear_buf     = deque(maxlen=maxlen)
        self.perclos_buf = deque(maxlen=maxlen)
        self.fatigue_buf = deque(maxlen=maxlen)
        self.yaw_buf     = deque(maxlen=maxlen)
        self.hr_buf      = deque(maxlen=maxlen)
        self.events      = deque(maxlen=200)   # (time, type, level)

        self._running  = False
        self._visible  = False
        self._thread   = None
        self._fig      = None
        self._start_t  = time.time()

        if not self.enabled:
            print("[Analytics] Disabled.")
        else:
            print("[Analytics] Ready. Press 'A' to open analytics window.")

    def push(self, metrics: Dict):
        """Push current frame metrics into buffers (call from main loop)."""
        if not self.enabled:
            return
        now = time.time() - self._start_t
        self.t_buf.append(now)
        self.ear_buf.append(metrics.get("ear",          0.0))
        self.perclos_buf.append(metrics.get("perclos",  0.0) * 100)
        self.fatigue_buf.append(metrics.get("fatigue_score", 0.0))
        self.yaw_buf.append(metrics.get("yaw",          0.0))
        self.hr_buf.append(metrics.get("hr_bpm",        0.0))

    def push_event(self, event_type: str, level: str):
        if not self.enabled:
            return
        self.events.append((time.time() - self._start_t, event_type, level))

    def toggle(self):
        """Toggle analytics window visibility."""
        if not self.enabled:
            print("[Analytics] Not available.")
            return
        if self._visible:
            self._close()
        else:
            self._open()

    def _open(self):
        if self._running:
            return
        self._running = True
        self._visible = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[Analytics] Window opened.")

    def _close(self):
        self._running = False
        self._visible = False
        if self._fig:
            try:
                plt.close(self._fig)
            except Exception:
                pass
        print("[Analytics] Window closed.")

    def _run(self):
        """Analytics window main loop (runs in background thread)."""
        plt.ion()
        self._fig, axes = plt.subplots(3, 2, figsize=(12, 8))
        self._fig.patch.set_facecolor('#111118')
        self._fig.suptitle('DMS V2 — Live Analytics', color='white',
                            fontsize=13, fontweight='bold')

        ax_ear, ax_perc = axes[0]
        ax_fat, ax_yaw  = axes[1]
        ax_ev,  ax_hr   = axes[2]

        for ax in axes.flat:
            ax.set_facecolor('#1a1a24')
            ax.tick_params(colors='#888899', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#333344')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        while self._running:
            try:
                t = list(self.t_buf)
                if len(t) < 2:
                    time.sleep(config.ANALYTICS_UPDATE_SEC)
                    continue

                ear   = list(self.ear_buf)
                perc  = list(self.perclos_buf)
                fat   = list(self.fatigue_buf)
                yaw   = list(self.yaw_buf)
                hr    = list(self.hr_buf)

                # Trim to same length
                n = min(len(t), len(ear), len(perc), len(fat), len(yaw), len(hr))
                t = t[-n:]
                ear = ear[-n:]
                perc = perc[-n:]
                fat = fat[-n:]
                yaw = yaw[-n:]
                hr = hr[-n:]

                # EAR plot
                ax_ear.clear()
                ax_ear.set_facecolor('#1a1a24')
                ax_ear.plot(t, ear, color='#d4c020', lw=1, label='EAR')
                ax_ear.axhline(config.EAR_THRESHOLD, color='#ff4040',
                               lw=0.8, ls='--', label=f'thresh {config.EAR_THRESHOLD}')
                ax_ear.set_ylim(0, 0.5)
                ax_ear.set_title('Eye Aspect Ratio',
                                                            color='white', fontsize=9)
                ax_ear.legend(fontsize=7, labelcolor='white', facecolor='#222')

                # PERCLOS plot
                ax_perc.clear()
                ax_perc.set_facecolor('#1a1a24')
                ax_perc.fill_between(t, perc, alpha=0.4, color='#e05000')
                ax_perc.plot(t, perc, color='#ff8040', lw=1)
                ax_perc.axhline(config.PERCLOS_ALERT_THRESH * 100,
                                color='#ff2020', lw=0.8, ls='--')
                ax_perc.axhline(config.PERCLOS_WARN_THRESH * 100,
                                color='#ffa020', lw=0.8, ls='--')
                ax_perc.set_ylim(0, 35)
                ax_perc.set_title('PERCLOS (%)', color='white', fontsize=9)

                # Fatigue score plot
                ax_fat.clear()
                ax_fat.set_facecolor('#1a1a24')
                # Colored bands
                ax_fat.axhspan(0,  config.FATIGUE_MILD,     alpha=0.08, color='green')
                ax_fat.axhspan(config.FATIGUE_MILD,  config.FATIGUE_WARN,     alpha=0.08, color='yellow')
                ax_fat.axhspan(config.FATIGUE_WARN,  config.FATIGUE_CRITICAL, alpha=0.08, color='orange')
                ax_fat.axhspan(config.FATIGUE_CRITICAL, 100,  alpha=0.08, color='red')
                ax_fat.plot(t, fat, color='#60d0ff', lw=1.5)
                ax_fat.set_ylim(0, 100)
                ax_fat.set_title('Fatigue Score', color='white', fontsize=9)

                # Yaw plot
                ax_yaw.clear()
                ax_yaw.set_facecolor('#1a1a24')
                ax_yaw.plot(t, yaw, color='#80ffcc', lw=1)
                ax_yaw.axhline( config.YAW_THRESH, color='#ff4040', lw=0.8, ls='--')
                ax_yaw.axhline(-config.YAW_THRESH, color='#ff4040', lw=0.8, ls='--')
                ax_yaw.axhline(0, color='#555566', lw=0.5)
                ax_yaw.set_ylim(-60, 60)
                ax_yaw.set_title('Head Yaw (°)', color='white', fontsize=9)

                # Events timeline
                ax_ev.clear()
                ax_ev.set_facecolor('#1a1a24')
                ev_list = list(self.events)
                if ev_list:
                    ev_t    = [e[0] for e in ev_list]
                    ev_type = [e[1] for e in ev_list]
                    ev_lv   = [e[2] for e in ev_list]
                    colors_map = {"CRITICAL":"#ff2020","WARNING":"#ff8020",
                                  "ALERT":"#ffcc00","INFO":"#4488ff"}
                    ev_colors = [colors_map.get(lv, "#888") for lv in ev_lv]
                    ax_ev.scatter(ev_t, range(len(ev_t)), c=ev_colors,
                                  s=20, marker='|', zorder=3)
                    unique_types = list(dict.fromkeys(ev_type))[:6]
                    for i, et in enumerate(unique_types):
                        ax_ev.text(0.01, 0.85 - i * 0.13, et,
                                   transform=ax_ev.transAxes,
                                   color='#aaaacc', fontsize=7)
                ax_ev.set_title('Alert Events', color='white', fontsize=9)
                ax_ev.set_yticks([])

                # HR plot
                ax_hr.clear()
                ax_hr.set_facecolor('#1a1a24')
                hr_valid = [v for v in hr if v > 0]
                if hr_valid:
                    ax_hr.plot(t[-len(hr_valid):], hr_valid, color='#ff6080', lw=1.5)
                    ax_hr.axhline(config.RPPG_HR_STRESS_THRESH, color='#ff2020',
                                  lw=0.8, ls='--')
                    ax_hr.axhline(config.RPPG_HR_LOW_THRESH, color='#2060ff',
                                  lw=0.8, ls='--')
                    ax_hr.set_ylim(40, 160)
                ax_hr.set_title('Heart Rate (BPM)', color='white', fontsize=9)

                self._fig.canvas.draw()
                self._fig.canvas.flush_events()
                time.sleep(config.ANALYTICS_UPDATE_SEC)

            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"[Analytics] Plot error: {e}")
                time.sleep(1.0)

        plt.ioff()
