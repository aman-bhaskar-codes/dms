"""
Ollama AI Reporter V2 — Enhanced Prompts + Contextual Mid-Drive Alerts

V2 improvements:
  - Richer prompt with fatigue score, HR, blink dynamics
  - Contextual mid-drive safety tips (triggered by fatigue level)
  - Better session report structure
  - Fallback summary if Ollama unavailable
"""

import requests
import time
import threading
from typing import Dict, Optional
import config


class AIReporter:
    def __init__(self):
        self.enabled          = config.OLLAMA_ENABLED
        self.model            = config.OLLAMA_MODEL
        self.last_report_time = 0.0
        self.last_report      = ""
        self.generating       = False
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            if not self._check_ollama():
                self.enabled = False
                print("[AIReporter V2] Ollama unavailable. Reports disabled.")
            else:
                print(f"[AIReporter V2] Ready. Model: {self.model}")
        else:
            print("[AIReporter V2] Disabled.")

    def _check_ollama(self) -> bool:
        try:
            resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(self.model.split(":")[0] in m for m in models):
                for fb in ["llama3.2:3b", "phi4-mini:latest", "qwen2.5:3b",
                           "llama3:8b", "mistral:7b"]:
                    if any(fb.split(":")[0] in m for m in models):
                        self.model = fb
                        print(f"[AIReporter V2] Using fallback model: {self.model}")
                        return True
                return False
            return True
        except Exception as e:
            print(f"[AIReporter V2] Connection error: {e}")
            return False

    def _call(self, prompt: str, system: str = "", max_tokens: int = 400) -> str:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = requests.post(
                f"{config.OLLAMA_HOST}/api/chat",
                json={
                    "model":    self.model,
                    "messages": messages,
                    "stream":   False,
                    "options": {"temperature": 0.25, "num_predict": max_tokens}
                },
                timeout=config.OLLAMA_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()["message"]["content"].strip()
            return f"[HTTP {resp.status_code}]"
        except requests.Timeout:
            return "[Timeout]"
        except Exception as e:
            return f"[Error: {e}]"

    def should_report(self) -> bool:
        return (self.enabled and not self.generating
                and time.time() - self.last_report_time
                >= config.OLLAMA_REPORT_INTERVAL_MIN * 60)

    def report_async(self, metrics: Dict, events: list):
        if not self.enabled or self.generating:
            return
        self._thread = threading.Thread(
            target=self._do_report, args=(metrics, events), daemon=True)
        self._thread.start()

    def _do_report(self, metrics: Dict, events: list):
        self.generating = True

        system = ("You are a driver safety AI. Give a direct 3-sentence safety "
                  "assessment with one clear action item. No markdown. Plain text.")

        ev_str = ", ".join(f"{e[1]}({e[2]})" for e in events[-20:]) or "none"

        prompt = f"""Driver monitoring summary (last {config.OLLAMA_REPORT_INTERVAL_MIN} min):

FATIGUE SCORE: {metrics.get('fatigue_score', 0):.1f}/100 ({metrics.get('fatigue_level', 'normal').upper()})
EAR: {metrics.get('ear', 0):.3f}  PERCLOS: {metrics.get('perclos', 0)*100:.1f}%
Head yaw: {metrics.get('yaw', 0):.1f}°  Pitch: {metrics.get('pitch', 0):.1f}°
Heart rate: {metrics.get('hr_bpm', 0):.0f} bpm ({metrics.get('hr_state', 'unknown')})
Slow blink ratio: {metrics.get('slow_blink_r', 0)*100:.1f}%
Yawns: {metrics.get('yawns', 0)}  Drowsy events: {metrics.get('drowsy_ev', 0)}
Phone detected: {metrics.get('phone_detected', False)}
Recent alerts: {ev_str}

Provide a 3-sentence safety assessment and one action recommendation."""

        report = self._call(prompt, system)
        self.last_report      = report
        self.last_report_time = time.time()
        self.generating       = False

        print(f"\n{'='*60}\n[AI REPORT — {time.strftime('%H:%M:%S')}]\n{report}\n{'='*60}\n")

    def generate_session_report(self, session_data: Dict) -> str:
        if not self.enabled:
            return self._fallback_report(session_data)

        stats   = session_data.get("frame_stats") or [None]*9
        events  = session_data.get("events", [])
        from collections import Counter
        ec = Counter(e[1] for e in events)

        def s(i, default=0.0):
            v = stats[i] if stats and i < len(stats) else None
            return v if v is not None else default

        system = ("You are a professional road safety analyst. Write a structured "
                  "driver safety report: safety rating 1–10, key risk factors, "
                  "and 3 specific recommendations. Plain text, no bullet points.")

        prompt = f"""End-of-session driver analysis:

Statistics:
- Avg EAR: {s(0):.3f}  Min EAR: {s(1):.3f}
- Avg PERCLOS: {s(2)*100:.1f}%  Peak PERCLOS: {s(3)*100:.1f}%
- Avg head yaw: {s(4):.1f}°  Avg pitch: {s(5):.1f}°
- Avg fatigue score: {s(6):.1f}  Peak fatigue: {s(7):.1f}
- Avg heart rate: {s(8):.0f if s(8,0)>0 else 'N/A'} bpm

Events:
- Drowsiness alerts: {ec.get('drowsy_warn',0) + ec.get('drowsy_crit',0)}
- Yawns: {ec.get('yawn',0)}
- Head nods: {ec.get('head_nod',0) + ec.get('head_jerk',0)}
- Distraction alerts: {ec.get('distracted',0)}
- Phone detections: {ec.get('phone',0)}
- Total alerts: {len(events)}

Generate a 6-sentence professional safety report with safety rating (1–10) and 3 actionable recommendations."""

        return self._call(prompt, system, max_tokens=600)

    def _fallback_report(self, session_data: Dict) -> str:
        """Plain statistics report when Ollama is unavailable."""
        events = session_data.get("events", [])
        stats  = session_data.get("frame_stats") or [None]*9
        def s(i, d=0.0):
            v = stats[i] if stats and i < len(stats) else None
            return v if v is not None else d

        return (
            f"Session Report (AI unavailable)\n"
            f"Avg EAR: {s(0):.3f}  |  Peak PERCLOS: {s(3)*100:.1f}%  |  "
            f"Peak Fatigue: {s(7):.1f}/100\n"
            f"Total alerts: {len(events)}\n"
            f"Install Ollama for AI-powered analysis: https://ollama.ai"
        )
