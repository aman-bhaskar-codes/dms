"""
Ollama API integration for DMS V4 (replaces Claude API).
Handles connection to local Ollama server using ollama package.
"""
import ollama
import json
import asyncio
from typing import Dict, Optional
import time
from config import settings


class OllamaClient:
    def __init__(self):
        self.enabled = settings.ollama_enabled
        self.fast_model = settings.ollama_model_fast
        self.smart_model = settings.ollama_model_smart
        self._circuit_breaker = False
        self._last_failure = 0.0

    async def get_driving_tip(self, context: str, metrics: dict) -> str:
        """Ask local Ollama for a short, coaching tip based on current state."""
        if not self.enabled:
            return "AI disabled. Stay alert!"

        if self._circuit_breaker:
            if time.time() - self._last_failure > 60.0:
                self._circuit_breaker = False
            else:
                return "Keep your eyes on the road and stay focused."

        sys_prompt = (
            "You are an AI driver coaching assistant. "
            "Provide a short, direct, and actionable tip (max 2 sentences) "
            "based on the current driver metrics and recent events. "
            "Do not greet, just give the tip."
        )
        user_prompt = f"Recent Context:\n{context}\n\nCurrent Metrics:\n{json.dumps(metrics, indent=2)}"

        try:
            # ollama.chat is sync by default, but we can wrap it in asyncio.to_thread
            response = await asyncio.wait_for(asyncio.to_thread(
                ollama.chat,
                model=self.fast_model,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            ), timeout=3.0)
            self._circuit_breaker = False
            return response['message']['content'].strip()
        except asyncio.TimeoutError:
            print("[Ollama] Tip generation timed out. Breaking circuit.")
            self._circuit_breaker = True
            self._last_failure = time.time()
            return "Please stay focused on the road ahead."
        except Exception as e:
            print(f"[Ollama] Error getting tip: {e}. Breaking circuit.")
            self._circuit_breaker = True
            self._last_failure = time.time()
            return "Focus on driving safely."

    async def generate_session_report(self, session_data: dict) -> str:
        """Generate a detailed end-of-session report."""
        if not self.enabled:
            return "Session complete. AI reports disabled."

        sys_prompt = (
            "You are a professional safety analyst for a commercial fleet. "
            "Review the provided session metrics and write a 3-paragraph summary report. "
            "Highlight any dangerous behaviors, fatigue trends, and give a final recommendation."
        )
        user_prompt = f"Session Data:\n{json.dumps(session_data, indent=2)}"

        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.smart_model,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"[Ollama] Error generating report: {e}")
            return "Failed to generate AI report due to connection error."
