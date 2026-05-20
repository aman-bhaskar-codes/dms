"""
Ollama API integration for DMS V5 (replaces Claude API).
Handles connection to local Ollama server using httpx.
Includes Circuit Breaker to prevent hanging on server failure.
"""
import asyncio
import httpx
from typing import Optional
import time
from config import settings


class CircuitBreaker:
    """Prevents hammering a failing service."""
    def __init__(self, failure_threshold=3, timeout_sec=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.timeout = timeout_sec
        self._open_time = 0.0
    
    @property
    def is_open(self) -> bool:
        if self.failures >= self.threshold:
            if time.time() - self._open_time > self.timeout:
                self.failures = 0  # Try again after timeout
                return False
            return True
        return False
    
    def record_failure(self):
        self.failures += 1
        self._open_time = time.time()
    
    def record_success(self):
        self.failures = 0


class OllamaClient:
    def __init__(self):
        self.enabled = settings.ollama_enabled
        self.base_url = settings.ollama_host
        self.fast_model = settings.ollama_model_fast
        self.smart_model = settings.ollama_model_smart
        self._breaker = CircuitBreaker()
        self._client = httpx.AsyncClient(timeout=8.0)  # 8s hard timeout
    
    async def get_driving_tip(self, context: dict, metrics: dict, override_prompt: str = None) -> str:
        """Get a safety tip. Returns fallback string if Ollama is slow/down."""
        if not self.enabled:
            return "AI disabled. Stay alert!"
            
        if self._breaker.is_open:
            return self._get_fallback_tip(metrics)
        
        try:
            prompt = override_prompt if override_prompt else self._build_tip_prompt(context, metrics)
            response = await asyncio.wait_for(
                self._client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.fast_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": 60, "temperature": 0.3}
                    }
                ),
                timeout=6.0  # Absolute max 6 seconds
            )
            self._breaker.record_success()
            data = response.json()
            return data.get("response", "").strip()
            
        except (asyncio.TimeoutError, httpx.ConnectError, Exception) as e:
            self._breaker.record_failure()
            print(f"[Ollama] Failed: {type(e).__name__} — using fallback")
            return self._get_fallback_tip(metrics)
            
    async def generate_session_report(self, session_data: dict) -> str:
        """Generate a detailed end-of-session report."""
        if not self.enabled:
            return "Session complete. AI reports disabled."

        sys_prompt = (
            "You are a professional safety analyst for a commercial fleet. "
            "Review the provided session metrics and write a 3-paragraph summary report. "
            "Highlight any dangerous behaviors, fatigue trends, and give a final recommendation."
        )
        user_prompt = f"Session Data:\n{session_data}"

        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.smart_model,
                    "prompt": f"{sys_prompt}\n\n{user_prompt}",
                    "stream": False,
                    "options": {"temperature": 0.5}
                },
                timeout=15.0
            )
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"[Ollama] Error generating report: {e}")
            return "Failed to generate AI report due to connection error."
    
    def _get_fallback_tip(self, metrics: dict) -> str:
        """Rule-based fallback when Ollama is unavailable."""
        score = metrics.get("fatigue_score", 0)
        if score >= 70:
            return "Your fatigue is critical. Please pull over safely."
        elif score >= 45:
            return "Your fatigue is elevated. Consider a rest break."
        elif metrics.get("yawn_rate", 0) >= 3:
            return "Frequent yawning detected. Hydration and a short break may help."
        return "Drive safely. You're doing well."
    
    def _build_tip_prompt(self, context: dict, metrics: dict) -> str:
        summary = context.get('summary', 'first session') if isinstance(context, dict) else context
        return f"""You are a calm driver safety co-pilot.
Current data: fatigue={metrics.get('fatigue_score',0):.0f}/100, 
EAR={metrics.get('ear',0):.3f}, PERCLOS={metrics.get('perclos',0)*100:.1f}%,
drive_time={metrics.get('drive_minutes',0):.0f}min, yawns={metrics.get('yawn_rate',0)}.
History: {summary}.

Give ONE safety tip, max 20 words. No markdown. Direct and calm."""
