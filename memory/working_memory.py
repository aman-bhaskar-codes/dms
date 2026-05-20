from dataclasses import dataclass, field
from collections import deque
import time

@dataclass
class WorkingMemory:
    """
    Volatile state for the current driving session.
    Used by perception loop and orchestrator for fast read/write.
    """
    session_id: str
    driver_id: str
    start_time: float = field(default_factory=time.time)
    
    # Current perception state
    current_score: float = 0.0
    current_level: str = "normal"
    
    # Active alerts preventing agent spam
    active_alerts: dict = field(default_factory=dict)  # {'drowsy': timestamp}
    
    # Recent agent thoughts/actions
    recent_agent_logs: deque = field(default_factory=lambda: deque(maxlen=20))
    
    def log_agent_action(self, agent_name: str, action: str):
        self.recent_agent_logs.append(f"[{agent_name}] {action}")

    def can_trigger_alert(self, alert_type: str, cooldown: float) -> bool:
        last = self.active_alerts.get(alert_type, 0)
        if time.time() - last > cooldown:
            self.active_alerts[alert_type] = time.time()
            return True
        return False
