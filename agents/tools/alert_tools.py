from langchain.tools import tool

# Note: In a production app, these would query the actual MemoryManager or DB.
# For this orchestration layer, we simulate the interaction.

@tool
def trigger_alert(alert_type: str, severity: str) -> str:
    """
    Triggers a real-time alert in the vehicle.
    Args:
        alert_type: Type of alert (e.g., 'drowsy', 'distracted', 'break_suggestion')
        severity: 'info', 'warning', or 'critical'
    """
    print(f"[ALERT FIRED] {severity.upper()} - {alert_type}")
    return f"Alert {alert_type} triggered with severity {severity}"

@tool
def voice_warn(message: str) -> str:
    """
    Speaks a warning or suggestion directly to the driver using TTS.
    Args:
        message: The brief, direct message to speak to the driver.
    """
    print(f"[VOICE AGENT] Speaking: '{message}'")
    return "Voice warning delivered."

@tool
def suggest_break(minutes: int = 15) -> str:
    """
    Suggests a break to the driver via the HUD and Voice.
    Args:
        minutes: Suggested duration of the break.
    """
    print(f"[HUD] Suggesting a {minutes}-minute break.")
    return f"Suggested a {minutes} minute break."

@tool
def escalate() -> str:
    """
    Escalates the situation to fleet management if the driver is unresponsive.
    """
    print(f"[ESCALATION] Notifying fleet manager!")
    return "Escalated to fleet manager."

@tool
def dismiss(reason: str) -> str:
    """
    Dismisses the current alert context if determined to be a false positive or handled.
    Args:
        reason: Why the alert is being dismissed.
    """
    print(f"[ALERT DISMISSED] Reason: {reason}")
    return "Alert dismissed."
