from langchain.tools import tool

@tool
def save_event(driver_id: str, event_type: str, severity: str, reasoning: str) -> str:
    """
    Saves a safety event to the driver's episodic memory.
    """
    print(f"[MEMORY] Saving event '{event_type}' ({severity}) for {driver_id}")
    return "Event saved successfully."

@tool
def query_history(driver_id: str, query: str) -> str:
    """
    Queries the semantic memory for past patterns or similar events.
    Args:
        driver_id: ID of the driver.
        query: Search query (e.g., 'drowsy events', 'past warnings').
    """
    print(f"[MEMORY] Querying history for {driver_id}: {query}")
    # Simulated response
    return "No similar severe events found in recent history."

@tool
def update_profile(driver_id: str, observation: str) -> str:
    """
    Updates the driver's behavioral profile with a new observation.
    """
    print(f"[MEMORY] Updating profile for {driver_id}: {observation}")
    return "Profile updated."
