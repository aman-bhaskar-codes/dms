from langchain.tools import tool

@tool
def get_current_metrics() -> str:
    """
    Returns the current live biometric and perceptual metrics of the driver.
    """
    return "Fatigue: 42/100, EAR: 0.28, Blink Rate: Normal, Heart Rate: 72 bpm"

@tool
def forecast_fatigue() -> str:
    """
    Returns the predicted fatigue score 5 minutes into the future based on recent trends.
    """
    return "Predicted 5-min score: 48. Trend is slowly rising."
