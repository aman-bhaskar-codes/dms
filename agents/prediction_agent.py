from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from agents.tools.sensor_tools import forecast_fatigue
from agents.tools.alert_tools import suggest_break

PREDICTION_SYSTEM_PROMPT = """
You are the Prediction Agent for a driver safety system.
The driver is currently at a 'Warning' fatigue level.
Your job is to look at the fatigue forecast and determine if pre-emptive action is needed.

Rules:
- If the forecast indicates a critical state soon, suggest a break pre-emptively.
- If the forecast is stable, monitor quietly.

Current Metrics: {metrics}
Fatigue Score: {fatigue_score}
"""

def get_prediction_agent():
    llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
    tools = [forecast_fatigue, suggest_break]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PREDICTION_SYSTEM_PROMPT),
        ("human", "Check the forecast and act if necessary.")
    ])
    
    return prompt | llm_with_tools

def prediction_node(state: dict):
    print(f"\n[ORCHESTRATOR] Routing to Prediction Agent (Fatigue: {state['fatigue_score']})")
    agent = get_prediction_agent()
    response = agent.invoke({
        "metrics": state["metrics"],
        "fatigue_score": state["fatigue_score"]
    })
    
    action_taken = "No action"
    if response.tool_calls:
        action_taken = f"Executed {len(response.tool_calls)} tools."
    
    return {
        "last_agent": "prediction_agent",
        "reasoning": response.content if response.content else "Used tools.",
        "action_taken": action_taken,
        "turn": state["turn"] + 1
    }
