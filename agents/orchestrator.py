from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

from agents.safety_agent import safety_node
from agents.memory_agent import memory_node
from agents.prediction_agent import prediction_node
from agents.coaching_agent import coaching_node

class AgentState(TypedDict):
    driver_id: str
    session_id: str
    metrics: dict
    fatigue_score: float
    fatigue_level: str
    active_alerts: list
    last_agent: str
    reasoning: str
    memory_context: str
    action_taken: str
    turn: int

def route(state: AgentState) -> str:
    """Conditional routing based on fatigue score."""
    fs = state['fatigue_score']
    if fs >= 70:
        return 'safety_agent'      # Critical
    if fs >= 45:
        return 'prediction_agent'  # Warning
    if fs >= 25:
        return 'coaching_agent'    # Mild
    return 'memory_agent'          # Normal

def create_orchestrator():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("safety_agent", safety_node)
    workflow.add_node("memory_agent", memory_node)
    workflow.add_node("prediction_agent", prediction_node)
    workflow.add_node("coaching_agent", coaching_node)
    
    # Entry point relies on conditional routing
    workflow.set_conditional_entry_point(
        route,
        {
            "safety_agent": "safety_agent",
            "prediction_agent": "prediction_agent",
            "coaching_agent": "coaching_agent",
            "memory_agent": "memory_agent"
        }
    )
    
    # After any agent acts, the turn ends
    workflow.add_edge("safety_agent", END)
    workflow.add_edge("prediction_agent", END)
    workflow.add_edge("coaching_agent", END)
    workflow.add_edge("memory_agent", END)
    
    return workflow.compile()
