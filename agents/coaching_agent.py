from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from agents.tools.alert_tools import voice_warn
from agents.tools.memory_tools import query_history

COACHING_SYSTEM_PROMPT = """
You are the Coaching Agent for a driver safety system.
The driver is currently at a 'Mild' fatigue level.
Your job is to provide personalized, non-intrusive tips based on their history to improve their driving.

Rules:
- Keep tips very brief and positive.
- Example: "Try opening the window for some fresh air."
- If the driver's history indicates they don't like voice alerts, do not use the voice tool.

Current Metrics: {metrics}
Fatigue Score: {fatigue_score}
"""

def get_coaching_agent():
    llm = ChatOllama(model="llama3.2:3b", temperature=0.3)
    tools = [voice_warn, query_history]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", COACHING_SYSTEM_PROMPT),
        ("human", "Provide a coaching tip if appropriate.")
    ])
    
    return prompt | llm_with_tools

def coaching_node(state: dict):
    print(f"\n[ORCHESTRATOR] Routing to Coaching Agent (Fatigue: {state['fatigue_score']})")
    agent = get_coaching_agent()
    response = agent.invoke({
        "metrics": state["metrics"],
        "fatigue_score": state["fatigue_score"]
    })
    
    action_taken = "No action"
    if response.tool_calls:
        action_taken = f"Executed {len(response.tool_calls)} tools."
    
    return {
        "last_agent": "coaching_agent",
        "reasoning": response.content if response.content else "Used tools.",
        "action_taken": action_taken,
        "turn": state["turn"] + 1
    }
