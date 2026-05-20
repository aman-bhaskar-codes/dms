from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from agents.tools.memory_tools import save_event, update_profile

MEMORY_SYSTEM_PROMPT = """
You are the Memory Agent for a driver safety system.
Your job is to observe the driver's state and selectively log meaningful patterns or events to long-term memory.

Rules:
- Do NOT log every frame.
- Log if you notice a sustained shift (e.g., "blink rate has been rising for 5 mins").
- Log if a safety event occurred but wasn't critical enough for the Safety Agent.
- Use tools to save your observations.

Current Metrics: {metrics}
Fatigue Score: {fatigue_score}
"""

def get_memory_agent():
    llm = ChatOllama(model="llama3.2:3b", temperature=0.0)
    tools = [save_event, update_profile]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", MEMORY_SYSTEM_PROMPT),
        ("human", "Observe the current state and log if necessary.")
    ])
    
    return prompt | llm_with_tools

def memory_node(state: dict):
    print(f"\n[ORCHESTRATOR] Routing to Memory Agent (Fatigue: {state['fatigue_score']})")
    agent = get_memory_agent()
    response = agent.invoke({
        "metrics": state["metrics"],
        "fatigue_score": state["fatigue_score"]
    })
    
    action_taken = "No action"
    if response.tool_calls:
        action_taken = f"Logged {len(response.tool_calls)} memories."
    
    return {
        "last_agent": "memory_agent",
        "reasoning": response.content if response.content else "Used tools.",
        "action_taken": action_taken,
        "turn": state["turn"] + 1
    }
