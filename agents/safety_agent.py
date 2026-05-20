from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from agents.tools.alert_tools import trigger_alert, voice_warn, suggest_break, escalate, dismiss
from agents.tools.memory_tools import query_history

SAFETY_SYSTEM_PROMPT = """
You are an expert driver safety AI operating in real-time.
You have access to the driver's current biometric data and their full driving history.

Your job:
1. ASSESS the current risk level using provided metrics + memory context
2. DECIDE the most appropriate intervention (if any)
3. EXPLAIN your reasoning in 1 sentence
4. EXECUTE the intervention via the available tools

Rules:
- Be direct, brief, non-alarming unless critical
- Consider the driver's baseline from their profile
- Compare to similar past events in their history
- A drowsy driver who has pulled over before needs a harder push
- Never alert for conditions that are normal for this driver (from profile)

Current Metrics: {metrics}
Fatigue Score: {fatigue_score}
Memory Context: {memory_context}
"""

def get_safety_agent():
    llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
    tools = [trigger_alert, voice_warn, suggest_break, escalate, dismiss, query_history]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SAFETY_SYSTEM_PROMPT),
        ("human", "Assess the situation and take action if needed.")
    ])
    
    return prompt | llm_with_tools

def safety_node(state: dict):
    print(f"\n[ORCHESTRATOR] Routing to Safety Agent (Fatigue: {state['fatigue_score']})")
    agent = get_safety_agent()
    response = agent.invoke({
        "metrics": state["metrics"],
        "fatigue_score": state["fatigue_score"],
        "memory_context": state.get("memory_context", "None")
    })
    
    # Process tool calls if any
    action_taken = "No action"
    if response.tool_calls:
        action_taken = f"Executed {len(response.tool_calls)} tools."
        # In a full implementation, we'd execute the tools here via ToolNode
    
    return {
        "last_agent": "safety_agent",
        "reasoning": response.content if response.content else "Used tools.",
        "action_taken": action_taken,
        "turn": state["turn"] + 1
    }
