from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.tools import tool
import json

@tool
def gen_session_report(session_id: str) -> str:
    """Generates a summary report for the specified session."""
    return f"Report generated for {session_id}."

REPORT_SYSTEM_PROMPT = """
You are the Report Agent for a driver safety system.
Your job is to summarize a driver's session into a concise report for fleet managers.
Include key metrics, total alerts, and overall risk assessment.
"""

def get_report_agent():
    llm = ChatOllama(model="llama3.2:3b", temperature=0.0)
    tools = [gen_session_report]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", REPORT_SYSTEM_PROMPT),
        ("human", "Generate a session report based on this data: {data}")
    ])
    
    return prompt | llm_with_tools

# The report agent is typically invoked at the end of a session, not directly in the real-time orchestrator loop.
