from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from memory.semantic_memory import SemanticMemory

RAG_SYSTEM_PROMPT = """
You are a driver safety assistant.
You have access to the driver's current status and their driving history.

CURRENT DRIVE STATUS:
Fatigue Score: {fatigue_score}/100
Active Alerts: {active_alerts}

RETRIEVED CONTEXT FROM DRIVER HISTORY:
{retrieved_context}

Instructions:
- Keep responses under 3 sentences.
- Be brief and direct.
- Reference past data when relevant.
- Do not say "I don't know", provide your best guidance based on the data.
"""

class RAGPipeline:
    def __init__(self, semantic_memory: SemanticMemory):
        self.memory = semantic_memory
        self.llm = ChatOllama(model="llama3.2:3b", temperature=0.3)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{query}")
        ])
        self.chain = self.prompt | self.llm
        
    def generate_response(self, driver_id: str, query: str, current_metrics: dict) -> str:
        # Retrieve context
        context_docs = self.memory.query_history(driver_id, query)
        context_str = "\n".join(context_docs) if context_docs else "No historical context available."
        
        # Generate response
        response = self.chain.invoke({
            "fatigue_score": current_metrics.get("fatigue_score", 0),
            "active_alerts": current_metrics.get("active_alerts", []),
            "retrieved_context": context_str,
            "query": query
        })
        
        return response.content
