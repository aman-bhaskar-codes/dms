from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import json

INTENT_SYSTEM_PROMPT = """
You are the Intent Router for a driver safety voice assistant.
Classify the user's speech into one of these intents:
- QUERY_HISTORY (asking about past performance or events)
- QUERY_CURRENT (asking about current fatigue or metrics)
- COACHING (asking for advice or tips)
- CONVERSATION (general chatter, stating they are tired)
- EMERGENCY (indicating they might crash, medical emergency)

Reply ONLY with a JSON object: {"intent": "THE_INTENT"}
"""

class IntentRouter:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2:3b", temperature=0.0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", INTENT_SYSTEM_PROMPT),
            ("human", "{query}")
        ])
        self.chain = self.prompt | self.llm
        
    def route(self, query: str) -> str:
        response = self.chain.invoke({"query": query})
        try:
            # Simple parse
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            data = json.loads(content)
            return data.get("intent", "CONVERSATION")
        except:
            return "CONVERSATION"
