import uuid
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.working_memory import WorkingMemory

class MemoryManager:
    """
    Unified API for all 3 memory layers.
    Passed to the Orchestrator and Agents.
    """
    def __init__(self, driver_id: str):
        self.driver_id = driver_id
        self.session_id = str(uuid.uuid4())
        
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.working = WorkingMemory(
            session_id=self.session_id,
            driver_id=self.driver_id
        )
        
        # Initialize session in DB
        self.episodic.get_or_create_driver(self.driver_id)
        self.episodic.create_session(self.session_id, self.driver_id)
        
    def close(self):
        self.episodic.end_session(self.session_id)
