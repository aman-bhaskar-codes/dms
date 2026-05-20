import time
import os
import chromadb
import config
from langchain_community.embeddings import OllamaEmbeddings

class SemanticMemory:
    """
    Vector store for driver history, behavior patterns, and past coaching.
    Enables RAG for the Voice Agent.
    """
    def __init__(self, db_path: str = config.CHROMA_DB_PATH):
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        
        # We wrap the ollama embedding call because chromadb expects an embedding function
        class OllamaEmbeddingFunction:
            def __init__(self):
                # Using the standard langchain community package (ignore deprecation for now)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.embedder = OllamaEmbeddings(
                        model=config.OLLAMA_EMBED_MODEL,
                        base_url=config.OLLAMA_HOST
                    )
            
            def __call__(self, input: list[str]) -> list[list[float]]:
                # OllamaEmbeddings has embed_documents for batch
                return self.embedder.embed_documents(input)
                
            def name(self) -> str:
                return "OllamaEmbeddingFunction"

        self.embedding_func = OllamaEmbeddingFunction()
        
        # Collection for high-level session summaries and insights
        self.insights = self.client.get_or_create_collection(
            name=f"{config.CHROMA_COLLECTION_PREFIX}_insights",
            embedding_function=self.embedding_func
        )
        # Collection for specific driver preferences (e.g. "prefers no sound alerts")
        self.preferences = self.client.get_or_create_collection(
            name=f"{config.CHROMA_COLLECTION_PREFIX}_preferences",
            embedding_function=self.embedding_func
        )

    def add_insight(self, driver_id: str, text: str, metadata: dict = None):
        """Embeds and stores a new insight about the driver."""
        meta = metadata or {}
        meta['driver_id'] = driver_id
        
        self.insights.add(
            documents=[text],
            metadatas=[meta],
            ids=[f"insight_{driver_id}_{int(time.time()*1000)}"]
        )

    def query_history(self, driver_id: str, query: str, k: int = 3) -> list:
        """Retrieves relevant past insights based on current context."""
        results = self.insights.query(
            query_texts=[query],
            n_results=k,
            where={"driver_id": driver_id}
        )
        return results['documents'][0] if results['documents'] else []
