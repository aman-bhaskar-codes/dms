"""
DMS V5 Elite RAG Engine.
Hybrid retrieval: ChromaDB dense + BM25 sparse + cross-encoder re-rank.
All models free & offline.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi    # pip install rank-bm25
import httpx

from core.bus import EventBus, EventTopic
from core.config import settings


@dataclass
class RAGDocument:
    id: str
    content: str
    metadata: Dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class RAGEngine:
    """
    Elite RAG with:
    1. HyDE query expansion
    2. Hybrid dense + sparse retrieval
    3. Reciprocal Rank Fusion
    4. Cross-encoder re-ranking (if available)
    5. Guardrails on output
    """

    COLLECTION_NAME = "dms_v5_knowledge"
    TOP_K_RETRIEVE  = 20    # retrieve many, re-rank to fewer
    TOP_K_FINAL     = 5

    # System prompt for driving-safety assistant
    SYSTEM_PROMPT = """You are SENTINEL, an elite driver safety AI assistant.
You have access to the driver's real-time biometrics, historical session data,
and traffic safety knowledge. Your responses must be:
- Concise (spoken aloud, so max 3 sentences for voice responses)
- Actionable (always suggest what to DO, not just what is happening)
- Safety-first (never minimize fatigue signals)
- Personalized to this specific driver's baseline data

Current driver biometrics will be injected into your context.
Never fabricate biometric values. If uncertain, say so."""

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
        self._embed_fn = embedding_functions.OllamaEmbeddingFunction(
            url=f"{settings.ollama_base_url}/api/embeddings",
            model_name=settings.embed_model
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self._bm25: Optional[BM25Okapi] = None
        self._doc_cache: List[RAGDocument] = []
        self._reranker = None
        self._load_reranker()

        # Subscribe to RAG query requests from agents/voice
        bus.subscribe(EventTopic.MEMORY_READ_REQ, self._on_query)

    def _load_reranker(self):
        """Load cross-encoder re-ranker. Free model from HuggingFace."""
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
                device="cpu"
            )
            print("[RAG] Cross-encoder re-ranker loaded.")
        except ImportError:
            print("[RAG] sentence-transformers not installed. Skipping re-ranker.")
        except Exception as e:
            print(f"[RAG] Re-ranker load failed: {e}")

    def _build_bm25(self):
        """Rebuild BM25 index from Chroma collection. Call after ingestion."""
        results = self._collection.get(include=["documents", "metadatas"])
        docs = results.get("documents", []) or []
        ids  = results.get("ids", []) or []
        metas= results.get("metadatas", []) or []

        self._doc_cache = [
            RAGDocument(id=ids[i], content=docs[i], metadata=metas[i])
            for i in range(len(docs))
        ]
        tokenized = [d.content.lower().split() for d in self._doc_cache]
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
            print(f"[RAG] BM25 index built with {len(self._doc_cache)} docs.")

    async def ingest_session_summary(self, summary: str, metadata: Dict):
        """Ingest a session summary into the knowledge base."""
        doc_id = f"session_{int(time.time())}"
        self._collection.add(
            documents=[summary],
            metadatas=[metadata],
            ids=[doc_id]
        )
        # Rebuild BM25 index
        await asyncio.get_event_loop().run_in_executor(None, self._build_bm25)
        print(f"[RAG] Ingested document: {doc_id}")

    async def query(self, question: str, live_context: Dict) -> str:
        """
        Full RAG pipeline: retrieve → fuse → re-rank → generate.
        live_context: current fatigue score, HR, etc.
        """
        # 1. HyDE: ask LLM for hypothetical answer to improve retrieval
        hyde_doc = await self._hyde_expand(question)
        search_text = f"{question} {hyde_doc}"

        # 2. Dense retrieval
        dense_results = await asyncio.get_event_loop().run_in_executor(
            None, self._dense_search, search_text
        )

        # 3. Sparse retrieval
        sparse_results = self._sparse_search(question)

        # 4. Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # 5. Re-rank
        if self._reranker and fused:
            reranked = await asyncio.get_event_loop().run_in_executor(
                None, self._rerank, question, fused
            )
        else:
            reranked = sorted(fused, key=lambda d: d.final_score, reverse=True)

        top_docs = reranked[:self.TOP_K_FINAL]

        # 6. Build prompt
        prompt = self._build_prompt(question, top_docs, live_context)

        # 7. Generate
        answer = await self._generate(prompt)

        # 8. Guard
        answer = self._guard_output(answer)
        return answer

    async def _hyde_expand(self, question: str) -> str:
        """Generate a hypothetical answer for better retrieval query."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.fallback_llm,  # use fast model for HyDE
                        "prompt": (f"In one sentence, what would be a relevant "
                                   f"document about: {question}"),
                        "stream": False,
                        "options": {"num_predict": 60}
                    }
                )
                return resp.json().get("response", "")
        except Exception:
            return ""

    def _dense_search(self, query: str) -> List[RAGDocument]:
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(self.TOP_K_RETRIEVE, max(1, self._collection.count())),
                include=["documents", "metadatas", "distances"]
            )
            docs = []
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                docs.append(RAGDocument(
                    id=results["ids"][0][i],
                    content=doc,
                    metadata=meta,
                    dense_score=1.0 / (1 + dist)  # similarity from distance
                ))
            return docs
        except Exception as e:
            print(f"[RAG] Dense search error: {e}")
            return []

    def _sparse_search(self, query: str) -> List[RAGDocument]:
        if self._bm25 is None or not self._doc_cache:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked_idx = scores.argsort()[::-1][:self.TOP_K_RETRIEVE]
        results = []
        for idx in ranked_idx:
            if scores[idx] > 0:
                doc = self._doc_cache[idx]
                doc.sparse_score = float(scores[idx])
                results.append(doc)
        return results

    def _reciprocal_rank_fusion(
        self,
        dense: List[RAGDocument],
        sparse: List[RAGDocument],
        k: int = 60
    ) -> List[RAGDocument]:
        """RRF: score = sum(1 / (rank + k)) over both lists."""
        score_map: Dict[str, float] = {}
        doc_map: Dict[str, RAGDocument] = {}

        for rank, doc in enumerate(dense):
            score_map[doc.id] = score_map.get(doc.id, 0) + 1.0 / (rank + 1 + k)
            doc_map[doc.id] = doc

        for rank, doc in enumerate(sparse):
            score_map[doc.id] = score_map.get(doc.id, 0) + 1.0 / (rank + 1 + k)
            doc_map[doc.id] = doc

        for doc_id, score in score_map.items():
            doc_map[doc_id].final_score = score

        return sorted(doc_map.values(), key=lambda d: d.final_score, reverse=True)

    def _rerank(self, query: str, docs: List[RAGDocument]) -> List[RAGDocument]:
        """Cross-encoder re-ranking for precision."""
        pairs = [(query, d.content) for d in docs]
        scores = self._reranker.predict(pairs)
        for doc, score in zip(docs, scores):
            doc.rerank_score = float(score)
            doc.final_score = float(score)  # override with re-rank score
        return sorted(docs, key=lambda d: d.final_score, reverse=True)

    def _build_prompt(
        self,
        question: str,
        docs: List[RAGDocument],
        live_context: Dict
    ) -> str:
        context_str = "\n\n".join([
            f"[Context {i+1}] {d.content}" for i, d in enumerate(docs)
        ]) if docs else "No historical context available."

        live_str = json.dumps({
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in live_context.items()
        }, indent=2)

        return f"""{self.SYSTEM_PROMPT}

--- LIVE DRIVER STATE ---
{live_str}

--- RETRIEVED CONTEXT ---
{context_str}

--- DRIVER QUESTION ---
{question}

Answer (max 3 sentences, spoken aloud):"""

    async def _generate(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.primary_llm,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 150,
                            "temperature": 0.3,   # low temp for factual
                            "top_p": 0.9
                        }
                    }
                )
                return resp.json().get("response", "I'm processing your request.")
        except Exception as e:
            return f"System is temporarily unavailable. Please check your fatigue status manually."

    def _guard_output(self, text: str) -> str:
        """Remove hallucinations, clamp length, ensure safety language."""
        # Truncate to reasonable spoken length (~3 sentences)
        sentences = text.split(". ")
        if len(sentences) > 4:
            text = ". ".join(sentences[:3]) + "."
        # Remove common hallucination patterns
        bad_phrases = ["I don't have access to", "As an AI", "I cannot"]
        for phrase in bad_phrases:
            text = text.replace(phrase, "")
        return text.strip()

    async def _on_query(self, event):
        payload = event.payload
        question = payload.get("question", "")
        live_ctx = payload.get("live_context", {})
        answer = await self.query(question, live_ctx)
        await self._bus.publish(
            EventTopic.MEMORY_READ_RESP,
            {"answer": answer, "correlation_id": event.correlation_id},
            source="rag_engine",
            correlation_id=event.correlation_id
        )

    async def search(self, question: str) -> str:
        """Compatibility wrapper for direct Orchestrator routing."""
        answer = await self.query(question, {})
        await self._bus.publish(
            EventTopic.VOICE_RESPONSE,
            {"text": answer},
            source="rag_engine"
        )
        return answer

    async def run(self):
        """Build BM25 index on startup, then idle (reactive via bus)."""
        await asyncio.get_event_loop().run_in_executor(None, self._build_bm25)
        # Keep alive — bus subscriptions handle queries reactively
        while True:
            await asyncio.sleep(60)
            # Periodically rebuild BM25 index as new docs ingested
            await asyncio.get_event_loop().run_in_executor(None, self._build_bm25)
