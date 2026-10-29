"""Approved local knowledge. Optional dense retrieval never downloads at startup."""

import hashlib
import math
import re
from collections import Counter
from pathlib import Path


def tokens(text):
    return re.findall(r"[a-z]{3,}", text.lower())


class KnowledgeBase:
    def __init__(self, path, embedding_path=None, index_path=None):
        self.path = Path(path)
        self.docs = {}
        self.checksums = {}
        self.dense = None
        self.embedder = None
        self.embedding_path = embedding_path
        self.index_path = index_path
        self.status = "lexical"

    def ingest(self):
        changed = 0
        present = set()
        for path in sorted(self.path.glob("*")):
            if path.suffix.lower() not in (".txt", ".md") or path.is_symlink():
                continue
            content = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(content.encode()).hexdigest()
            present.add(path.name)
            if self.checksums.get(path.name) == checksum:
                continue
            changed += 1
            self.checksums[path.name] = checksum
            self.docs = {k: v for k, v in self.docs.items() if v["file"] != path.name}
            meta = dict(title=path.stem, source="local-approved", revision="unknown")
            if content.startswith("---\n"):
                _, front, content = content.split("---", 2)
                for line in front.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        if k.strip() in meta:
                            meta[k.strip()] = v.strip()
            # Bounded paragraph chunks preserve readable citation excerpts.
            paragraphs = re.split(r"\n\s*\n", content.strip())
            for i, paragraph in enumerate(paragraphs):
                for offset in range(0, len(paragraph), 1800):
                    chunk = paragraph[offset : offset + 1800].strip()
                    if not chunk:
                        continue
                    identity = hashlib.sha256(f"{path.name}:{i}:{offset}".encode()).hexdigest()[:20]
                    self.docs[identity] = dict(
                        id=identity,
                        content=chunk,
                        file=path.name,
                        checksum=checksum,
                        section=f"Excerpt {i + 1}",
                        **meta,
                    )
        removed = [name for name in self.checksums if name not in present]
        for name in removed:
            del self.checksums[name]
            self.docs = {k: v for k, v in self.docs.items() if v["file"] != name}
            changed += 1
        if changed and self.dense is not None:
            self._sync_dense()
        return changed

    def enable_dense(self):
        if not self.embedding_path or not Path(self.embedding_path).is_dir():
            self.status = "lexical: local embedding model not installed"
            return
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(str(self.embedding_path), local_files_only=True)
            self.dense = chromadb.PersistentClient(
                path=str(self.index_path)
            ).get_or_create_collection("sentinel-approved-v1", metadata={"hnsw:space": "cosine"})
            self._sync_dense()
            self.status = "hybrid"
        except Exception:
            self.dense = self.embedder = None
            self.status = "lexical: dense dependencies unavailable"

    def _sync_dense(self):
        existing = self.dense.get(include=["metadatas"])
        old = dict(zip(existing["ids"], existing["metadatas"]))
        deleted = list(set(old) - set(self.docs))
        if deleted:
            self.dense.delete(ids=deleted)
        changed = [
            d for d in self.docs.values() if old.get(d["id"], {}).get("checksum") != d["checksum"]
        ]
        if changed:
            embeddings = self.embedder.encode([d["content"] for d in changed]).tolist()
            self.dense.upsert(
                ids=[d["id"] for d in changed],
                embeddings=embeddings,
                documents=[d["content"] for d in changed],
                metadatas=[{"checksum": d["checksum"]} for d in changed],
            )

    def search(self, query):
        query_tokens = tokens(query)
        scored = []
        corpus = [tokens(d["content"] + " " + d["title"]) for d in self.docs.values()]
        avg_len = sum(map(len, corpus)) / max(1, len(corpus))
        for doc, words in zip(self.docs.values(), corpus):
            freq = Counter(words)
            score = 0.0
            for token in query_tokens:
                n = sum(token in terms for terms in corpus)
                idf = math.log(1 + (len(corpus) - n + 0.5) / (n + 0.5))
                score += (
                    idf
                    * freq[token]
                    * 2.5
                    / (freq[token] + 1.5 * (0.25 + 0.75 * len(words) / max(1, avg_len)))
                )
            if score > 0:
                scored.append((score, doc["id"]))
        scored.sort(reverse=True)
        ranks = {identity: 1 / (61 + i) for i, (_, identity) in enumerate(scored[:20])}
        if self.dense and self.docs:
            result = self.dense.query(
                query_embeddings=self.embedder.encode([query]).tolist(),
                n_results=min(20, len(self.docs)),
            )
            for i, identity in enumerate(result["ids"][0]):
                # Require lexical overlap as an abstention guard for this small corpus.
                if identity in ranks:
                    ranks[identity] += 1 / (61 + i)
        return [self.docs[k] for k in sorted(ranks, key=ranks.get, reverse=True)[:5]]

    def document(self, identity):
        return self.docs[identity]
