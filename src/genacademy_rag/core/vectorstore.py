"""VectorStore seam. ChromaStore = raw chromadb PersistentClient holding precomputed embeddings
(we embed via ModelProvider, Chroma just stores+searches). Phase-2 swap target: PineconeStore."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from genacademy_rag.core.types import Chunk, Citation


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def query(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]: ...
    def get_chunk(self, chunk_id: str) -> Chunk: ...
    def get_all_chunks(self) -> list[Chunk]: ...


class ChromaStore:
    def __init__(self, persist_dir, collection: str = "genacademy"):
        import chromadb
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        # cosine space; we pass our own normalized embeddings.
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(map(float, e)) for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[{**c.citation.to_metadata(), "ordinal": c.ordinal} for c in chunks],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        # Return (chunk_id, cosine_similarity). Chroma cosine space gives DISTANCE; sim = 1 - dist.
        # The similarity is the confidence signal the grader's cosine fallback uses (see Task 7);
        # RRF (Task 6) handles ranking. Returning raw IDs only would leave the fallback no signal.
        res = self._col.query(query_embeddings=[list(map(float, query_embedding))],
                              n_results=top_k, include=["distances"])
        if not res["ids"] or not res["ids"][0]:
            return []
        ids, dists = res["ids"][0], res["distances"][0]
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, dists, strict=True)]

    def get_chunk(self, chunk_id: str) -> Chunk:
        res = self._col.get(ids=[chunk_id], include=["documents", "metadatas"])
        text = res["documents"][0]
        meta = dict(res["metadatas"][0])
        ordinal = int(meta.pop("ordinal", 0))
        return Chunk(chunk_id=chunk_id, doc_id=meta["doc_id"], ordinal=ordinal,
                     text=text, citation=Citation.from_metadata(meta))

    def get_all_chunks(self) -> list[Chunk]:
        """Public accessor so callers never reach into `_col` (keeps the Pinecone swap clean)."""
        res = self._col.get(include=["documents", "metadatas"])
        out: list[Chunk] = []
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True):
            m = dict(meta)
            ordinal = int(m.pop("ordinal", 0))
            out.append(Chunk(chunk_id=cid, doc_id=m["doc_id"], ordinal=ordinal,
                             text=doc, citation=Citation.from_metadata(m)))
        return out
