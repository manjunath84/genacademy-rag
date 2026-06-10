"""VectorStore seam. ChromaStore = raw chromadb PersistentClient; PineconeStore = serverless
Pinecone index (Phase 2 preset, GENACADEMY_VECTORSTORE=pinecone). We embed via ModelProvider;
the store only holds+searches precomputed vectors. Score contract: query() returns cosine
SIMILARITY — Chroma reports distance (converted via 1-d), Pinecone reports similarity directly."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from genacademy_rag.core.types import Chunk, Citation


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def query(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]: ...
    def get_chunk(self, chunk_id: str) -> Chunk: ...
    def get_all_chunks(self) -> list[Chunk]: ...
    def delete_doc(self, doc_id: str) -> None: ...


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

    def delete_doc(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

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


class VectorStoreSetupError(RuntimeError):
    """Raised when a vector store preset is selected but cannot be constructed."""


# all-MiniLM-L6-v2 output dimension; Pinecone index dim must match (specs/tech-stack.md).
EMBED_DIM = 384

# Pinecone metadata comes back as JSON numbers (floats); these Citation/Chunk fields are ints.
_INT_META_FIELDS = ("ordinal", "line_start", "line_end", "char_start", "char_end")

_PINECONE_BATCH = 100


class PineconeStore:
    """Serverless Pinecone index behind the VectorStore Protocol. Chunk text and citation live in
    vector metadata (Pinecone rejects None values; Citation.to_metadata already strips them).
    `collection` maps to a Pinecone namespace. delete_doc lists ids by `{doc_id}::` prefix because
    serverless indexes do not support metadata-filtered deletes."""

    def __init__(self, *, api_key: str, index_name: str, namespace: str = "",
                 dimension: int = EMBED_DIM, cloud: str = "aws", region: str = "us-east-1",
                 client=None):
        if client is None:
            from pinecone import Pinecone
            client = Pinecone(api_key=api_key)
        if not client.has_index(index_name):
            from pinecone import ServerlessSpec
            client.create_index(name=index_name, dimension=dimension, metric="cosine",
                                spec=ServerlessSpec(cloud=cloud, region=region))
        self._index = client.Index(index_name)
        self._namespace = namespace

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        vectors = [
            {
                "id": c.chunk_id,
                "values": [float(x) for x in emb],
                "metadata": {**c.citation.to_metadata(), "ordinal": c.ordinal, "text": c.text},
            }
            for c, emb in zip(chunks, embeddings, strict=True)
        ]
        for i in range(0, len(vectors), _PINECONE_BATCH):
            self._index.upsert(vectors=vectors[i:i + _PINECONE_BATCH], namespace=self._namespace)

    def query(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        # Pinecone cosine score IS similarity (no 1-dist conversion, unlike Chroma above).
        res = self._index.query(vector=[float(x) for x in query_embedding], top_k=top_k,
                                namespace=self._namespace)
        return [(m.id, float(m.score)) for m in res.matches]

    def _chunk_from_metadata(self, chunk_id: str, metadata: dict) -> Chunk:
        m = dict(metadata)
        text = m.pop("text")
        for field in _INT_META_FIELDS:
            if field in m:
                m[field] = int(m[field])
        ordinal = int(m.pop("ordinal", 0))
        return Chunk(chunk_id=chunk_id, doc_id=m["doc_id"], ordinal=ordinal,
                     text=text, citation=Citation.from_metadata(m))

    def get_chunk(self, chunk_id: str) -> Chunk:
        res = self._index.fetch(ids=[chunk_id], namespace=self._namespace)
        return self._chunk_from_metadata(chunk_id, res.vectors[chunk_id].metadata)

    def get_all_chunks(self) -> list[Chunk]:
        ids = [item.id for page in self._index.list(namespace=self._namespace)
               for item in page.vectors]
        chunks: list[Chunk] = []
        for i in range(0, len(ids), _PINECONE_BATCH):
            res = self._index.fetch(ids=ids[i:i + _PINECONE_BATCH], namespace=self._namespace)
            chunks.extend(self._chunk_from_metadata(cid, v.metadata)
                          for cid, v in res.vectors.items())
        # Pinecone list order is arbitrary; BM25 index build needs a deterministic corpus order.
        return sorted(chunks, key=lambda c: (c.doc_id, c.ordinal))

    def delete_doc(self, doc_id: str) -> None:
        ids = [item.id for page in self._index.list(prefix=f"{doc_id}::",
                                                    namespace=self._namespace)
               for item in page.vectors]
        if ids:
            self._index.delete(ids=ids, namespace=self._namespace)


def build_vectorstore(settings, *, collection: str) -> VectorStore:
    """The 'one config line' swap: GENACADEMY_VECTORSTORE=chroma|pinecone."""
    if settings.vectorstore == "chroma":
        return ChromaStore(persist_dir=settings.chroma_dir, collection=collection)
    if settings.vectorstore == "pinecone":
        if not settings.pinecone_api_key:
            raise VectorStoreSetupError(
                "GENACADEMY_VECTORSTORE=pinecone requires PINECONE_API_KEY in the environment"
            )
        return PineconeStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index,
            namespace=collection,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
    raise ValueError(
        f"unknown vectorstore {settings.vectorstore!r}; expected 'chroma' or 'pinecone'"
    )
