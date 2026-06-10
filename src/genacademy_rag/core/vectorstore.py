"""VectorStore seam. ChromaStore = raw chromadb PersistentClient; PineconeStore = serverless
Pinecone index (Phase 2 preset, GENACADEMY_VECTORSTORE=pinecone). We embed via ModelProvider;
the store only holds+searches precomputed vectors. Score contract: query() returns cosine
SIMILARITY — Chroma reports distance (converted via 1-d), Pinecone reports similarity directly."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from genacademy_rag.core.types import Chunk, Citation

logger = logging.getLogger(__name__)


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
        # The similarity is the confidence signal the grader's cosine fallback uses; RRF fusion in
        # HybridRetriever handles ranking. Returning raw IDs would leave the fallback no signal.
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
        """Public accessor so callers never reach into `_col` (keeps stores interchangeable)."""
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
_PINECONE_DELETE_BATCH = 1000   # API cap: at most 1000 ids per delete request


class PineconeStore:
    """Serverless Pinecone index behind the VectorStore Protocol. Chunk text and citation live in
    vector metadata (Pinecone rejects None values; Citation.to_metadata already strips them; the
    ~40KB-per-vector metadata cap comfortably fits the project's default <=1500-char chunks —
    env-tunable, so stay well under the cap if raising chunk sizes).
    `collection` maps to a Pinecone namespace. delete_doc lists ids by `{doc_id}::` prefix because
    serverless indexes do not support metadata-filtered deletes.

    Consistency caveat: serverless reads (query/fetch/list) lag writes by seconds. Callers must
    not assume read-your-writes — the web app derives its in-memory corpus from the retriever's
    snapshot plus local deltas on every mutation, and the admin reindex is the one deliberate
    remote re-read (recovery path for missing chunks), filtered against the datastore's deletion
    ledger so lagged deletes cannot resurrect."""

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

    def _chunk_from_metadata(self, chunk_id: str, metadata: dict | None) -> Chunk:
        if not metadata or "text" not in metadata or "doc_id" not in metadata:
            raise ValueError(
                f"vector {chunk_id!r} in Pinecone namespace {self._namespace!r} lacks required "
                "metadata (text/doc_id) — was it written by something other than this app?"
            )
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
        if chunk_id not in res.vectors:
            raise KeyError(
                f"chunk not found in Pinecone namespace {self._namespace!r}: {chunk_id}"
            )
        return self._chunk_from_metadata(chunk_id, res.vectors[chunk_id].metadata)

    def get_all_chunks(self) -> list[Chunk]:
        ids = [item.id for page in self._index.list(namespace=self._namespace)
               for item in page.vectors]
        chunks: list[Chunk] = []
        for i in range(0, len(ids), _PINECONE_BATCH):
            res = self._index.fetch(ids=ids[i:i + _PINECONE_BATCH], namespace=self._namespace)
            chunks.extend(self._chunk_from_metadata(cid, v.metadata)
                          for cid, v in res.vectors.items())
        if len(chunks) != len(ids):
            # fetch silently omits missing ids (eventually-consistent read or a racing
            # delete) — surface it, or the corpus shrinks with no signal at all.
            logger.warning(
                "Pinecone get_all_chunks: listed %d ids but fetched %d in namespace %r — "
                "read may be partial", len(ids), len(chunks), self._namespace,
            )
        # Pinecone list order is arbitrary; BM25 index build needs a deterministic corpus order.
        return sorted(chunks, key=lambda c: (c.doc_id, c.ordinal))

    def delete_doc(self, doc_id: str) -> None:
        ids = [item.id for page in self._index.list(prefix=f"{doc_id}::",
                                                    namespace=self._namespace)
               for item in page.vectors]
        if not ids:
            # A lagging list() means the doc's vectors are orphaned remotely. They can
            # never be served (the retriever drops unknown ids) but the operator should
            # know they exist; the reindex deletion-ledger filter keeps them out forever.
            logger.warning(
                "Pinecone delete_doc(%r): listing returned no ids in namespace %r — "
                "vectors may be orphaned by a lagging read", doc_id, self._namespace,
            )
        for i in range(0, len(ids), _PINECONE_DELETE_BATCH):
            self._index.delete(ids=ids[i:i + _PINECONE_DELETE_BATCH], namespace=self._namespace)


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
            # Pinned to the all-MiniLM-L6-v2 dimension. A different GENACADEMY_EMBED_MODEL
            # needs an index whose dimension matches, or every upsert/query fails remotely.
            dimension=EMBED_DIM,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
    raise ValueError(
        f"unknown vectorstore {settings.vectorstore!r}; expected 'chroma' or 'pinecone'"
    )
