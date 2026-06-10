"""Pipelines. IngestPipeline (offline): Document → chunk → embed → store + record metadata.
QueryPipeline (online): question → graph → {answer, citations}. Both pure."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from genacademy_rag.core.graph import build_graph
from genacademy_rag.core.types import Chunk, Citation, Document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedIngest:
    doc: Document
    chunks: list[Chunk]
    embeddings: list[list[float]]


class IngestPipeline:
    def __init__(self, *, chunker, provider, store, datastore):
        self._chunker = chunker
        self._provider = provider
        self._store = store
        self._datastore = datastore

    def prepare(self, docs: list[Document]) -> list[PreparedIngest]:
        prepared = []
        for doc in docs:
            chunks: list[Chunk] = self._chunker.chunk(doc)
            if not chunks:
                continue
            embeddings = self._provider.embed([c.text for c in chunks])
            prepared.append(PreparedIngest(doc=doc, chunks=chunks, embeddings=embeddings))
        return prepared

    def commit(self, prepared: list[PreparedIngest]) -> int:
        total = 0
        for item in prepared:
            doc = item.doc
            # Vector store first: if a (possibly remote) upsert fails, no SQLite row exists,
            # so the admin list never shows a document that is not actually searchable.
            self._store.upsert(item.chunks, item.embeddings)
            try:
                self._datastore.add_document(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source_type=doc.source_type,
                    repo=doc.repo,
                    file_path=doc.file_path,
                    commit_hash=doc.commit_hash,
                    filename=doc.filename,
                    uploaded_by=doc.uploaded_by,
                    stored_path=doc.stored_path,
                    n_chunks=len(item.chunks),
                )
                self._datastore.add_chunks_meta(item.chunks)
            except Exception:
                # Compensate: vectors without a ledger row are invisible to the admin
                # list yet would be kept by the reindex filter (which deliberately
                # retains ledger-less eval-seed docs) — roll them back, then surface
                # the original ledger error.
                try:
                    self._store.delete_doc(doc.doc_id)
                except Exception:
                    logger.exception(
                        "vector rollback for %s failed after ledger write error; "
                        "orphaned vectors may resurface on reindex", doc.doc_id,
                    )
                raise
            total += len(item.chunks)
        return total

    def ingest(self, docs: list[Document]) -> int:
        return self.commit(self.prepare(docs))


@dataclass(frozen=True)
class QueryResult:
    answer: str
    citations: list[Citation]
    refused: bool
    confidence: int
    used_fallback: bool = False


class QueryPipeline:
    def __init__(self, *, retriever, provider, cosine_threshold: float = 0.2):
        self._graph = build_graph(
            retriever=retriever, provider=provider, cosine_threshold=cosine_threshold
        )

    def answer(self, question: str) -> QueryResult:
        out = self._graph.invoke({"question": question})
        # Index required keys directly (never `.get(default)`): the graph always sets answer,
        # citations, refused, confidence, used_fallback, so a missing key is a wiring bug we want
        # to surface as a KeyError — not paper over with an uncited / zero-confidence answer.
        return QueryResult(
            answer=out["answer"],
            citations=out["citations"],
            refused=out["refused"],
            confidence=out["confidence"],
            used_fallback=out["used_fallback"],
        )
