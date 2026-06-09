"""Pipelines. IngestPipeline (offline): Document → chunk → embed → store + record metadata.
QueryPipeline (online, Task 11): question → graph → {answer, citations}. Both pure."""
from __future__ import annotations

from genacademy_rag.core.types import Chunk, Document


class IngestPipeline:
    def __init__(self, *, chunker, provider, store, datastore):
        self._chunker = chunker
        self._provider = provider
        self._store = store
        self._datastore = datastore

    def ingest(self, docs: list[Document]) -> int:
        total = 0
        for doc in docs:
            chunks: list[Chunk] = self._chunker.chunk(doc)
            if not chunks:
                continue
            embeddings = self._provider.embed([c.text for c in chunks])
            self._store.upsert(chunks, embeddings)
            self._datastore.add_document(
                doc_id=doc.doc_id,
                title=doc.title,
                source_type=doc.source_type,
                repo=doc.repo,
                file_path=doc.file_path,
                commit_hash=doc.commit_hash,
                filename=doc.filename,
                n_chunks=len(chunks),
            )
            self._datastore.add_chunks_meta(chunks)
            total += len(chunks)
        return total
