# Diagrams

Architecture diagrams:

- [System overview](01-system-overview.svg)
- [Query flow](02-query-flow.svg)
- [Ingest and two-tier corpus](03-ingest-two-tier-corpus.svg)
- [Hugging Face Space deployment](04-deployment-hf-space.svg)
- [Editable Draw.io source](architecture.drawio)

Concept diagrams:

- [GenAcademy Compass operating model](05-genacademy-compass-operating-model.svg) - a non-architecture companion visual
  for the project write-up or demo narrative. It reflects the live Space posture: two-tier corpus
  policy, cited answer/refusal path, hybrid retrieval with the cross-encoder rerank live in the
  Space (pool 20, model baked into the image, env kill switch), Chroma/Pinecone and local/Nebius
  preset seams, section-aware chunking kept off after regressing on eval, product/admin surface,
  eval results, deployment caveats, and scale guardrails.
