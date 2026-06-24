"""
Central configuration for the RAG system.

All tuneable parameters live here — chunk sizes, model names, thresholds.
To change behaviour, edit this file rather than hunting through source files.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536           # fixed for text-embedding-3-small

    # ── Generation ─────────────────────────────────────────────────────────────
    generation_model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = 512               # tokens per child chunk
    chunk_overlap: int = 64            # overlap between consecutive chunks

    # ── Retrieval ──────────────────────────────────────────────────────────────
    default_top_k: int = 5             # chunks returned to the LLM after reranking
    retrieval_top_k: int = 20          # candidate pool fetched from each search pass
    min_relevance_score: float = 0.35  # kept for dense-only fallback mode

    # ── Phase 2: Hybrid search & reranking ────────────────────────────────────
    dense_vector_name: str = "dense"   # named vector field in Qdrant
    sparse_vector_name: str = "sparse" # sparse BM25 field in Qdrant
    rrf_k: int = 60                    # RRF constant (higher = smoother fusion)
    hyde_max_tokens: int = 256         # max tokens for HyDE hypothetical doc
    cohere_rerank_model: str = "rerank-english-v3.0"

    # ── Vector DB (Qdrant Cloud) ───────────────────────────────────────────────
    qdrant_url: str = field(default_factory=lambda: os.environ["QDRANT_URL"])
    qdrant_api_key: str = field(default_factory=lambda: os.environ["QDRANT_API_KEY"])
    collection_name: str = field(
        default_factory=lambda: os.environ.get("QDRANT_COLLECTION", "clinical_docs")
    )

    # ── API ────────────────────────────────────────────────────────────────────
    api_title: str = "Clinical RAG API"
    api_version: str = "1.0.0"


# Single shared instance — import this everywhere
config = Config()
