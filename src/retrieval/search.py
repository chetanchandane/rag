"""
search.py — Hybrid vector search against Qdrant Cloud

Stage 2: HyDE query expansion + hybrid BM25/dense search with Reciprocal Rank Fusion.

Pipeline per query:
  1. HyDE: Claude generates a hypothetical regulatory passage from the question.
  2. Dense embedding: OpenAI embeds the hypothetical doc (richer signal than raw query).
  3. Sparse BM25: fastembed tokenizes the original query into a sparse vector.
  4. Parallel search: fire both against Qdrant (top-20 each).
  5. RRF merge: combine ranked lists with k=60, dedup, return top-k.

Every public method is decorated with @traceable for LangSmith observability.
"""

import asyncio
import os
from functools import lru_cache

from anthropic import AsyncAnthropic
from langsmith import traceable
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector

from src.config import config


@lru_cache(maxsize=1)
def _get_sparse_model():
    """Lazy-load BM25 model once per process (downloads ~5 MB on first call)."""
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name="Qdrant/bm25")


class Searcher:

    def __init__(self):
        self.openai    = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.anthropic = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.qdrant    = AsyncQdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )

    # ── Step 1: HyDE ──────────────────────────────────────────────────────────

    @traceable(name="hyde_expand_query")
    async def expand_query(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings): generate a dense regulatory
        passage that would answer the user's question. Embedding this text
        gives better semantic alignment than embedding the short query itself.
        """
        response = await self.anthropic.messages.create(
            model=config.generation_model,
            max_tokens=config.hyde_max_tokens,
            system=(
                "You are a clinical regulatory expert. Write a single dense paragraph "
                "from an ICH or FDA guidance document that directly answers the question. "
                "Write only the passage text — no preamble, no citations, no headers."
            ),
            messages=[{"role": "user", "content": query}],
        )
        return response.content[0].text

    # ── Dense embedding ───────────────────────────────────────────────────────

    @traceable(name="embed_query")
    async def embed_query(self, text: str) -> list[float]:
        """Embed text using the OpenAI dense model (same model used at index time)."""
        response = await self.openai.embeddings.create(
            model=config.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    # ── Sparse BM25 embedding ─────────────────────────────────────────────────

    def sparse_embed_query(self, text: str) -> SparseVector:
        """Generate a BM25 sparse vector. Synchronous — CPU-only, no network."""
        model = _get_sparse_model()
        result = next(model.embed([text]))
        return SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )

    # ── RRF fusion ────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_merge(
        dense_results: list,
        sparse_results: list,
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion:
            score(d) = Σ  1 / (k + rank_in_list)
        Deduplicates by point ID, sorts descending by RRF score.
        """
        scores:   dict[str, float] = {}
        payloads: dict[str, dict]  = {}

        for rank, hit in enumerate(dense_results, start=1):
            pid = str(hit.id)
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
            payloads[pid] = hit.payload

        for rank, hit in enumerate(sparse_results, start=1):
            pid = str(hit.id)
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
            if pid not in payloads:
                payloads[pid] = hit.payload

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "text":   payloads[pid].get("text", ""),
                "source": payloads[pid].get("source", "unknown"),
                "page":   payloads[pid].get("page", 0),
                "score":  round(rrf_score, 6),
            }
            for pid, rrf_score in ranked
        ]

    # ── Main search ───────────────────────────────────────────────────────────

    @traceable(name="hybrid_search")
    async def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Full Phase 2 retrieval:
          1. HyDE expand the query.
          2. Embed the hypothetical doc (dense) — network call, async.
          3. BM25-encode the original query (sparse) — CPU, sync in executor.
          4. Fire both Qdrant searches concurrently (top-20 each).
          5. Merge with RRF and return top_k results.
        """
        k            = top_k or config.default_top_k
        candidate_k  = config.retrieval_top_k

        # 1. HyDE
        hypothetical_doc = await self.expand_query(query)

        # 2. Dense embed (async) + sparse embed (sync in thread) — parallel
        loop = asyncio.get_running_loop()
        dense_embedding, sparse_vec = await asyncio.gather(
            self.embed_query(hypothetical_doc),
            loop.run_in_executor(None, self.sparse_embed_query, query),
        )

        # 3. Both Qdrant searches in parallel
        dense_results, sparse_results = await asyncio.gather(
            self.qdrant.search(
                collection_name=config.collection_name,
                query_vector=NamedVector(
                    name=config.dense_vector_name,
                    vector=dense_embedding,
                ),
                limit=candidate_k,
                with_payload=True,
            ),
            self.qdrant.search(
                collection_name=config.collection_name,
                query_vector=NamedSparseVector(
                    name=config.sparse_vector_name,
                    vector=sparse_vec,
                ),
                limit=candidate_k,
                with_payload=True,
            ),
        )

        # 4. RRF merge → top_k
        merged = self._rrf_merge(dense_results, sparse_results, k=config.rrf_k)
        return merged[:k]
