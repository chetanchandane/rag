"""
search.py — Vector search against Qdrant Cloud

Stage 1: Dense semantic search only (cosine similarity on embeddings).
Stage 2 (planned): Hybrid search — dense + BM25 sparse vectors with
                   Reciprocal Rank Fusion (RRF).

Every public method is decorated with @traceable so LangSmith
automatically captures query, retrieved chunks, scores, and latency.
"""

import os

from langsmith import traceable
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from src.config import config


class Searcher:
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.qdrant = AsyncQdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )

    @traceable(name="embed_query")
    async def embed_query(self, query: str) -> list[float]:
        """Embed the user query using the same model used during ingestion."""
        response = await self.openai.embeddings.create(
            model=config.embedding_model,
            input=query,
        )
        return response.data[0].embedding

    @traceable(name="vector_search")
    async def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Embed the query and retrieve the top_k most similar chunks from Qdrant.

        Returns a list of dicts:
            [{"text", "source", "page", "score"}, ...]

        Chunks with a cosine score below config.min_relevance_score are filtered out.
        """
        k = top_k or config.default_top_k
        embedding = await self.embed_query(query)

        results = await self.qdrant.search(
            collection_name=config.collection_name,
            query_vector=embedding,
            limit=k,
            with_payload=True,
        )

        contexts = [
            {
                "text":   r.payload.get("text", ""),
                "source": r.payload.get("source", "unknown"),
                "page":   r.payload.get("page", 0),
                "score":  round(r.score, 4),
            }
            for r in results
            if r.score >= config.min_relevance_score
        ]

        return contexts
