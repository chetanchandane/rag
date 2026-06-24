"""
reranker.py — Cohere cross-encoder re-ranking (Phase 2)

Uses Cohere's rerank-english-v3.0 model to score each (query, chunk) pair
directly, rather than relying on embedding similarity. Receives the top-20
RRF-fused candidates from hybrid search and returns the top-5 by relevance.

The rerank call is wrapped in asyncio.to_thread because the Cohere sync
client is simpler to maintain across SDK versions than the async client.
"""

import asyncio
import os

import cohere
from langsmith import traceable

from src.config import config


class Reranker:

    def __init__(self):
        self._co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

    @traceable(name="rerank")
    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int | None = None,
    ) -> list[dict]:
        """
        Rerank chunks with Cohere cross-encoder.

        Args:
            query:  the original user question.
            chunks: RRF-fused candidate list (up to retrieval_top_k=20 items).
            top_n:  how many to return (defaults to config.default_top_k=5).

        Returns:
            top_n chunks sorted by Cohere relevance score (descending),
            each with an added "rerank_score" key.
        """
        n = top_n or config.default_top_k
        if not chunks:
            return []

        docs = [c["text"] for c in chunks]

        def _sync_rerank():
            return self._co.rerank(
                query=query,
                documents=docs,
                model=config.cohere_rerank_model,
                top_n=n,
            )

        response = await asyncio.to_thread(_sync_rerank)

        reranked: list[dict] = []
        for hit in response.results:
            chunk = dict(chunks[hit.index])
            chunk["rerank_score"] = round(hit.relevance_score, 6)
            reranked.append(chunk)

        # Results from Cohere are already sorted descending by relevance_score
        return reranked
