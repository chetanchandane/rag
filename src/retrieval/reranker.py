"""
reranker.py — Cross-encoder re-ranking

Stage 1: Passthrough — returns chunks in the same order received.
Stage 2 (planned): Cohere rerank-english-v3.0 cross-encoder.
                   Replace the body of `rerank()` with the Cohere API call.

Keeping this module as a no-op in Stage 1 means the API and search module
never need to change — just swap in the real implementation here.
"""

from langsmith import traceable


class Reranker:

    @traceable(name="rerank")
    async def rerank(self, query: str, chunks: list[dict], top_n: int | None = None) -> list[dict]:
        """
        Stage 1: passthrough — return chunks unchanged.

        Stage 2 implementation (drop-in replacement):
        -----------------------------------------------
        import cohere
        co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
        docs = [c["text"] for c in chunks]
        results = co.rerank(query=query, documents=docs,
                            model="rerank-english-v3.0", top_n=top_n or 5)
        reranked = []
        for r in results.results:
            chunk = chunks[r.index]
            chunk["rerank_score"] = r.relevance_score
            reranked.append(chunk)
        return reranked
        """
        # Stage 1: identity function
        if top_n:
            return chunks[:top_n]
        return chunks
