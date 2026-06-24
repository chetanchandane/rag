"""
indexer.py — Stage 3 of ingestion

Responsibility: embed chunks and upsert them into Qdrant Cloud with both
dense (OpenAI) and sparse (BM25 via fastembed) vectors for hybrid search.

Usage:
    python -m src.ingestion.indexer --file data/raw/ich_e6.pdf
    python -m src.ingestion.indexer --dir  data/raw/

Phase 2 changes:
  - Collection now stores named dense + sparse vectors.
  - If an old single-vector collection is detected it is deleted and recreated
    automatically (re-indexing required).
  - Sparse BM25 embeddings are generated locally via fastembed (no API call).
"""

import asyncio
import argparse
import os
import uuid
from functools import lru_cache
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config import config
from src.ingestion.loader import load_pdf_bytes
from src.ingestion.splitter import chunk_pages

# Tier-1 OpenAI accounts: 40k TPM limit for text-embedding-3-small.
EMBED_BATCH_SIZE  = 20    # chunks per API call
EMBED_BATCH_DELAY = 1.5   # seconds between batches


# ── Sparse model (loaded once per process) ────────────────────────────────────

@lru_cache(maxsize=1)
def _get_sparse_model():
    """Lazy-load BM25 fastembed model once per process."""
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name="Qdrant/bm25")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _qdrant() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)

def _openai() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


async def _already_indexed(qdrant: AsyncQdrantClient, filename: str) -> bool:
    """Return True if any chunks from this file already exist in Qdrant."""
    try:
        result = await qdrant.scroll(
            collection_name=config.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=filename))]
            ),
            limit=1,
        )
        return len(result[0]) > 0
    except Exception:
        return False


async def _ensure_collection(qdrant: AsyncQdrantClient) -> None:
    """
    Create the Qdrant collection with dense + sparse vector support.
    If an old single-vector (unnamed) collection exists it is dropped and
    recreated — re-indexing will be required.
    """
    needs_create = False
    try:
        info  = await qdrant.get_collection(config.collection_name)
        vconf = info.config.params.vectors
        # Old schema: VectorParams object (not a dict of named vectors)
        if not isinstance(vconf, dict):
            print(
                f"  ⚠️  Old single-vector schema detected on '{config.collection_name}'.\n"
                f"      Recreating with dense + sparse support (re-index required)."
            )
            await qdrant.delete_collection(config.collection_name)
            needs_create = True
    except Exception:
        needs_create = True

    if needs_create:
        await qdrant.create_collection(
            collection_name=config.collection_name,
            vectors_config={
                config.dense_vector_name: VectorParams(
                    size=config.embedding_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                config.sparse_vector_name: SparseVectorParams()
            },
        )
        print(f"  ✅ Created '{config.collection_name}' with dense + sparse vectors.")


def _sparse_embed_batch(texts: list[str]) -> list[SparseVector]:
    """
    Generate BM25 sparse vectors for a list of texts.
    Synchronous — CPU-only, no network required.
    """
    model = _get_sparse_model()
    return [
        SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )
        for emb in model.embed(texts)
    ]


async def _embed(chunks: list[dict], openai: AsyncOpenAI) -> list[list[float]]:
    """
    Dense-embed chunks in batches with TPM rate-limit handling.
    Retries once on 429 after a 60-second cooldown.
    """
    texts: list[str] = [c["text"] for c in chunks]
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        try:
            resp = await openai.embeddings.create(
                model=config.embedding_model,
                input=batch,
            )
        except RateLimitError:
            print(f"  Rate limit hit — waiting 60s before retrying batch {i // EMBED_BATCH_SIZE + 1}...")
            await asyncio.sleep(60)
            resp = await openai.embeddings.create(
                model=config.embedding_model,
                input=batch,
            )
        embeddings.extend([r.embedding for r in resp.data])
        print(f"  Dense: {min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)} chunks embedded...")

        if i + EMBED_BATCH_SIZE < len(texts):
            await asyncio.sleep(EMBED_BATCH_DELAY)

    return embeddings


async def _upload(
    chunks: list[dict],
    dense_embeddings: list[list[float]],
    sparse_embeddings: list[SparseVector],
    qdrant: AsyncQdrantClient,
) -> None:
    """Upsert chunks with both dense and sparse vectors into Qdrant."""
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={
                config.dense_vector_name:  dense_embeddings[i],
                config.sparse_vector_name: sparse_embeddings[i],
            },
            payload=chunks[i],
        )
        for i in range(len(chunks))
    ]
    batch_size = 200
    for i in range(0, len(points), batch_size):
        await qdrant.upsert(
            collection_name=config.collection_name,
            points=points[i : i + batch_size],
        )


# ── Public API ────────────────────────────────────────────────────────────────

async def index_pdf_bytes(content: bytes, filename: str) -> int:
    """
    Full ingestion pipeline for a PDF supplied as bytes.
    Returns the number of chunks indexed (0 if already indexed).
    Called by the FastAPI /ingest endpoint.
    """
    print(f"\n[Indexer] {filename}")

    qdrant = _qdrant()
    openai = _openai()

    if await _already_indexed(qdrant, filename):
        print(f"  ⏭️  Skipping '{filename}' — already indexed.")
        return 0

    pages  = load_pdf_bytes(content, filename)
    print(f"  {len(pages)} pages parsed.")

    chunks = chunk_pages(pages)
    print(f"  {len(chunks)} chunks created.")

    # Dense embeddings (async — OpenAI API)
    dense_embeddings = await _embed(chunks, openai)
    print(f"  {len(dense_embeddings)} dense embeddings generated.")

    # Sparse BM25 embeddings (sync — local fastembed)
    texts = [c["text"] for c in chunks]
    sparse_embeddings = await asyncio.to_thread(_sparse_embed_batch, texts)
    print(f"  {len(sparse_embeddings)} sparse BM25 embeddings generated.")

    await _ensure_collection(qdrant)
    await _upload(chunks, dense_embeddings, sparse_embeddings, qdrant)
    print(f"  ✅ Indexed {len(chunks)} chunks.")

    return len(chunks)


async def index_pdf_file(path: str | Path) -> int:
    """Full ingestion pipeline for a PDF on disk. Called by the CLI."""
    path = Path(path)
    return await index_pdf_bytes(path.read_bytes(), path.name)


# ── CLI ───────────────────────────────────────────────────────────────────────

async def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF(s) into Qdrant Cloud.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single PDF.")
    group.add_argument("--dir",  help="Path to a directory of PDFs.")
    args = parser.parse_args()

    if args.file:
        await index_pdf_file(args.file)
    else:
        pdf_files = list(Path(args.dir).glob("*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {args.dir}")
            return
        for pdf in pdf_files:
            await index_pdf_file(pdf)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(_cli_main())
