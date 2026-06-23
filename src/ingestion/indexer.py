"""
indexer.py — Stage 3 of ingestion

Responsibility: embed chunks with OpenAI and upsert them into Qdrant Cloud.
Also the CLI entry point for ingesting documents from the command line.

Usage:
    python -m src.ingestion.indexer --file data/raw/ich_e6.pdf
    python -m src.ingestion.indexer --dir  data/raw/
"""

import asyncio
import argparse
import os
import uuid
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from src.config import config
from src.ingestion.loader import load_pdf_bytes
from src.ingestion.splitter import chunk_pages

# Tier-1 OpenAI accounts: 40k TPM limit for text-embedding-3-small.
# Keep batches small and pause between them to stay under the limit.
EMBED_BATCH_SIZE = 20       # chunks per API call (~10k tokens safely under 40k TPM)
EMBED_BATCH_DELAY = 1.5     # seconds to wait between batches


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
        return False  # collection doesn't exist yet — safe to proceed


async def _ensure_collection(qdrant: AsyncQdrantClient) -> None:
    """Create the Qdrant collection if it doesn't exist yet."""
    try:
        await qdrant.get_collection(config.collection_name)
    except Exception:
        await qdrant.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(
                size=config.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        print(f"  Created collection '{config.collection_name}'.")


async def _embed(chunks: list[dict], openai: AsyncOpenAI) -> list[list[float]]:
    """
    Embed chunks in small batches with a delay between each to avoid
    OpenAI's TPM (tokens-per-minute) rate limit on Tier-1 accounts.
    Retries once automatically on a 429 before giving up.
    """
    texts = [c["text"] for c in chunks]
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
        print(f"  Embedded {min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)} chunks...")

        # Pause between batches to stay under the TPM limit
        if i + EMBED_BATCH_SIZE < len(texts):
            await asyncio.sleep(EMBED_BATCH_DELAY)

    return embeddings


async def _upload(
    chunks: list[dict],
    embeddings: list[list[float]],
    qdrant: AsyncQdrantClient,
) -> None:
    """Upsert chunks + vectors into Qdrant in batches of 200."""
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=embeddings[i], payload=chunks[i])
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
    Returns the number of chunks indexed.
    Called by the FastAPI /ingest endpoint.
    """
    print(f"\n[Indexer] {filename}")

    qdrant = _qdrant()
    openai = _openai()

    if await _already_indexed(qdrant, filename):
        print(f"  ⏭️  Skipping '{filename}' — already indexed.")
        return 0

    pages = load_pdf_bytes(content, filename)
    print(f"  {len(pages)} pages parsed.")

    chunks = chunk_pages(pages)
    print(f"  {len(chunks)} chunks created.")

    embeddings = await _embed(chunks, openai)
    print(f"  {len(embeddings)} embeddings generated.")

    await _ensure_collection(qdrant)
    await _upload(chunks, embeddings, qdrant)
    print(f"  ✅ Indexed {len(chunks)} chunks.")

    return len(chunks)


async def index_pdf_file(path: str | Path) -> int:
    """
    Full ingestion pipeline for a PDF on disk.
    Called by the CLI.
    """
    path = Path(path)
    return await index_pdf_bytes(path.read_bytes(), path.name)


# ── CLI ───────────────────────────────────────────────────────────────────────

async def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF(s) into Qdrant Cloud.")
    group = parser.add_mutually_exclusive_group(required=True)
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
