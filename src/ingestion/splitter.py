"""
splitter.py — Stage 2 of ingestion

Responsibility: take page-level text dicts and split them into
overlapping chunks ready for embedding.

All chunking strategy decisions live here.  To experiment with
different chunk sizes or strategies, edit this file only.

Stage 1: Fixed-size recursive character splitting
Stage 2 (planned): Hierarchical parent-child splitting
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split each page into overlapping text chunks.

    Input:  list of {"text", "page", "source"} dicts  (from loader.py)
    Output: list of {"text", "page", "source"} dicts  (chunk-level)

    Chunk size and overlap are read from src/config.py.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            text = split.strip()
            if text:
                chunks.append({
                    "text": text,
                    "page": page["page"],
                    "source": page["source"],
                })

    return chunks
