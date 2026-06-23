"""
Tests for retrieval: reranker passthrough and prompt building.

Kept lightweight in Stage 1 — no live API calls.
Integration tests against Qdrant/Claude live in tests/integration/ (Stage 3).

Run with:  pytest tests/test_retrieval.py -v
"""

import pytest
from src.retrieval.reranker import Reranker
from src.generation.prompts import build_user_message, SYSTEM_PROMPT


SAMPLE_CHUNKS = [
    {"text": "SAEs must be reported within 24 hours.", "source": "ich_e6.pdf", "page": 12, "score": 0.91},
    {"text": "The sponsor must notify all investigators.", "source": "fda_guidance.pdf", "page": 5, "score": 0.78},
    {"text": "Protocol deviations require IRB notification.", "source": "ich_e6.pdf", "page": 23, "score": 0.65},
]


# ── Reranker tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reranker_passthrough_returns_all_chunks():
    """Stage 1 reranker should return all chunks unchanged."""
    reranker = Reranker()
    result = await reranker.rerank("What is the SAE reporting timeline?", SAMPLE_CHUNKS)
    assert len(result) == len(SAMPLE_CHUNKS)


@pytest.mark.asyncio
async def test_reranker_top_n_truncates():
    """Reranker with top_n should return at most top_n chunks."""
    reranker = Reranker()
    result = await reranker.rerank("SAE reporting", SAMPLE_CHUNKS, top_n=2)
    assert len(result) <= 2


@pytest.mark.asyncio
async def test_reranker_preserves_chunk_keys():
    """Reranked chunks must still have text, source, page, score keys."""
    reranker = Reranker()
    result = await reranker.rerank("SAE reporting", SAMPLE_CHUNKS)
    for chunk in result:
        assert "text" in chunk
        assert "source" in chunk
        assert "page" in chunk
        assert "score" in chunk


# ── Prompt builder tests ──────────────────────────────────────────────────────

def test_build_user_message_contains_question():
    msg = build_user_message("What is the SAE timeline?", SAMPLE_CHUNKS)
    assert "What is the SAE timeline?" in msg


def test_build_user_message_contains_sources():
    msg = build_user_message("What is the SAE timeline?", SAMPLE_CHUNKS)
    assert "ich_e6.pdf" in msg
    assert "fda_guidance.pdf" in msg


def test_build_user_message_empty_contexts():
    """With no contexts, message should still be returned (not raise)."""
    msg = build_user_message("Any question?", [])
    assert "Any question?" in msg
    assert isinstance(msg, str)


def test_system_prompt_is_non_empty():
    assert len(SYSTEM_PROMPT) > 50
