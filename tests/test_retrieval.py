"""
Tests for retrieval: Searcher (RRF merge + HyDE), Reranker (Cohere), prompt building.

All external API calls (Anthropic, OpenAI, Qdrant, Cohere) are mocked so no
API keys or network access are required.

Run with:  pytest tests/ -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from src.retrieval.search import Searcher
from src.retrieval.reranker import Reranker
from src.generation.prompts import build_user_message, SYSTEM_PROMPT


SAMPLE_CHUNKS = [
    {"text": "SAEs must be reported within 24 hours.", "source": "ich_e6.pdf", "page": 12, "score": 0.91},
    {"text": "The sponsor must notify all investigators.", "source": "fda_guidance.pdf", "page": 5, "score": 0.78},
    {"text": "Protocol deviations require IRB notification.", "source": "ich_e6.pdf", "page": 23, "score": 0.65},
]


# ── RRF merge tests (pure logic — no mocks needed) ───────────────────────────

def _make_hit(id_: str, payload: dict, score: float = 0.9):
    h = SimpleNamespace()
    h.id      = id_
    h.payload = payload
    h.score   = score
    return h


def test_rrf_merge_deduplicates():
    """Same point ID appearing in both lists should appear exactly once."""
    payload = {"text": "x", "source": "a.pdf", "page": 1}
    dense   = [_make_hit("id-1", payload), _make_hit("id-2", payload)]
    sparse  = [_make_hit("id-1", payload), _make_hit("id-3", payload)]
    result  = Searcher._rrf_merge(dense, sparse)
    ids     = [r["source"] for r in result]  # can't check id directly, check count
    assert len(result) == 3  # id-1, id-2, id-3


def test_rrf_merge_boost_overlap():
    """A doc in both lists should score higher than one in only one list."""
    payload_a = {"text": "a", "source": "a.pdf", "page": 1}
    payload_b = {"text": "b", "source": "b.pdf", "page": 2}
    dense  = [_make_hit("id-A", payload_a), _make_hit("id-B", payload_b)]
    sparse = [_make_hit("id-A", payload_a)]          # id-A in both
    result = Searcher._rrf_merge(dense, sparse)
    # id-A should rank first (boosted by appearing in both lists)
    assert result[0]["source"] == "a.pdf"


def test_rrf_merge_sorted_descending():
    """Output should always be sorted by RRF score descending."""
    payloads = [{"text": str(i), "source": f"{i}.pdf", "page": i} for i in range(5)]
    dense    = [_make_hit(str(i), payloads[i]) for i in range(5)]
    sparse   = [_make_hit(str(i), payloads[i]) for i in range(4, -1, -1)]
    result   = Searcher._rrf_merge(dense, sparse)
    scores   = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_merge_empty_lists():
    """Both lists empty → empty result."""
    assert Searcher._rrf_merge([], []) == []


def test_rrf_merge_one_empty():
    """One empty list → results from the non-empty list only."""
    payload = {"text": "a", "source": "a.pdf", "page": 1}
    dense   = [_make_hit("id-1", payload)]
    result  = Searcher._rrf_merge(dense, [])
    assert len(result) == 1


# ── HyDE test ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hyde_expand_query_returns_string():
    """expand_query should call Anthropic and return the text content."""
    with patch("src.retrieval.search.AsyncAnthropic") as mock_anthropic_cls:
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client

        fake_content = SimpleNamespace(text="Hypothetical regulatory passage.")
        mock_client.messages.create = AsyncMock(
            return_value=SimpleNamespace(content=[fake_content])
        )

        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }):
            with patch("src.retrieval.search.AsyncOpenAI"), \
                 patch("src.retrieval.search.AsyncQdrantClient"):
                searcher = Searcher()
                searcher.anthropic = mock_client
                result = await searcher.expand_query("What is the SAE reporting timeline?")

        assert isinstance(result, str)
        assert len(result) > 0


# ── Reranker tests (Cohere mocked) ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reranker_returns_top_n():
    """Reranker should return at most top_n chunks."""
    with patch("src.retrieval.reranker.cohere.Client") as mock_cohere_cls:
        mock_co = MagicMock()
        mock_cohere_cls.return_value = mock_co

        # Simulate Cohere returning 2 results (top_n=2)
        mock_co.rerank.return_value = SimpleNamespace(results=[
            SimpleNamespace(index=0, relevance_score=0.95),
            SimpleNamespace(index=1, relevance_score=0.72),
        ])

        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            reranker = Reranker()
            reranker._co = mock_co
            result = await reranker.rerank("SAE reporting", SAMPLE_CHUNKS, top_n=2)

        assert len(result) == 2


@pytest.mark.asyncio
async def test_reranker_adds_rerank_score():
    """Each returned chunk must have a rerank_score key."""
    with patch("src.retrieval.reranker.cohere.Client") as mock_cohere_cls:
        mock_co = MagicMock()
        mock_cohere_cls.return_value = mock_co

        mock_co.rerank.return_value = SimpleNamespace(results=[
            SimpleNamespace(index=0, relevance_score=0.88),
            SimpleNamespace(index=2, relevance_score=0.61),
        ])

        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            reranker = Reranker()
            reranker._co = mock_co
            result = await reranker.rerank("SAE reporting", SAMPLE_CHUNKS, top_n=2)

        for chunk in result:
            assert "rerank_score" in chunk
            assert isinstance(chunk["rerank_score"], float)


@pytest.mark.asyncio
async def test_reranker_preserves_chunk_keys():
    """Reranked chunks must still have text, source, page, score keys."""
    with patch("src.retrieval.reranker.cohere.Client") as mock_cohere_cls:
        mock_co = MagicMock()
        mock_cohere_cls.return_value = mock_co

        mock_co.rerank.return_value = SimpleNamespace(results=[
            SimpleNamespace(index=0, relevance_score=0.9),
        ])

        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            reranker = Reranker()
            reranker._co = mock_co
            result = await reranker.rerank("SAE", SAMPLE_CHUNKS, top_n=1)

        chunk = result[0]
        for key in ("text", "source", "page", "score"):
            assert key in chunk


@pytest.mark.asyncio
async def test_reranker_empty_chunks():
    """Empty chunk list should return empty list without calling Cohere."""
    with patch("src.retrieval.reranker.cohere.Client") as mock_cohere_cls:
        mock_co = MagicMock()
        mock_cohere_cls.return_value = mock_co

        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            reranker = Reranker()
            reranker._co = mock_co
            result = await reranker.rerank("anything", [], top_n=5)

        assert result == []
        mock_co.rerank.assert_not_called()


# ── Prompt builder tests ──────────────────────────────────────────────────────

def test_build_user_message_contains_question():
    msg = build_user_message("What is the SAE timeline?", SAMPLE_CHUNKS)
    assert "What is the SAE timeline?" in msg


def test_build_user_message_contains_sources():
    msg = build_user_message("What is the SAE timeline?", SAMPLE_CHUNKS)
    assert "ich_e6.pdf" in msg
    assert "fda_guidance.pdf" in msg


def test_build_user_message_empty_contexts():
    msg = build_user_message("Any question?", [])
    assert "Any question?" in msg
    assert isinstance(msg, str)


def test_system_prompt_is_non_empty():
    assert len(SYSTEM_PROMPT) > 50
