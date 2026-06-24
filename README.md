# Clinical Trial Compliance RAG

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![LangSmith](https://img.shields.io/badge/LangSmith-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://smith.langchain.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com)
[![Render](https://img.shields.io/badge/Render-000000?style=flat-square&logo=render&logoColor=white)](https://render.com)

A production-grade Retrieval-Augmented Generation system for querying FDA and ICH regulatory documents. Ask compliance questions in plain English and get grounded, cited answers — no hallucinations.

---

## Architecture

```
User Query
    │
    ▼
FastAPI (/query)
    │
    ├── OpenAI text-embedding-3-small  →  Qdrant Cloud (vector search)
    │                                          │
    │                                     Top-K chunks
    │                                          │
    └── Claude claude-sonnet-4-6  ←───── Retrieved context
                │
                ▼
        Answer + Citations
                │
                ▼
        LangSmith (auto-traced)
```

---

## Tech Stack

| Component | Tool |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Anthropic Claude (`claude-sonnet-4-6`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | Qdrant Cloud |
| Observability | LangSmith |
| Frontend | Vanilla JS + CSS (dark theme) |
| Deployment | Render |

---

## Project Structure

```
RAG/
├── data/
│   ├── raw/                  # Place source PDFs here
│   └── processed/            # Intermediate outputs
├── src/
│   ├── config.py             # All settings (models, chunk size, thresholds)
│   ├── ingestion/
│   │   ├── loader.py         # PDF parsing
│   │   ├── splitter.py       # Text chunking
│   │   └── indexer.py        # Embed + upload to Qdrant (CLI entry point)
│   ├── retrieval/
│   │   ├── search.py         # Qdrant vector search
│   │   └── reranker.py       # Re-ranking (passthrough in Stage 1)
│   ├── generation/
│   │   ├── prompts.py        # System prompt and context formatting
│   │   └── llm_client.py     # Claude wrapper
│   └── api/
│       └── main.py           # FastAPI app + RAG orchestration
├── ui/
│   ├── index.html            # App shell
│   ├── css/styles.css        # Dark theme, all 4 UI states
│   └── js/app.js             # State machine, API calls, markdown rendering
├── tests/
│   ├── test_ingestion.py
│   └── test_retrieval.py
├── .env.example
├── Dockerfile
├── render.yaml
└── requirements.txt
```

---

## Prerequisites

You'll need accounts and API keys from:

- [Anthropic](https://console.anthropic.com) — Claude API key
- [OpenAI](https://platform.openai.com) — Embeddings API key
- [Qdrant Cloud](https://cloud.qdrant.io) — Free cluster URL + API key
- [LangSmith](https://smith.langchain.com) — API key
- [Render](https://render.com) — For deployment (free tier works)

---

## Local Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/chetanchandane/rag.git
cd rag

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and fill in all API keys
```

---

## Data Ingestion

**Download source documents** (free, no sign-up needed):

- [ICH E6(R3) Good Clinical Practice](https://database.ich.org/sites/default/files/ICH_E6(R3)_Step4_FinalGuideline_2025_0106.pdf) ← start here
- [FDA Clinical Trials Guidance Documents](https://www.fda.gov/science-research/clinical-trials-and-human-subject-protection/clinical-trials-guidance-documents)

Place PDFs in `data/raw/`, then run:

```bash
# Single file
python -m src.ingestion.indexer --file data/raw/ICH_E6_R3.pdf

# Entire folder
python -m src.ingestion.indexer --dir data/raw/
```

Re-running is safe — already-indexed files are automatically skipped.

---

## Running Locally

```bash
uvicorn src.api.main:app --reload
```

Open **http://localhost:8000** — the UI loads directly.

---

## API Reference

**Health check**
```bash
curl http://localhost:8000/health
```

**Ask a compliance question**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the required timeline for reporting SAEs under ICH E6?", "top_k": 5}'
```

Response:
```json
{
  "answer": "Under ICH E6(R3), serious adverse events must be reported... [Source: ICH_E6_R3.pdf, p.34]",
  "sources": [
    {"source": "ICH_E6_R3.pdf", "page": 34, "score": 0.91}
  ],
  "chunks_used": 4
}
```

**Ingest a PDF via API**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/raw/ICH_E6_R3.pdf"
```

---

## LangSmith Tracing

Every `/query` call is automatically traced with no extra code. Set these in `.env`:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=clinical-rag
```

View traces at [smith.langchain.com](https://smith.langchain.com). Each trace shows child spans for `embed_query`, `vector_search`, `rerank`, and `generate` with latency and token counts.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover chunking logic, metadata preservation, reranker passthrough, and prompt formatting — no API keys required.

---

## Deployment (Render)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo
3. Render auto-detects `render.yaml`
4. Add your API keys under **Environment → Secret Files**
5. Deploy — you'll get a live URL

---

## Roadmap

**Stage 1 (current)** — Basic RAG: dense search, Claude generation, LangSmith tracing

**Stage 2** — Advanced retrieval: hybrid BM25 + dense search, Cohere cross-encoder reranking, HyDE query expansion

**Stage 3** — Evaluation: synthetic golden dataset, Ragas metrics (Faithfulness, Context Precision), CI/CD quality gate
