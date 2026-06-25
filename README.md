# Clinical Trial Compliance RAG

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![LangSmith](https://img.shields.io/badge/LangSmith-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://smith.langchain.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com)
[![Render](https://img.shields.io/badge/Render-000000?style=flat-square&logo=render&logoColor=white)](https://render.com)
[![Claude](https://img.shields.io/badge/Claude-D97757?style=flat-square&logo=anthropic&logoColor=white)](https://www.anthropic.com/claude)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/embeddings)
[![Cohere](https://img.shields.io/badge/Cohere-39594D?style=flat-square&logo=cohere&logoColor=white)](https://cohere.com/rerank)
[![Ragas](https://img.shields.io/badge/Ragas-4B5563?style=flat-square&logo=python&logoColor=white)](https://docs.ragas.io)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)

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
| Embeddings | OpenAI `text-embedding-3-small` (dense) + fastembed BM25 (sparse) |
| Reranking | Cohere `rerank-english-v3.0` |
| Vector DB | Qdrant Cloud (hybrid dense + sparse collection) |
| Observability | LangSmith (traces + offline eval experiments) |
| Evaluation | Ragas 0.4 — Faithfulness, Answer Relevancy, Context Precision, Context Recall |
| Frontend | Vanilla JS + CSS (dark theme) |
| Deployment | Render |
| CI/CD | GitHub Actions — Ragas quality gate on every PR |

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
│   │   ├── search.py         # HyDE + hybrid BM25/dense search with RRF
│   │   └── reranker.py       # Cohere cross-encoder reranking
│   ├── generation/
│   │   ├── prompts.py        # System prompt and context formatting
│   │   └── llm_client.py     # Claude wrapper
│   ├── api/
│   │   └── main.py           # FastAPI app + RAG orchestration
│   └── evals/
│       ├── generate_dataset.py  # Synthetic Q&A generation → LangSmith Dataset
│       └── run_evals.py         # Ragas metrics → LangSmith Experiment + CI gate
├── ui/
│   ├── index.html            # App shell
│   ├── css/styles.css        # Dark theme, all 4 UI states
│   └── js/app.js             # State machine, API calls, markdown rendering
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── evals/
│       └── test_evals.py     # Eval logic unit tests (requires requirements-eval.txt)
├── .github/
│   └── workflows/
│       └── eval_gate.yml     # Ragas quality gate on every PR to main
├── .env.example
├── Dockerfile
├── render.yaml
├── requirements.txt
└── requirements-eval.txt     # Eval-only deps (never installed in production)
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
# Unit tests (no API keys required)
pytest tests/ -v --ignore=tests/evals

# Eval unit tests (requires: pip install -r requirements-eval.txt)
pytest tests/evals/ -v
```

Unit tests cover chunking, metadata preservation, RRF merge logic, HyDE, and Cohere reranker — no external API calls.

---

## Offline Evaluation (Phase 3)

Evaluation runs outside of production — no changes to `requirements.txt` or the Dockerfile.

**Step 1 — Install eval dependencies (one time)**

```bash
pip install -r requirements-eval.txt
```

**Step 2 — Generate the golden dataset**

```bash
# Smoke test first (3 questions, ~$0.01)
python -m src.evals.generate_dataset --sample 3 --dataset clinical-rag-golden-set-smoketest

# Full dataset (30 questions)
python -m src.evals.generate_dataset --sample 30 --dataset clinical-rag-golden-set
```

This samples chunks from Qdrant, asks Claude to generate one Q&A pair per chunk, and uploads examples to a LangSmith Dataset. View at [smith.langchain.com](https://smith.langchain.com) → **Datasets**.

**Step 3 — Run Ragas evals**

```bash
# Full run (all examples)
python -m src.evals.run_evals --dataset clinical-rag-golden-set

# CI mode (first 15 examples — what GitHub Actions runs)
python -m src.evals.run_evals --dataset clinical-rag-golden-set --ci
```

Scores appear as a **LangSmith Experiment** under Datasets → `clinical-rag-golden-set` → **Experiments**.

**Metrics and thresholds (CI gate)**

| Metric | Threshold | What it measures |
|---|---|---|
| Faithfulness | ≥ 0.70 | Does the answer stay grounded in the retrieved context? |
| Answer Relevancy | ≥ 0.70 | Does the answer address the question? |
| Context Precision | ≥ 0.60 | Are the retrieved chunks relevant to the ground truth? |
| Context Recall | ≥ 0.60 | Do the retrieved chunks cover the ground truth? |

All metrics use Claude Haiku as the judge LLM (cheaper than Sonnet for high-volume scoring).

---

## Deployment (Render)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo
3. Render auto-detects `render.yaml`
4. Add your API keys under **Environment → Secret Files**
5. Deploy — you'll get a live URL

---

## Roadmap

**Stage 1 (complete)** — Basic RAG: dense search, Claude generation, LangSmith tracing, dark-theme UI

**Stage 2 (complete)** — Advanced retrieval: HyDE query expansion, hybrid BM25 + dense search with RRF, Cohere cross-encoder reranking

**Stage 3 (complete)** — Evaluation: synthetic golden dataset via Claude, Ragas metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall), LangSmith Experiments, GitHub Actions CI/CD quality gate
