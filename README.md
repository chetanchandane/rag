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

A production-grade Retrieval-Augmented Generation system for querying FDA and ICH regulatory documents. Ask compliance questions in plain English and get grounded, **fully source-cited** answers — engineered for an accuracy-critical domain where hallucinations carry real regulatory and patient-safety risk.

---

## Key Achievements

- Built and deployed a production RAG system over FDA/ICH clinical-trial regulatory documents (FastAPI, Qdrant, Claude), generating fully source-cited answers at **79% Faithfulness** and **75% Answer Relevancy** (Ragas) across a 294-question evaluation set.
- Engineered an advanced retrieval pipeline — HyDE query expansion, hybrid BM25/dense search with RRF fusion, and Cohere cross-encoder reranking (20→5) — lifting **Context Precision +37%** and **Context Recall +24%** over a dense-only baseline to reach **80% / 89%**.
- Built an automated Ragas evaluation harness (4 metrics, LLM-as-judge) wired into a **GitHub Actions CI gate** that blocks regressing deploys, with end-to-end **LangSmith tracing** for per-stage observability.

---

## Results

Measured on a 294-question golden evaluation set. Judge: Claude Haiku · Generation: Claude Sonnet 4.6.

| Metric | Full pipeline | Dense-only baseline | Lift |
|---|:--:|:--:|:--:|
| Faithfulness | **0.79** | — | — |
| Answer Relevancy | **0.75** | — | — |
| Context Precision | **0.80** | 0.59 | **+37.1%** |
| Context Recall | **0.89** | 0.72 | **+24.3%** |

The lift columns isolate the contribution of advanced retrieval (HyDE + hybrid + RRF + Cohere rerank) against a plain dense-vector baseline on the same corpus and questions.

**Latency** (end-to-end, seconds): median **15.2** · p95 **18.9** · mean 20.9. Per-stage means show generation (10.5s) and HyDE expansion (9.4s) dominate, while retrieval itself (embed + hybrid search + rerank) totals under 1s — the clear optimization target.

> Reproduce these numbers locally with `python -m src.evals.resume_metrics` (see [Evaluation](#evaluation)). Full breakdown saved in `eval_results/`.

---

## Architecture

```
User Query
    │
    ▼
FastAPI (/query)
    │
    ├── HyDE expansion (Claude) ──► hypothetical regulatory passage
    │           │
    │           ├── OpenAI text-embedding-3-small ─► Qdrant dense search
    │           └── fastembed BM25 ────────────────► Qdrant sparse search
    │                                  │
    │                          RRF fusion (k=60) ─► top-20 candidates
    │                                  │
    │                          Cohere rerank ─────► top-5 chunks
    │                                  │
    └── Claude (claude-sonnet-4-6) ◄── retrieved context
                │
                ▼
        Answer + Citations  ──►  LangSmith (auto-traced, per-stage spans)
```

---

## Tech Stack

| Component | Tool |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Anthropic Claude (`claude-sonnet-4-6`) |
| Embeddings | OpenAI `text-embedding-3-small` (dense) + fastembed BM25 (sparse) |
| Query expansion | HyDE (Hypothetical Document Embeddings) |
| Reranking | Cohere `rerank-english-v3.0` (cross-encoder) |
| Vector DB | Qdrant Cloud (hybrid dense + sparse collection) |
| Observability | LangSmith (per-stage traces + offline eval experiments) |
| Evaluation | Ragas 0.4 — Faithfulness, Answer Relevancy, Context Precision, Context Recall |
| Frontend | Vanilla JS + CSS (dark theme) |
| Deployment | Docker + Render |
| CI/CD | GitHub Actions — Ragas quality gate on every PR |

---

## How It Works

**Retrieval (`src/retrieval/`)**

1. **HyDE** — Claude drafts a hypothetical regulatory passage that would answer the query; embedding this dense text aligns better than embedding the short question.
2. **Hybrid search** — the hypothetical doc drives dense vector search (OpenAI) while the original query drives sparse BM25 search; both run concurrently against Qdrant (top-20 each).
3. **RRF fusion** — Reciprocal Rank Fusion (k=60) merges the two ranked lists into a single deduplicated candidate set.
4. **Reranking** — Cohere's cross-encoder rescores each (query, chunk) pair directly, narrowing 20 candidates to the 5 most relevant.

**Generation (`src/generation/`)** — Claude answers strictly from the retrieved context, returning inline source citations (document + page).

**Observability** — every stage (`embed_query`, `hybrid_search`, `rerank`, `generate`) is auto-traced to LangSmith with latency and token counts.

---

## Project Structure

```
RAG/
├── data/
│   ├── raw/                  # Source PDFs (FDA / ICH guidance)
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
│       ├── run_evals.py         # Ragas metrics → LangSmith Experiment + CI gate
│       └── resume_metrics.py    # Local exact scoring + dense-only baseline + latency
├── ui/                       # Vanilla JS dark-theme frontend
├── tests/                    # Unit tests + eval logic tests
├── eval_results/             # Saved metric runs (JSON / CSV / markdown)
├── .github/workflows/
│   └── eval_gate.yml         # Ragas quality gate on every PR to main
├── Dockerfile
├── render.yaml
├── requirements.txt
└── requirements-eval.txt     # Eval-only deps (never installed in production)
```

---

## Quick Start

### Prerequisites

API keys from: [Anthropic](https://console.anthropic.com) (Claude), [OpenAI](https://platform.openai.com) (embeddings), [Qdrant Cloud](https://cloud.qdrant.io), [Cohere](https://cohere.com), and [LangSmith](https://smith.langchain.com). [Render](https://render.com) is optional, for deployment.

### Setup

```bash
git clone https://github.com/chetanchandane/rag.git
cd rag

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then fill in all API keys
```

### Ingest documents

Place PDFs in `data/raw/` (start with [ICH E6(R3) Good Clinical Practice](https://database.ich.org/sites/default/files/ICH_E6(R3)_Step4_FinalGuideline_2025_0106.pdf)), then:

```bash
python -m src.ingestion.indexer --dir data/raw/                 # whole folder
python -m src.ingestion.indexer --file data/raw/ICH_E6_R3.pdf   # single file
```

Re-running is safe — already-indexed files are skipped.

### Run

```bash
uvicorn src.api.main:app --reload
```

Open **http://localhost:8000** — the UI loads directly.

---

## API Reference

**Ask a compliance question**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the required timeline for reporting SAEs under ICH E6?", "top_k": 5}'
```

```json
{
  "answer": "Under ICH E6(R3), serious adverse events must be reported... [Source: ICH_E6_R3.pdf, p.34]",
  "sources": [{"source": "ICH_E6_R3.pdf", "page": 34, "score": 0.91}],
  "chunks_used": 4
}
```

**Other endpoints:** `GET /health` · `POST /ingest` (multipart PDF upload).

---

## Evaluation

Evaluation runs entirely outside production — no changes to `requirements.txt` or the Dockerfile.

```bash
pip install -r requirements-eval.txt      # one-time, eval-only deps
```

**1. Generate a golden dataset** — samples chunks from Qdrant and has Claude generate Q&A pairs (concurrent; supports multiple pairs per chunk).

```bash
python -m src.evals.generate_dataset \
  --sample 150 --per-chunk 3 --concurrency 5 \
  --dataset clinical-rag-golden-set
```

**2. Score exact metrics locally** — runs the full pipeline plus a dense-only baseline, capturing Ragas scores, the precision/recall lift, and per-stage latency. Checkpoints after every example and resumes if interrupted; results land in `eval_results/<dataset>/`.

```bash
python -m src.evals.resume_metrics --dataset clinical-rag-golden-set
python -m src.evals.resume_metrics --dataset clinical-rag-golden-set --limit 80   # quick sample
```

**3. CI quality gate** — `run_evals.py` scores a sample and logs a LangSmith Experiment, exiting non-zero if any metric misses its threshold. Runs automatically on every PR via `.github/workflows/eval_gate.yml`.

| Metric | CI threshold | Measured |
|---|:--:|:--:|
| Faithfulness | ≥ 0.70 | 0.79 |
| Answer Relevancy | ≥ 0.70 | 0.75 |
| Context Precision | ≥ 0.60 | 0.80 |
| Context Recall | ≥ 0.60 | 0.89 |

All metrics use Claude Haiku as the judge LLM (cheaper than Sonnet for high-volume scoring).

---

## Testing

```bash
pytest tests/ -v --ignore=tests/evals        # unit tests, no API keys needed
pytest tests/evals/ -v                        # eval logic (needs requirements-eval.txt)
```

Unit tests cover chunking, metadata preservation, RRF merge logic, HyDE, and the Cohere reranker — no external API calls.

---

## Deployment (Render)

1. Push to GitHub.
2. [render.com](https://render.com) → **New Web Service** → connect the repo (auto-detects `render.yaml`).
3. Add API keys under **Environment → Secret Files**.
4. Deploy — you get a live URL.

---

## Roadmap

- **Stage 1 ✅** — Core RAG: dense search, Claude generation, LangSmith tracing, dark-theme UI.
- **Stage 2 ✅** — Advanced retrieval: HyDE, hybrid BM25 + dense with RRF, Cohere reranking.
- **Stage 3 ✅** — Evaluation: synthetic golden dataset, 4 Ragas metrics, LangSmith Experiments, GitHub Actions CI gate.
- **Stage 4 ◻️** — Latency optimization: parallelize/cache HyDE, stream generation to cut p95.
