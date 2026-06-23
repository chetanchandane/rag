"""
FastAPI application — Clinical RAG API

Endpoints:
  GET  /health   → liveness check
  POST /query    → ask a compliance question
  POST /ingest   → upload and index a PDF
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langsmith import traceable
from pydantic import BaseModel, Field

from src.config import config
from src.ingestion.indexer import index_pdf_bytes
from src.retrieval.search import Searcher
from src.retrieval.reranker import Reranker
from src.generation.llm_client import ClaudeClient


# ── App lifecycle ──────────────────────────────────────────────────────────────

searcher:  Searcher  | None = None
reranker:  Reranker  | None = None
llm:       ClaudeClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global searcher, reranker, llm
    searcher = Searcher()
    reranker = Reranker()
    llm      = ClaudeClient()
    print("✅ RAG components ready.")
    yield


app = FastAPI(
    title=config.api_title,
    version=config.api_version,
    description="RAG over FDA/ICH clinical trial compliance documents.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=5,
        example="What is the required timeline for reporting SAEs under ICH E6?"
    )
    top_k: int = Field(default=5, ge=1, le=20)


class SourceRef(BaseModel):
    source: str
    page: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    chunks_used: int


class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_indexed: int


# ── RAG orchestration ─────────────────────────────────────────────────────────

@traceable(name="rag_pipeline", run_type="chain")
async def run_rag(question: str, top_k: int) -> dict:
    """
    Orchestrate the full RAG pipeline:
        search → rerank → generate

    Decorated with @traceable so LangSmith captures this as the
    top-level "chain" span, with search/rerank/generate as child spans.
    """
    # 1. Retrieve candidate chunks
    chunks = await searcher.search(question, top_k=top_k)

    # 2. Rerank (Stage 1: passthrough; Stage 2: Cohere cross-encoder)
    chunks = await reranker.rerank(question, chunks, top_n=top_k)

    # 3. Generate grounded answer
    answer = await llm.generate(question, chunks)

    return {
        "answer": answer,
        "sources": [
            {"source": c["source"], "page": c["page"], "score": c["score"]}
            for c in chunks
        ],
        "chunks_used": len(chunks),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "model": config.generation_model}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Submit a compliance question. Returns a grounded answer with citations.
    Every call is traced in LangSmith automatically.
    """
    if not searcher or not llm:
        raise HTTPException(503, "Service not ready.")
    try:
        result = await run_rag(request.question, top_k=request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF to be parsed, chunked, and indexed in Qdrant Cloud.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(400, "File is empty.")

    try:
        n = await index_pdf_bytes(content, file.filename)
        return IngestResponse(status="success", filename=file.filename, chunks_indexed=n)
    except Exception as e:
        raise HTTPException(500, str(e))
