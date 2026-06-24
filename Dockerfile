# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the BM25 fastembed model so it's baked into the image.
# This avoids a network fetch on the first request in production.
RUN PYTHONPATH=/install/lib/python3.12/site-packages \
    python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')"


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code and frontend (not data/ or .env)
COPY src/ ./src/
COPY ui/ ./ui/

# Non-root user for security
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Render injects $PORT; locally defaults to 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
