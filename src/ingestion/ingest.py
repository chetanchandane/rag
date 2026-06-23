# DEPRECATED — superseded by the split modules below.
# This file is kept only for backward compatibility.
# Use the following instead:
#   src/ingestion/loader.py   → parse PDFs
#   src/ingestion/splitter.py → chunk text
#   src/ingestion/indexer.py  → embed + upload + CLI

from src.ingestion.indexer import index_pdf_bytes as ingest_pdf  # noqa: F401
