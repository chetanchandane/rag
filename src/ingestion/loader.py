"""
loader.py — Stage 1 of ingestion

Responsibility: read raw files from disk or bytes and return a list of
page-level dicts.  Nothing else — no chunking, no embedding here.

Supported formats (Stage 1): PDF
Planned (Stage 2):           DOCX, HTML, Markdown
"""

import io
from pathlib import Path

from pypdf import PdfReader


def load_pdf_bytes(content: bytes, filename: str) -> list[dict]:
    """
    Parse a PDF from raw bytes.

    Returns a list of page dicts:
        [{"text": "...", "page": 1, "source": "filename.pdf"}, ...]

    Pages with no extractable text are silently skipped.
    """
    reader = PdfReader(io.BytesIO(content))
    pages = []

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({
                "text": text,
                "page": i + 1,
                "source": filename,
            })

    return pages


def load_pdf_file(path: str | Path) -> list[dict]:
    """
    Convenience wrapper — load a PDF from a file path.
    """
    path = Path(path)
    content = path.read_bytes()
    return load_pdf_bytes(content, path.name)
