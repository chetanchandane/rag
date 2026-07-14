"""
Settings for the relational (text-to-SQL) backend.

Deliberately decoupled from src/config.py: that config hard-requires the Qdrant
env vars at import time, but the SQL path must be runnable on its own (and
testable without a vector store). Everything here reads os.environ with safe
defaults, so importing this module never raises.
"""

import os
from dataclasses import dataclass, field

try:  # optional: load a local .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


@dataclass
class SQLSettings:

    # ── Postgres ─────────────────────────────────────────────────────────────
    # Ingestion/migrations use the OWNER role (write). The app query path uses a
    # separate READ-ONLY role. Keep both URLs out of source control.
    database_url: str = field(
        default_factory=lambda: os.environ.get("DATABASE_URL", "").strip()
    )
    database_url_readonly: str = field(
        default_factory=lambda: os.environ.get("DATABASE_URL_RO", "").strip()
    )

    # ── SQL governance ───────────────────────────────────────────────────────
    allowed_tables: tuple[str, ...] = ("approvals", "recalls")
    sql_row_cap: int = 100                 # max rows any generated query may return
    sql_statement_timeout_ms: int = 5000   # per-statement timeout (set on read-only role)

    # ── Text-to-SQL generation ───────────────────────────────────────────────
    generation_model: str = field(
        default_factory=lambda: os.environ.get("SQL_GEN_MODEL", "claude-sonnet-4-6").strip()
    )
    generation_max_tokens: int = 1024

    # ── openFDA ──────────────────────────────────────────────────────────────
    openfda_base_url: str = field(
        default_factory=lambda: os.environ.get("OPENFDA_BASE_URL", "https://api.fda.gov").strip()
    )
    openfda_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENFDA_API_KEY", "").strip()
    )
    openfda_page_size: int = 100          # rows per request (openFDA max is 1000)
    openfda_max_records: int = 300        # Day 1 cap — keep the baseline eval fast


settings = SQLSettings()
