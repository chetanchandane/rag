"""
Day 1 ingestion — openFDA drug/enforcement -> Postgres `recalls` table.

Independent mirror of ingest_approvals: pulls FDA drug recall (enforcement)
records, maps each to one `recalls` row (grain = recall_number), parses the
FDA date fields, and upserts idempotently.

Governance and design mirror the approvals script: capped by default, OWNER
role for writes, psycopg lazy-imported so --dry-run needs no driver.

Usage
-----
    python -m src.sql.ingest_recalls --dry-run --limit 20
    python -m src.sql.ingest_recalls --apply-schema
    python -m src.sql.ingest_recalls --search 'classification:"Class I"'
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from src.sql.settings import settings

ENFORCEMENT_ENDPOINT = "/drug/enforcement.json"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_SEARCH = 'product_type:"Drugs"'  # restrict enforcement feed to drugs

try:
    import certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:  # pragma: no cover - certifi optional
    _SSL_CONTEXT = ssl.create_default_context()


# ── openFDA fetch ────────────────────────────────────────────────────────────
def _build_url(skip: int, limit: int, search: str | None) -> str:
    params: dict[str, Any] = {"limit": limit, "skip": skip}
    if search:
        params["search"] = search
    if settings.openfda_api_key:
        params["api_key"] = settings.openfda_api_key
    return f"{settings.openfda_base_url}{ENFORCEMENT_ENDPOINT}?{urllib.parse.urlencode(params)}"


def fetch_records(
    search: str | None = DEFAULT_SEARCH,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Page through openFDA until max_records collected (or data exhausted)."""
    cap = max_records or settings.openfda_max_records
    collected: list[dict[str, Any]] = []
    skip = 0
    while len(collected) < cap:
        page = min(settings.openfda_page_size, cap - len(collected))
        url = _build_url(skip, page, search)
        with urllib.request.urlopen(url, timeout=30, context=_SSL_CONTEXT) as resp:  # noqa: S310
            payload = json.load(resp)
        results = payload.get("results", [])
        if not results:
            break
        collected.extend(results)
        skip += len(results)
        total = payload.get("meta", {}).get("results", {}).get("total", 0)
        if skip >= total:
            break
    return collected[:cap]


# ── flatten ──────────────────────────────────────────────────────────────────
def _parse_fda_date(raw: str | None) -> date | None:
    if not raw or len(raw) != 8:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def flatten_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """One enforcement record -> one recalls row (None if no recall_number)."""
    recall_number = rec.get("recall_number")
    if not recall_number:
        return None

    openfda = rec.get("openfda") or {}

    def _first(key: str) -> str | None:
        val = openfda.get(key)
        return val[0] if isinstance(val, list) and val else None

    return {
        "recall_number": recall_number,
        "event_id": rec.get("event_id"),
        "product_description": rec.get("product_description"),
        "brand_name": _first("brand_name"),
        "generic_name": _first("generic_name"),
        "recalling_firm": rec.get("recalling_firm"),
        "reason_for_recall": rec.get("reason_for_recall"),
        "classification": rec.get("classification"),
        "status": rec.get("status"),
        "voluntary_mandated": rec.get("voluntary_mandated"),
        "distribution_pattern": rec.get("distribution_pattern"),
        "city": rec.get("city"),
        "state": rec.get("state"),
        "country": rec.get("country"),
        "recall_initiation_date": _parse_fda_date(rec.get("recall_initiation_date")),
        "report_date": _parse_fda_date(rec.get("report_date")),
        "termination_date": _parse_fda_date(rec.get("termination_date")),
    }


def flatten_all(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten and de-duplicate on the recall_number PK."""
    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        row = flatten_record(rec)
        if row:
            seen[row["recall_number"]] = row
    return list(seen.values())


# ── database ─────────────────────────────────────────────────────────────────
_UPSERT_SQL = """
INSERT INTO recalls (
    recall_number, event_id, product_description, brand_name, generic_name,
    recalling_firm, reason_for_recall, classification, status, voluntary_mandated,
    distribution_pattern, city, state, country,
    recall_initiation_date, report_date, termination_date
) VALUES (
    %(recall_number)s, %(event_id)s, %(product_description)s, %(brand_name)s, %(generic_name)s,
    %(recalling_firm)s, %(reason_for_recall)s, %(classification)s, %(status)s, %(voluntary_mandated)s,
    %(distribution_pattern)s, %(city)s, %(state)s, %(country)s,
    %(recall_initiation_date)s, %(report_date)s, %(termination_date)s
)
ON CONFLICT (recall_number) DO UPDATE SET
    event_id               = EXCLUDED.event_id,
    product_description    = EXCLUDED.product_description,
    brand_name             = EXCLUDED.brand_name,
    generic_name           = EXCLUDED.generic_name,
    recalling_firm         = EXCLUDED.recalling_firm,
    reason_for_recall      = EXCLUDED.reason_for_recall,
    classification         = EXCLUDED.classification,
    status                 = EXCLUDED.status,
    voluntary_mandated     = EXCLUDED.voluntary_mandated,
    distribution_pattern   = EXCLUDED.distribution_pattern,
    city                   = EXCLUDED.city,
    state                  = EXCLUDED.state,
    country                = EXCLUDED.country,
    recall_initiation_date = EXCLUDED.recall_initiation_date,
    report_date            = EXCLUDED.report_date,
    termination_date       = EXCLUDED.termination_date;
"""


def _connect():
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env (owner role for ingestion)."
        )
    import psycopg  # lazy import so --dry-run needs no driver

    return psycopg.connect(settings.database_url)


def apply_schema() -> None:
    ddl = SCHEMA_PATH.read_text()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(ddl)
        conn.commit()


def upsert_rows(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    with _connect() as conn, conn.cursor() as cur:
        cur.executemany(_UPSERT_SQL, rows)
        conn.commit()
    return len(rows)


# ── cli ──────────────────────────────────────────────────────────────────────
def _preview(rows: list[dict[str, Any]], n: int = 5) -> None:
    print(f"\nFlattened {len(rows)} recall rows. First {min(n, len(rows))}:")
    for row in rows[:n]:
        print("  " + json.dumps(row, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest openFDA drug recalls into Postgres.")
    parser.add_argument("--search", default=DEFAULT_SEARCH, help='openFDA search filter (default: product_type:"Drugs")')
    parser.add_argument("--limit", type=int, default=None, help="max records to fetch (default: settings cap)")
    parser.add_argument("--apply-schema", action="store_true", help="run schema.sql before loading")
    parser.add_argument("--dry-run", action="store_true", help="fetch + flatten + preview, no DB writes")
    args = parser.parse_args()

    records = fetch_records(search=args.search, max_records=args.limit)
    rows = flatten_all(records)
    print(f"Fetched {len(records)} recall records from openFDA.")

    if args.dry_run:
        _preview(rows)
        print("\n[dry-run] no database writes performed.")
        return

    if args.apply_schema:
        apply_schema()
        print("Applied schema.sql.")

    written = upsert_rows(rows)
    print(f"Upserted {written} rows into recalls.")


if __name__ == "__main__":
    main()
