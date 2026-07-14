"""
Day 1 ingestion — openFDA drugsfda -> Postgres `approvals` table.

Pulls FDA drug application data from openFDA, flattens each application into
product-level rows (one row per product), derives the original approval date
from the ORIG/AP submission, and upserts into `approvals`.

Design notes
------------
* Capped by default (settings.openfda_max_records) so the Day 1 baseline stays
  fast. Raise the cap or pass --limit once the pipeline is proven.
* Idempotent: re-running upserts on the (application_number, product_number)
  primary key, so it is safe to run repeatedly.
* Governance: ingestion connects with the OWNER role (write). The application's
  query path uses a separate READ-ONLY role — never this script's connection.
* psycopg is imported lazily inside the DB functions so --dry-run works with no
  driver installed and no database reachable.

Usage
-----
    # Fetch + flatten + preview, no DB writes (works offline-ish, needs network):
    python -m src.sql.ingest_approvals --dry-run --limit 20

    # Create the table then load the Day 1 capped set:
    python -m src.sql.ingest_approvals --apply-schema

    # Load only brand names starting with A:
    python -m src.sql.ingest_approvals --search 'products.brand_name:A*'
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

DRUGSFDA_ENDPOINT = "/drug/drugsfda.json"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# macOS python.org builds don't use the system keychain, so urllib has no CA
# bundle by default. Point at certifi's bundle when available.
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
    return f"{settings.openfda_base_url}{DRUGSFDA_ENDPOINT}?{urllib.parse.urlencode(params)}"


def fetch_records(
    search: str | None = None,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Page through openFDA until max_records collected (or data exhausted)."""
    cap = max_records or settings.openfda_max_records
    collected: list[dict[str, Any]] = []
    skip = 0
    while len(collected) < cap:
        page = min(settings.openfda_page_size, cap - len(collected))
        url = _build_url(skip, page, search)
        with urllib.request.urlopen(url, timeout=30, context=_SSL_CONTEXT) as resp:  # noqa: S310 (trusted host)
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


def _original_approval_date(submissions: Iterable[dict[str, Any]] | None) -> date | None:
    """Earliest ORIG submission that was approved (AP)."""
    dates = [
        d
        for s in (submissions or [])
        if s.get("submission_type") == "ORIG" and s.get("submission_status") == "AP"
        if (d := _parse_fda_date(s.get("submission_status_date")))
    ]
    return min(dates) if dates else None


def flatten_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One openFDA application -> one row per product."""
    app_no = result.get("application_number")
    if not app_no:
        return []
    sponsor = result.get("sponsor_name")
    approval_date = _original_approval_date(result.get("submissions"))

    rows: list[dict[str, Any]] = []
    for product in result.get("products", []) or []:
        product_number = product.get("product_number")
        if not product_number:
            continue
        ingredients = product.get("active_ingredients") or []
        ingredient = ", ".join(i.get("name", "") for i in ingredients if i.get("name")) or None
        rows.append(
            {
                "application_number": app_no,
                "product_number": product_number,
                "brand_name": product.get("brand_name"),
                "active_ingredient": ingredient,
                "dosage_form": product.get("dosage_form"),
                "route": product.get("route"),
                "marketing_status": product.get("marketing_status"),
                "sponsor": sponsor,
                "approval_date": approval_date,
            }
        )
    return rows


def flatten_all(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten and de-duplicate on the (application_number, product_number) PK."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        for row in flatten_result(result):
            seen[(row["application_number"], row["product_number"])] = row
    return list(seen.values())


# ── database ─────────────────────────────────────────────────────────────────
_UPSERT_SQL = """
INSERT INTO approvals (
    application_number, product_number, brand_name, active_ingredient,
    dosage_form, route, marketing_status, sponsor, approval_date
) VALUES (
    %(application_number)s, %(product_number)s, %(brand_name)s, %(active_ingredient)s,
    %(dosage_form)s, %(route)s, %(marketing_status)s, %(sponsor)s, %(approval_date)s
)
ON CONFLICT (application_number, product_number) DO UPDATE SET
    brand_name        = EXCLUDED.brand_name,
    active_ingredient = EXCLUDED.active_ingredient,
    dosage_form       = EXCLUDED.dosage_form,
    route             = EXCLUDED.route,
    marketing_status  = EXCLUDED.marketing_status,
    sponsor           = EXCLUDED.sponsor,
    approval_date     = EXCLUDED.approval_date;
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
    print(f"\nFlattened {len(rows)} product rows. First {min(n, len(rows))}:")
    for row in rows[:n]:
        print("  " + json.dumps(row, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest openFDA drug approvals into Postgres.")
    parser.add_argument("--search", default=None, help="openFDA search filter, e.g. 'products.brand_name:A*'")
    parser.add_argument("--limit", type=int, default=None, help="max applications to fetch (default: settings cap)")
    parser.add_argument("--apply-schema", action="store_true", help="run schema.sql before loading")
    parser.add_argument("--dry-run", action="store_true", help="fetch + flatten + preview, no DB writes")
    args = parser.parse_args()

    results = fetch_records(search=args.search, max_records=args.limit)
    rows = flatten_all(results)
    print(f"Fetched {len(results)} applications from openFDA.")

    if args.dry_run:
        _preview(rows)
        print("\n[dry-run] no database writes performed.")
        return

    if args.apply_schema:
        apply_schema()
        print("Applied schema.sql.")

    written = upsert_rows(rows)
    print(f"Upserted {written} rows into approvals.")


if __name__ == "__main__":
    main()
