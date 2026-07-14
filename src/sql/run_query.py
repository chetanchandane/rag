"""
Guarded execution for the text-to-SQL path.

execute() is the only sanctioned way to run generated SQL:

    guard.validate  ->  EXPLAIN dry-run  ->  run on the READ-ONLY connection

Layers of defense:
* AST guard (single read-only statement, allow-listed tables, row cap) runs
  BEFORE any database connection is opened.
* EXPLAIN dry-run plans the query (catches bad columns/types) without executing.
* The connection uses the read-only role and is pinned read-only at the session
  level, so writes are impossible even if the guard were bypassed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.sql.guard import validate
from src.sql.settings import settings


@dataclass
class QueryResult:
    sql: str                        # the secured SQL actually executed
    columns: list[str]
    rows: list[dict[str, Any]]

    @property
    def row_count(self) -> int:
        return len(self.rows)


def execute(sql_text: str) -> QueryResult:
    """Guard-validate, dry-run, and execute a query on the read-only connection."""
    secured = validate(sql_text)  # raises GuardRejection before any DB work

    if not settings.database_url_readonly:
        raise RuntimeError(
            "DATABASE_URL_RO is not set. Create the role first: "
            "python -m src.sql.setup_readonly_role"
        )

    import psycopg  # lazy import

    with psycopg.connect(settings.database_url_readonly) as conn:
        conn.read_only = True  # session-level read-only (belt and suspenders)
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN {secured}")  # dry-run: plan only, no execution
            cur.execute(secured)
            columns = [d.name for d in cur.description] if cur.description else []
            rows = [dict(zip(columns, r)) for r in cur.fetchall()]

    return QueryResult(sql=secured, columns=columns, rows=rows)


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "SELECT brand_name, marketing_status FROM approvals LIMIT 5"
    result = execute(query)
    print(f"SQL: {result.sql}")
    print(f"{result.row_count} rows")
    for row in result.rows:
        print(" ", row)
