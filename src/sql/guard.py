"""
SQL governance guard for the text-to-SQL path.

Every LLM-generated query passes through validate() BEFORE it touches the
database. This is the AST-level half of defense in depth; the other half is the
read-only Postgres role (which cannot write even if this guard were bypassed).

Checks enforced
---------------
1. Parses as exactly ONE statement (blocks stacked `...; DROP TABLE ...`).
2. The statement is a read query only: SELECT / WITH / set-operation. Any
   INSERT/UPDATE/DELETE/DDL/`SELECT ... INTO`/command is rejected.
3. Every referenced table is in the allow-list (settings.allowed_tables).
4. A row cap is injected: missing LIMIT gets one, oversized LIMIT is clamped.

validate() returns the secured SQL string (row cap applied) or raises
GuardRejection with a reason.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from src.sql.settings import settings

DIALECT = "postgres"

# Expression types that indicate a write / DDL / side effect. Presence anywhere
# in the tree is an immediate rejection.
_FORBIDDEN = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,   # catches DDL/utility statements sqlglot maps to Command (GRANT, COPY, etc.)
    exp.Into,      # SELECT ... INTO new_table
)

# Allowed top-level statement types (read-only shapes).
_ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


class GuardRejection(ValueError):
    """Raised when a query violates a governance rule."""


def validate(sql: str) -> str:
    """Validate and secure a generated SQL string. Returns secured SQL or raises."""
    if not sql or not sql.strip():
        raise GuardRejection("empty query")

    try:
        statements = sqlglot.parse(sql, read=DIALECT)
    except Exception as e:  # sqlglot.errors.ParseError and friends
        raise GuardRejection(f"unparseable SQL: {e}") from e

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise GuardRejection(f"exactly one statement required, found {len(statements)}")

    stmt = statements[0]

    if not isinstance(stmt, _ALLOWED_ROOTS):
        raise GuardRejection(f"only read queries allowed, got {stmt.key.upper()}")

    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN):
            raise GuardRejection(f"forbidden operation: {type(node).__name__}")

    _check_tables(stmt)

    return _apply_row_cap(stmt)


# ── internals ────────────────────────────────────────────────────────────────
def _check_tables(stmt: exp.Expression) -> None:
    allowed = {t.lower() for t in settings.allowed_tables}
    # Names introduced by CTEs are valid internal references, not real tables.
    cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        name = table.name.lower()
        if name in cte_names:
            continue
        if name not in allowed:
            raise GuardRejection(f"table not allowed: {table.name}")


def _apply_row_cap(stmt: exp.Expression) -> str:
    cap = settings.sql_row_cap
    limit = stmt.args.get("limit")
    if limit is None:
        stmt = stmt.limit(cap)
    else:
        try:
            current = int(limit.expression.name)
        except (AttributeError, ValueError):
            current = None
        if current is None or current > cap:
            stmt = stmt.limit(cap)
    return stmt.sql(dialect=DIALECT)
