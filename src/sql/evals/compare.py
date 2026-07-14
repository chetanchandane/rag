"""
Execution-accuracy comparison for the SQL eval.

A generated query is "correct" if its result set matches the gold query's
result set. This is robust to phrasing differences (two different SQL strings
that return the same rows both count as correct).

Matching rules, in order:
1. Scalar case: gold returns a single row / single column (e.g. COUNT(*)).
   Compare the lone value, ignoring column name.
2. Name-aware projection: if every generated row contains all of gold's column
   names, compare the multiset of rows projected onto gold's columns. This
   tolerates the generator selecting extra columns.
3. Fallback: compare the full value-multisets ignoring column names.

All comparisons are order-insensitive (multiset) and value-normalized.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

Row = dict[str, Any]


def _norm(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        f = float(value)
        return int(f) if f.is_integer() else f
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return value.strip()
    return value


def _multiset(rows: list[Row], columns: list[str] | None) -> list[tuple]:
    out: list[tuple] = []
    for row in rows:
        if columns is None:
            # Fallback: compare VALUES only, ignoring column names/order (execution
            # accuracy ignores how columns are labelled).
            out.append(tuple(sorted((repr(_norm(v)) for v in row.values()))))
        else:
            out.append(tuple(_norm(row.get(c)) for c in columns))
    return sorted(out, key=repr)


def results_match(gold_rows: list[Row], gen_rows: list[Row]) -> bool:
    """True if the generated result set matches the gold result set."""
    if not gold_rows:
        return len(gen_rows) == 0
    gold_cols = list(gold_rows[0].keys())

    # 1) scalar
    if len(gold_cols) == 1 and len(gold_rows) == 1:
        if len(gen_rows) != 1:
            return False
        gold_val = _norm(next(iter(gold_rows[0].values())))
        gen_val = _norm(next(iter(gen_rows[0].values())))
        return gold_val == gen_val

    # 2) name-aware projection
    if gen_rows and all(all(c in r for c in gold_cols) for r in gen_rows):
        return _multiset(gold_rows, gold_cols) == _multiset(gen_rows, gold_cols)

    # 3) fallback: full value-multiset
    return _multiset(gold_rows, None) == _multiset(gen_rows, None)
