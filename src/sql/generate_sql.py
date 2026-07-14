"""
Text-to-SQL generation for the relational path.

Turns a natural-language question into a single read-only PostgreSQL query,
using the schema DDL + semantic layer in the prompt and Anthropic tool-use for
structured (Pydantic) output. Generation does NOT trust the model: the returned
SQL still passes through guard.validate before it can execute.

Kept self-contained (reads ANTHROPIC_API_KEY and model from env/settings) so the
SQL package stays importable without the vector-store config.

Usage
-----
    python -m src.sql.generate_sql "How many Class I drug recalls are there?"
    python -m src.sql.generate_sql "Newest 5 approved drugs" --run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anthropic
from langsmith import traceable
from pydantic import BaseModel, Field

from src.sql.settings import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SEMANTIC_PATH = Path(__file__).with_name("semantic.yaml")


class GeneratedSQL(BaseModel):
    """Structured output of the generator."""

    sql: str = Field(description="A single read-only PostgreSQL SELECT query answering the question.")
    rationale: str = Field(default="", description="One sentence: which table was used and why.")


def _system_prompt() -> str:
    schema = SCHEMA_PATH.read_text()
    semantic = SEMANTIC_PATH.read_text()
    return (
        "You translate natural-language questions about FDA drug data into a single "
        "PostgreSQL query.\n\n"
        "Rules — follow exactly:\n"
        "1. Output exactly ONE read-only SELECT statement. Never write or modify data.\n"
        "2. Use ONLY the tables and columns below. Never invent columns.\n"
        "3. The two tables are INDEPENDENT — do NOT join them. Pick the one table that "
        "answers the question.\n"
        "4. Match drug names case-insensitively with ILIKE '%name%' (names are messy free text).\n"
        "5. classification and marketing_status are exact strings; match them exactly.\n"
        "6. Dates are real DATE columns; sort approval_date with NULLS LAST.\n"
        "7. Do not add a LIMIT yourself; one is enforced automatically.\n\n"
        f"=== SCHEMA (DDL) ===\n{schema}\n\n"
        f"=== SEMANTIC LAYER ===\n{semantic}"
    )


@traceable(name="generate_sql", run_type="llm")
def generate_sql(question: str) -> GeneratedSQL:
    """Generate structured SQL for a question (not yet validated or executed)."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tool = {
        "name": "emit_sql",
        "description": "Return the single read-only SQL query that answers the question.",
        "input_schema": GeneratedSQL.model_json_schema(),
    }
    message = client.messages.create(
        model=settings.generation_model,
        max_tokens=settings.generation_max_tokens,
        system=_system_prompt(),
        tools=[tool],
        tool_choice={"type": "tool", "name": "emit_sql"},
        messages=[{"role": "user", "content": question}],
    )
    tool_use = next(block for block in message.content if block.type == "tool_use")
    return GeneratedSQL(**tool_use.input)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SQL from a natural-language question.")
    parser.add_argument("question", nargs="+", help="the question to translate")
    parser.add_argument("--run", action="store_true", help="execute the SQL through the guarded read-only path")
    args = parser.parse_args()

    question = " ".join(args.question)
    generated = generate_sql(question)
    print(f"Question:  {question}")
    print(f"Rationale: {generated.rationale}")
    print(f"SQL:       {generated.sql}")

    if args.run:
        from src.sql.run_query import execute

        result = execute(generated.sql)
        print(f"\nSecured:   {result.sql}")
        print(f"{result.row_count} rows:")
        for row in result.rows:
            print(" ", row)


if __name__ == "__main__":
    main()
