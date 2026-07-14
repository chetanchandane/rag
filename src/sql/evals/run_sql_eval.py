"""
Day 1 text-to-SQL baseline eval.

For each question: generate SQL -> guard -> execute (read-only), then compare the
result set to a hand-written gold query's result set (execution accuracy).

Reports:
* accuracy         — fraction whose results match gold
* executable_rate  — fraction that generated + ran without error (any result)
* per-category breakdown

Saves a JSON report under eval_results/sql_router/ for the README progression table.

Usage
-----
    python -m src.sql.evals.run_sql_eval
    python -m src.sql.evals.run_sql_eval --limit 5        # quick subset
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.sql.evals.compare import results_match
from src.sql.generate_sql import generate_sql
from src.sql.run_query import execute
from src.sql.settings import settings

QUESTIONS_PATH = Path(__file__).with_name("day1_questions.jsonl")
RESULTS_DIR = Path(__file__).resolve().parents[3] / "eval_results" / "sql_router"


def _load_questions(limit: int | None) -> list[dict]:
    items = [json.loads(line) for line in QUESTIONS_PATH.read_text().splitlines() if line.strip()]
    return items[:limit] if limit else items


def _evaluate_one(item: dict) -> dict:
    out = {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "gold_sql": item["gold_sql"],
        "generated_sql": None,
        "secured_sql": None,
        "generated_ok": False,
        "correct": False,
        "error": None,
    }
    try:
        gold = execute(item["gold_sql"])
    except Exception as e:  # a broken gold query is an authoring bug, surface it
        out["error"] = f"GOLD FAILED: {e}"
        return out

    try:
        generated = generate_sql(item["question"])
        out["generated_sql"] = generated.sql
        result = execute(generated.sql)
        out["secured_sql"] = result.sql
        out["generated_ok"] = True
        out["correct"] = results_match(gold.rows, result.rows)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 1 text-to-SQL baseline eval.")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--out", default=None, help="output JSON path (default: eval_results/sql_router/...)")
    args = parser.parse_args()

    questions = _load_questions(args.limit)
    print(f"Running {len(questions)} questions against model {settings.generation_model}\n")

    results = []
    start = time.time()
    for item in questions:
        r = _evaluate_one(item)
        results.append(r)
        mark = "OK " if r["correct"] else ("gen" if r["generated_ok"] else "ERR")
        note = "" if r["correct"] else f"   <- {r['error'] or 'result mismatch'}"
        print(f"[{mark}] {r['id']:>2}. {r['question'][:60]}{note}")
    elapsed = time.time() - start

    n = len(results)
    correct = sum(r["correct"] for r in results)
    executable = sum(r["generated_ok"] for r in results)
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        by_cat[r["category"]]["n"] += 1
        by_cat[r["category"]]["correct"] += int(r["correct"])

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.generation_model,
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "executable_rate": round(executable / n, 4) if n else 0.0,
        "elapsed_seconds": round(elapsed, 1),
        "by_category": {k: dict(v) for k, v in by_cat.items()},
        "results": results,
    }

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"day1_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 52)
    print(f"  accuracy         {correct}/{n}  =  {report['accuracy']:.1%}")
    print(f"  executable_rate  {executable}/{n}  =  {report['executable_rate']:.1%}")
    for cat, v in sorted(by_cat.items()):
        print(f"    {cat:<12} {v['correct']}/{v['n']}")
    print(f"  saved -> {out_path}")
    print("=" * 52)


if __name__ == "__main__":
    main()
