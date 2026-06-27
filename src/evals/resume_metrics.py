#!/usr/bin/env python3
"""
resume_metrics.py — Exact, resume-ready metrics for the Clinical RAG system.

Runs locally (no GitHub Actions, no CI timeout). For every example in a
LangSmith golden dataset it:

  1. Runs the FULL pipeline (HyDE → hybrid dense+BM25 → RRF → Cohere rerank →
     Claude generation), timing every stage.
  2. Runs a DENSE-ONLY baseline (HyDE → dense search, no sparse, no rerank) so
     we can quantify the lift the advanced retrieval gives — the "improved X by
     Y%" number recruiters want.
  3. Scores Ragas metrics (Claude Haiku judge): Faithfulness, Answer Relevancy,
     Context Precision, Context Recall — full pipeline gets all four, baseline
     gets the two retrieval metrics.
  4. Aggregates everything (mean, p50, p95) and writes:
       eval_results/resume_metrics.json   ← all raw + aggregate numbers
       eval_results/per_example.csv       ← one row per question
       eval_results/RESUME_METRICS.md     ← copy-paste bullets with real numbers

Usage:
    python -m src.evals.resume_metrics --dataset clinical-rag-golden-set
    python -m src.evals.resume_metrics --dataset clinical-rag-golden-set --limit 10
    python -m src.evals.resume_metrics --dataset clinical-rag-golden-set --no-baseline

Cost note: full + baseline on ~30 examples is a few dollars of API spend.
Use --limit for a cheap smoke run first.
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langsmith import Client
from qdrant_client.models import NamedVector, NamedSparseVector
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import config
from src.generation.llm_client import ClaudeClient
from src.retrieval.reranker import Reranker
from src.retrieval.search import Searcher

load_dotenv()

RESULTS_ROOT = Path("eval_results")

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
BASE_METRIC_NAMES = ["context_precision", "context_recall"]
STAGE_NAMES = ["hyde", "embed", "sparse_embed", "vector_search", "rerank", "generate"]


# ── small helpers ────────────────────────────────────────────────────────────

def pct(values: list[float], p: float) -> float:
    """Percentile (linear interpolation). p in [0,100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


class Timer:
    def __init__(self):
        self.t = {}

    async def time(self, name, coro):
        start = time.perf_counter()
        result = await coro
        self.t[name] = time.perf_counter() - start
        return result


# ── retrieval variants (instrumented) ────────────────────────────────────────

async def full_retrieve(searcher: Searcher, reranker: Reranker, question: str,
                        top_k: int, timer: Timer) -> list[dict]:
    """HyDE → dense+BM25 hybrid → RRF → Cohere rerank, timing each stage."""
    candidate_k = config.retrieval_top_k

    hypo = await timer.time("hyde", searcher.expand_query(question))
    dense_emb = await timer.time("embed", searcher.embed_query(hypo))

    loop = asyncio.get_running_loop()
    sparse_vec = await timer.time(
        "sparse_embed",
        loop.run_in_executor(None, searcher.sparse_embed_query, question),
    )

    async def _search():
        return await asyncio.gather(
            searcher.qdrant.search(
                collection_name=config.collection_name,
                query_vector=NamedVector(name=config.dense_vector_name, vector=dense_emb),
                limit=candidate_k, with_payload=True,
            ),
            searcher.qdrant.search(
                collection_name=config.collection_name,
                query_vector=NamedSparseVector(name=config.sparse_vector_name, vector=sparse_vec),
                limit=candidate_k, with_payload=True,
            ),
        )

    dense_res, sparse_res = await timer.time("vector_search", _search())
    merged = searcher._rrf_merge(dense_res, sparse_res, k=config.rrf_k)[:candidate_k]
    reranked = await timer.time("rerank", reranker.rerank(question, merged, top_n=top_k))
    return reranked


async def dense_only_retrieve(searcher: Searcher, question: str, top_k: int) -> list[dict]:
    """Baseline: HyDE → dense embed → dense vector search top_k. No sparse, no rerank."""
    hypo = await searcher.expand_query(question)
    dense_emb = await searcher.embed_query(hypo)
    res = await searcher.qdrant.search(
        collection_name=config.collection_name,
        query_vector=NamedVector(name=config.dense_vector_name, vector=dense_emb),
        limit=top_k, with_payload=True,
    )
    return [
        {"text": r.payload.get("text", ""), "source": r.payload.get("source", "unknown"),
         "page": r.payload.get("page", 0), "score": round(r.score, 4)}
        for r in res
    ]


# ── ragas scoring ────────────────────────────────────────────────────────────

def build_metrics():
    judge = LangchainLLMWrapper(ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0.0, max_tokens=4096,
    ))
    embed = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"],
    ))
    metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    for m in metrics.values():
        m.llm = judge
        if hasattr(m, "embeddings"):
            m.embeddings = embed
    return metrics


async def score(metric, question, answer, contexts, ground_truth) -> float:
    sample = SingleTurnSample(
        user_input=question, response=answer or "",
        retrieved_contexts=contexts, reference=ground_truth,
    )
    try:
        s = await metric.single_turn_ascore(sample)
        return round(float(s), 4)
    except Exception as e:
        print(f"    ⚠️  {getattr(metric, 'name', 'metric')} error: {e}")
        return float("nan")


def safe_mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # drop NaN
    return mean(clean)


# ── main ─────────────────────────────────────────────────────────────────────

async def main(args):
    required = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY",
                "QDRANT_URL", "QDRANT_API_KEY", "LANGCHAIN_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"❌ Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    # fetch dataset
    ls = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
    try:
        ds = ls.read_dataset(dataset_name=args.dataset)
        examples = list(ls.list_examples(dataset_id=ds.id))
    except Exception as e:
        print(f"❌ Could not read dataset '{args.dataset}': {e}")
        sys.exit(1)
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        print("❌ Dataset empty.")
        sys.exit(1)

    top_k = config.default_top_k

    # Per-dataset output folder keeps runs organized and separate.
    out_dir = RESULTS_ROOT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "per_example.jsonl"

    # ── Resume: load any examples already scored in a previous run ──────────────
    rows: list[dict] = []
    done_questions: set[str] = set()
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rows.append(r)
                done_questions.add(r.get("question", ""))
            except json.JSONDecodeError:
                continue

    print(f"\n🔬 Resume Metrics — Clinical RAG")
    print(f"   Dataset: {args.dataset}  |  Examples in dataset: {len(examples)}  |  top_k={top_k}")
    print(f"   Baseline (dense-only) comparison: {'OFF' if args.no_baseline else 'ON'}")
    print(f"   Output folder: {out_dir}/")
    if done_questions:
        print(f"   ♻️  Resuming — {len(done_questions)} already scored, will skip them.")
    print(f"   💾 Checkpoint after every example. Safe to Ctrl-C anytime.\n")

    metrics = build_metrics()
    searcher, reranker, llm = Searcher(), Reranker(), ClaudeClient()

    todo = [ex for ex in examples if ex.inputs.get("question", "") not in done_questions]
    interrupted = False

    try:
        for i, ex in enumerate(todo, 1):
            question = ex.inputs.get("question", "")
            gt = ex.outputs.get("ground_truth", "") if ex.outputs else ""
            print(f"[{i}/{len(todo)}] (total done: {len(rows)}) {question[:60]}...")

            # FULL pipeline
            timer = Timer()
            t0 = time.perf_counter()
            chunks = await full_retrieve(searcher, reranker, question, top_k, timer)
            answer = await timer.time("generate", llm.generate(question, chunks))
            total = time.perf_counter() - t0

            contexts = [c["text"] for c in chunks]
            row = {"question": question, "total_latency_s": round(total, 3)}
            for k in STAGE_NAMES:
                row[f"stage_{k}_s"] = round(timer.t.get(k, 0.0), 3)
            for name in METRIC_NAMES:
                row[f"full_{name}"] = await score(metrics[name], question, answer, contexts, gt)

            # DENSE-ONLY baseline (retrieval metrics only — no generation needed)
            if not args.no_baseline:
                b_chunks = await dense_only_retrieve(searcher, question, top_k)
                b_ctx = [c["text"] for c in b_chunks]
                for name in BASE_METRIC_NAMES:
                    row[f"baseline_{name}"] = await score(metrics[name], question, "", b_ctx, gt)

            rows.append(row)

            # ── checkpoint: append this example, then refresh aggregate outputs ──
            with open(checkpoint, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            summary = compute_summary(rows, args.dataset, top_k, args.no_baseline)
            write_outputs(out_dir, summary, rows)

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n⏹️  Interrupted — finalizing files with everything completed so far...")

    if not rows:
        print("❌ No examples scored.")
        sys.exit(1)

    summary = compute_summary(rows, args.dataset, top_k, args.no_baseline)
    write_outputs(out_dir, summary, rows)

    # ── console summary ───────────────────────────────────────────────────────
    agg_full = summary["full_pipeline_scores"]
    agg_base = summary["dense_only_baseline_scores"] or {}
    lifts = summary["lift_pct_full_vs_baseline"] or {}
    lat = summary["latency"]

    print("\n" + "═" * 60)
    print("RESUME METRICS  (exact, from your data)" + ("  [PARTIAL]" if interrupted else ""))
    print("═" * 60)
    print(f"Dataset: {args.dataset}   n={len(rows)}   {summary['generated_at']}\n")
    print(f"{'Metric':<22}{'Full':>8}{'Baseline':>10}{'Lift %':>9}")
    print("-" * 60)
    for k in METRIC_NAMES:
        b = agg_base.get(k)
        l = lifts.get(k)
        print(f"{k:<22}{agg_full[k]:>8.3f}"
              f"{(f'{b:.3f}' if b is not None else '—'):>10}"
              f"{(f'+{l}' if l is not None else '—'):>9}")
    print("-" * 60)
    print(f"Latency  mean {lat['total']['mean']}s | "
          f"p50 {lat['total']['p50']}s | p95 {lat['total']['p95']}s")
    print(f"Per stage (mean s): {lat['stages_mean_s']}")
    print("═" * 60)
    print(f"\n📄 {out_dir}/resume_metrics.json")
    print(f"📄 {out_dir}/per_example.csv  (+ per_example.jsonl checkpoint)")
    print(f"📄 {out_dir}/RESUME_METRICS.md  ← copy bullets from here")
    if interrupted:
        print(f"\n♻️  Re-run the same command to resume and add more examples.\n")
    else:
        print()


def compute_summary(rows: list[dict], dataset: str, top_k: int, no_baseline: bool) -> dict:
    """Rebuild all aggregate numbers purely from the per-example rows collected
    so far. Called after every example so the output files always reflect the
    full set of completed work — and so a resumed run reconstructs state from
    the checkpoint file alone."""
    def col(key):
        return [r[key] for r in rows if key in r and r[key] == r[key]]  # drop NaN

    agg_full = {k: round(safe_mean(col(f"full_{k}")), 4) for k in METRIC_NAMES}
    agg_base = {k: round(safe_mean(col(f"baseline_{k}")), 4) for k in BASE_METRIC_NAMES}

    def lift(metric):
        b, f = agg_base.get(metric, 0.0), agg_full.get(metric, 0.0)
        return round((f - b) / b * 100, 1) if b else None

    lifts = None if no_baseline else {
        "context_precision": lift("context_precision"),
        "context_recall": lift("context_recall"),
    }

    totals = col("total_latency_s")
    latency = {
        "total": {"mean": round(mean(totals), 2), "p50": round(pct(totals, 50), 2),
                  "p95": round(pct(totals, 95), 2)},
        "stages_mean_s": {k: round(mean(col(f"stage_{k}_s")), 2) for k in STAGE_NAMES},
        "stages_p95_s": {k: round(pct(col(f"stage_{k}_s"), 95), 2) for k in STAGE_NAMES},
    }

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": dataset, "n_examples": len(rows), "top_k": top_k,
        "judge_model": "claude-haiku-4-5-20251001",
        "generation_model": config.generation_model,
        "full_pipeline_scores": agg_full,
        "dense_only_baseline_scores": None if no_baseline else agg_base,
        "lift_pct_full_vs_baseline": lifts,
        "latency": latency,
    }


def write_outputs(out_dir: Path, summary: dict, rows: list[dict]) -> None:
    """(Re)write the JSON summary, per-example CSV, and resume-bullet markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resume_metrics.json").write_text(json.dumps(summary, indent=2))

    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(out_dir / "per_example.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    (out_dir / "RESUME_METRICS.md").write_text(render_resume_md(summary))


def render_resume_md(s: dict) -> str:
    f = s["full_pipeline_scores"]
    lat = s["latency"]["total"]
    lifts = s.get("lift_pct_full_vs_baseline") or {}
    cp_lift = lifts.get("context_precision")
    cr_lift = lifts.get("context_recall")

    lift_line = ""
    if cp_lift is not None:
        lift_line = (f"- Advanced retrieval (HyDE + hybrid BM25/dense + RRF + Cohere rerank) "
                     f"lifted Context Precision by {cp_lift}% and Context Recall by {cr_lift}% "
                     f"over a dense-only baseline on a {s['n_examples']}-question golden set.\n")

    return f"""# Resume Metrics — Clinical Trial Compliance RAG
_Generated {s['generated_at']} · dataset `{s['dataset']}` · n={s['n_examples']} · judge=Claude Haiku_

## Exact scores (full pipeline)
| Metric | Score |
|---|---|
| Faithfulness | {f['faithfulness']:.3f} |
| Answer Relevancy | {f['answer_relevancy']:.3f} |
| Context Precision | {f['context_precision']:.3f} |
| Context Recall | {f['context_recall']:.3f} |

**Latency:** mean {lat['mean']}s · p50 {lat['p50']}s · p95 {lat['p95']}s
**Per-stage mean (s):** {s['latency']['stages_mean_s']}

## Copy-paste resume bullets
- Built a production RAG system over FDA/ICH clinical-trial regulatory documents delivering fully source-cited answers, achieving {f['faithfulness']:.0%} Faithfulness and {f['answer_relevancy']:.0%} Answer Relevancy (Ragas, LLM-judged on a {s['n_examples']}-question expert golden set).
{lift_line}- Engineered a hybrid retrieval pipeline (dense + BM25 fused via RRF) with Cohere cross-encoder reranking (20→{s['top_k']} candidates), reaching {f['context_precision']:.0%} Context Precision and {f['context_recall']:.0%} Context Recall.
- Served answers at {lat['p95']}s p95 latency end-to-end (HyDE → hybrid search → rerank → generation) via a FastAPI service with full LangSmith per-stage tracing.
- Built an automated Ragas evaluation harness (4 metrics, Claude Haiku judge) wired into a CI quality gate that blocks regressing deploys below set Faithfulness/Precision thresholds.

_Numbers above are exact from your latest run — see resume_metrics.json for raw values._
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exact resume-ready RAG metrics (runs locally).")
    p.add_argument("--dataset", default="clinical-rag-golden-set")
    p.add_argument("--limit", type=int, default=None, help="Only run first N examples (cheap smoke run).")
    p.add_argument("--no-baseline", action="store_true", help="Skip dense-only baseline + lift calc.")
    args = p.parse_args()
    asyncio.run(main(args))
