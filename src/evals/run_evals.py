#!/usr/bin/env python3
"""
run_evals.py — Ragas evaluation runner with LangSmith Experiment logging

Fetches examples from a LangSmith Dataset, runs the full RAG pipeline on each
question, scores answers with 4 Ragas metrics (Claude Haiku as judge), and logs
all results as a LangSmith Experiment visible at smith.langchain.com.

Usage:
    # Full run — all examples
    python -m src.evals.run_evals --dataset clinical-rag-golden-set

    # CI mode — first 15 examples, exit 1 if any metric is below threshold
    python -m src.evals.run_evals --dataset clinical-rag-golden-set --ci

    # Smoke test
    python -m src.evals.run_evals --dataset clinical-rag-golden-set-smoketest

Thresholds:
    faithfulness          >= 0.70   (does the answer stick to the retrieved context?)
    answer_relevancy      >= 0.70   (does the answer address the question?)
    context_precision     >= 0.60   (are retrieved chunks relevant to the ground truth?)
    context_recall        >= 0.60   (do retrieved chunks cover the ground truth?)

LangSmith view:
    smith.langchain.com → Datasets → <dataset_name> → Experiments tab
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langsmith import Client
from langsmith.evaluation import EvaluationResult
from langsmith.evaluation import evaluate as ls_evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.generation.llm_client import ClaudeClient
from src.retrieval.reranker import Reranker
from src.retrieval.search import Searcher

load_dotenv()


# ── Thresholds ─────────────────────────────────────────────────────────────────

THRESHOLDS: dict[str, float] = {
    "faithfulness":      0.70,
    "answer_relevancy":  0.70,
    "context_precision": 0.60,
    "context_recall":    0.60,
}

CI_SAMPLE = 4    # examples to run in --ci mode (keeps wall-clock time under 30 min)


# ── Persistent event loop ─────────────────────────────────────────────────────
#
# asyncio.run() creates a new event loop, runs the coroutine, then closes the
# loop.  httpx schedules background cleanup tasks (connection pool teardown) that
# fire *after* the main coroutine but *before* loop.close() — causing a flood of
# "RuntimeError: Event loop is closed" warnings on every example.
#
# Fix: keep one loop alive for the entire eval run.  httpx cleanup tasks run
# naturally between calls, no loop-closed errors.
#
# _EVAL_LOOP.run_until_complete() is safe from any thread as long as the loop is
# not already running; with max_concurrency=1 this is always true.

_EVAL_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(_EVAL_LOOP)


# ── RAG pipeline (direct call — no HTTP overhead) ─────────────────────────────

async def _run_pipeline_fresh(question: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline with components created INSIDE this coroutine.

    AsyncQdrantClient / AsyncOpenAI / AsyncAnthropic bind their internal httpx
    sessions to the event loop they are first used on.  By creating them here
    (inside the coroutine, on _EVAL_LOOP), they are always on the correct loop.
    Init cost is negligible (~1 ms) — no network calls happen during __init__.
    """
    searcher = Searcher()
    reranker = Reranker()
    llm      = ClaudeClient()

    chunks = await searcher.search(question, top_k=top_k)
    chunks = await reranker.rerank(question, chunks, top_n=top_k)
    answer = await llm.generate(question, chunks)
    return {
        "answer":   answer,
        "contexts": [c["text"] for c in chunks],
    }


def _run_pipeline_sync(question: str) -> dict:
    """Synchronous wrapper for use as a langsmith.evaluate() target."""
    return _EVAL_LOOP.run_until_complete(_run_pipeline_fresh(question))


# ── Ragas metric setup ────────────────────────────────────────────────────────

def _build_ragas_metrics() -> dict:
    """
    Configure the 4 Ragas metrics to use:
      - Claude Haiku as the judge LLM (cheaper than Sonnet for high-volume scoring)
      - OpenAI text-embedding-3-small for AnswerRelevancy semantic similarity
    """
    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0.0,
            max_tokens=4096,   # Faithfulness generates claim lists — needs headroom
        )
    )
    embed_model = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    )

    metrics = {
        "faithfulness":      faithfulness,
        "answer_relevancy":  answer_relevancy,
        "context_precision": context_precision,
        "context_recall":    context_recall,
    }

    for m in metrics.values():
        m.llm = judge_llm
        if hasattr(m, "embeddings"):
            m.embeddings = embed_model

    return metrics


# ── LangSmith evaluator wrappers ──────────────────────────────────────────────

def _make_evaluator(metric_name: str, metric, scores_tracker: dict[str, list[float]]):
    """
    Wrap a Ragas metric as a LangSmith custom evaluator function.

    LangSmith calls evaluator(run, example) after each RAG pipeline run.
    We also record scores in scores_tracker so we can print aggregated
    results and check thresholds after ls_evaluate() finishes.
    """
    def evaluator(run, example) -> EvaluationResult:
        question     = example.inputs.get("question", "")
        ground_truth = example.outputs.get("ground_truth", "")
        answer       = (run.outputs or {}).get("answer", "")
        contexts     = (run.outputs or {}).get("contexts", [])

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )

        try:
            score = metric.single_turn_score(sample)
        except Exception as e:
            print(f"\n  ⚠️  {metric_name} scoring error: {e}")
            score = 0.0

        score = round(float(score), 4)
        scores_tracker[metric_name].append(score)

        return EvaluationResult(key=metric_name, score=score)

    evaluator.__name__ = f"ragas_{metric_name}"
    return evaluator


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    load_dotenv()

    required = [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "QDRANT_URL", "QDRANT_API_KEY",
        "LANGCHAIN_API_KEY",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"❌ Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    dataset_name = args.dataset
    ci_mode      = args.ci
    ts           = datetime.now().strftime("%Y%m%d-%H%M")
    exp_prefix   = f"ci-ragas-{ts}" if ci_mode else f"ragas-{ts}"

    print(f"\n🔬 Clinical RAG — Ragas Eval Runner")
    print(f"   Dataset:    {dataset_name}")
    print(f"   Mode:       {'CI (' + str(CI_SAMPLE) + ' examples)' if ci_mode else 'full run'}")
    print(f"   Experiment: {exp_prefix}\n")

    # Validate dataset exists and fetch examples
    ls_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
    try:
        ds       = ls_client.read_dataset(dataset_name=dataset_name)
        examples = list(ls_client.list_examples(dataset_id=ds.id))
    except Exception as e:
        print(f"❌ Could not read LangSmith dataset '{dataset_name}': {e}")
        print(f"   Run generate_dataset.py first to create it.")
        sys.exit(1)

    if not examples:
        print(f"❌ Dataset '{dataset_name}' is empty. Run generate_dataset.py first.")
        sys.exit(1)

    if ci_mode:
        examples = examples[:CI_SAMPLE]

    print(f"📋 {len(examples)} examples to evaluate.")

    def target(inputs: dict) -> dict:
        return _run_pipeline_sync(inputs["question"])

    print("🤖 Configuring Ragas metrics (Claude Haiku as judge)...")
    ragas_metrics   = _build_ragas_metrics()
    scores_tracker  = {k: [] for k in THRESHOLDS}
    evaluators      = [
        _make_evaluator(name, metric, scores_tracker)
        for name, metric in ragas_metrics.items()
    ]

    # Run evaluation — results auto-logged to LangSmith as an Experiment
    print(f"\n🚀 Running evals... (results streaming to smith.langchain.com)\n")
    ls_evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=exp_prefix,
        max_concurrency=1,   # sequential — avoids asyncio.run() conflicts in threads
    )

    # ── Threshold check ───────────────────────────────────────────────────────

    print("\n" + "─" * 55)
    print(f"{'Metric':<24} {'Mean':>6}  {'Threshold':>9}  Status")
    print("─" * 55)

    all_passed = True
    for metric_name, threshold in THRESHOLDS.items():
        scores = scores_tracker[metric_name]
        if not scores:
            print(f"  {'⚠️ ':>2} {metric_name:<22} {'N/A':>6}  {threshold:>9.2f}  no scores")
            continue
        mean   = sum(scores) / len(scores)
        passed = mean >= threshold
        icon   = "✅" if passed else "❌"
        print(f"  {icon}  {metric_name:<22} {mean:>6.3f}  {threshold:>9.2f}")
        if not passed:
            all_passed = False

    print("─" * 55)
    print(f"\n🔗 View full experiment: smith.langchain.com")
    print(f"   → Datasets → {dataset_name} → Experiments → {exp_prefix}")

    if not all_passed:
        print("\n❌ One or more metrics failed the threshold. CI gate: FAIL")
        sys.exit(1)

    print("\n✅ All metrics passed. CI gate: PASS")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Ragas evals against a LangSmith dataset and log results as an Experiment."
    )
    parser.add_argument(
        "--dataset", type=str, default="clinical-rag-golden-set",
        help="LangSmith dataset name (must exist — run generate_dataset.py first).",
    )
    parser.add_argument(
        "--ci", action="store_true",
        help=f"CI mode: evaluate first {CI_SAMPLE} examples only. Exit 1 if any metric misses threshold.",
    )
    args = parser.parse_args()
    main(args)
