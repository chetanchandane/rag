#!/usr/bin/env python3
"""
generate_dataset.py — Synthetic golden dataset generator for clinical RAG evals

Samples chunks from Qdrant, asks Claude to generate one (question, ground_truth) pair
per chunk, saves results locally as JSONL, and uploads to a LangSmith Dataset.

Usage:
    # Smoke test — 3 chunks, separate dataset
    python -m src.evals.generate_dataset --sample 3 --dataset clinical-rag-golden-set-smoketest

    # Full run — 30 chunks → production eval dataset
    python -m src.evals.generate_dataset --sample 30 --dataset clinical-rag-golden-set

    # Dry run — generate locally, skip LangSmith upload
    python -m src.evals.generate_dataset --sample 5 --dry-run

LangSmith dataset schema:
    inputs:  {"question": str}
    outputs: {"ground_truth": str, "ground_truth_contexts": list[str]}

Cost note: each chunk = 1 Claude call (~300 input tokens + ~150 output tokens).
30 chunks ≈ 30 × claude-sonnet-4-6 calls. Use --sample 3 for smoke testing first.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from src.config import config

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = (
    "You are a clinical regulatory expert. "
    "Given a passage from an FDA or ICH guidance document, generate one realistic "
    "compliance question that a clinical trial professional might ask, along with a "
    "precise, complete answer based ONLY on the provided text.\n\n"
    "The question must be:\n"
    "- Specific and answerable solely from the passage\n"
    "- Phrased as a professional would ask it (not 'what does this say about X')\n"
    "- Focused on regulatory requirements, timelines, definitions, or procedures\n\n"
    "Respond ONLY with valid JSON — no markdown fences, no explanation:\n"
    '{"question": "...", "ground_truth": "..."}'
)


def _multi_qa_prompt(n: int) -> str:
    """System prompt for generating N *distinct* Q&A pairs from one passage."""
    return (
        "You are a clinical regulatory expert. "
        f"Given a passage from an FDA or ICH guidance document, generate {n} DISTINCT, "
        "non-overlapping compliance questions a clinical trial professional might ask, "
        "each with a precise, complete answer based ONLY on the provided text.\n\n"
        "Each question must be:\n"
        "- Specific and answerable solely from the passage\n"
        "- Phrased as a professional would ask it (not 'what does this say about X')\n"
        "- A different angle from the others (e.g. timeline, definition, responsibility, procedure)\n"
        "- Focused on regulatory requirements, timelines, definitions, or procedures\n\n"
        "If the passage genuinely supports fewer than "
        f"{n} distinct questions, return only as many as are well-grounded.\n\n"
        "Respond ONLY with a valid JSON array — no markdown fences, no explanation:\n"
        '[{"question": "...", "ground_truth": "..."}, ...]'
    )

OUTPUT_DIR = Path("tests/evals")


# ── Qdrant sampling ───────────────────────────────────────────────────────────

async def sample_chunks(n: int) -> list[dict]:
    """
    Scroll the entire Qdrant collection and return n evenly-spaced chunks
    so samples span the full corpus rather than just the first page.
    """
    from qdrant_client import AsyncQdrantClient

    qdrant = AsyncQdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
    )

    all_chunks: list[dict] = []
    offset = None

    while True:
        results, next_offset = await qdrant.scroll(
            collection_name=config.collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for hit in results:
            p = hit.payload or {}
            all_chunks.append({
                "text":   p.get("text", ""),
                "source": p.get("source", "unknown"),
                "page":   p.get("page", 0),
            })
        if next_offset is None:
            break
        offset = next_offset

    if not all_chunks:
        raise RuntimeError(
            "No chunks found in Qdrant collection. "
            f"Run ingestion first: python -m src.ingestion.indexer --dir data/raw/"
        )

    print(f"  Found {len(all_chunks)} total chunks in collection '{config.collection_name}'.")

    # Evenly sample n chunks across the corpus
    if n >= len(all_chunks):
        return all_chunks

    indices = [int(i * len(all_chunks) / n) for i in range(n)]
    return [all_chunks[i] for i in indices]


# ── Q&A generation ────────────────────────────────────────────────────────────

async def generate_qa(
    anthropic: AsyncAnthropic, chunk: dict, model: str = config.generation_model
) -> dict | None:
    """
    Ask Claude to generate a (question, ground_truth) pair from one chunk.
    Returns None if the chunk is too short or Claude returns malformed JSON.
    """
    text = chunk["text"].strip()
    if len(text) < 100:
        print(f" ✗ (skipped — too short: {len(text)} chars)")
        return None

    try:
        response = await anthropic.messages.create(
            model=model,
            max_tokens=512,
            system=QA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Passage:\n\n{text}"}],
        )
        raw = response.content[0].text.strip()
        qa  = json.loads(raw)

        if not qa.get("question") or not qa.get("ground_truth"):
            raise ValueError("Missing 'question' or 'ground_truth' key.")

        return {
            "question":             qa["question"],
            "ground_truth":         qa["ground_truth"],
            "ground_truth_contexts": [text],
            "source":               chunk["source"],
            "page":                 chunk["page"],
        }

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f" ✗ (skipped — parse error: {e})")
        return None


async def generate_qa_multi(
    anthropic: AsyncAnthropic, chunk: dict, per_chunk: int,
    model: str = config.generation_model,
) -> list[dict]:
    """
    Ask Claude for `per_chunk` distinct (question, ground_truth) pairs from one
    chunk. Returns a list (possibly shorter than per_chunk, or empty).
    """
    text = chunk["text"].strip()
    if len(text) < 100:
        return []

    try:
        response = await anthropic.messages.create(
            model=model,
            max_tokens=min(512 * per_chunk, 4096),
            system=_multi_qa_prompt(per_chunk),
            messages=[{"role": "user", "content": f"Passage:\n\n{text}"}],
        )
        raw = response.content[0].text.strip()
        # tolerate accidental code fences
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1].lstrip("json").strip()
        items = json.loads(raw)
        if isinstance(items, dict):
            items = [items]

        out: list[dict] = []
        for qa in items:
            if not qa.get("question") or not qa.get("ground_truth"):
                continue
            out.append({
                "question":              qa["question"],
                "ground_truth":          qa["ground_truth"],
                "ground_truth_contexts": [text],
                "source":                chunk["source"],
                "page":                  chunk["page"],
            })
        return out

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        print(f"  ✗ (chunk parse error: {e})")
        return []


# ── LangSmith upload ──────────────────────────────────────────────────────────

def upload_to_langsmith(dataset_name: str, records: list[dict]) -> str:
    """
    Create or extend a LangSmith dataset with the generated Q&A pairs.
    Returns the dataset URL.
    """
    from langsmith import Client
    from langsmith.utils import LangSmithNotFoundError

    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"  Dataset '{dataset_name}' exists — appending {len(records)} examples.")
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "Synthetic golden Q&A pairs for offline Ragas evaluation "
                "of the clinical trial compliance RAG system."
            ),
        )
        print(f"  Created dataset '{dataset_name}'.")

    client.create_examples(
        inputs=[{"question": r["question"]} for r in records],
        outputs=[
            {
                "ground_truth":          r["ground_truth"],
                "ground_truth_contexts": r["ground_truth_contexts"],
            }
            for r in records
        ],
        dataset_id=dataset.id,
    )

    return f"https://smith.langchain.com/o/default/datasets/{dataset.id}"


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    required_keys = ["ANTHROPIC_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    if not args.dry_run:
        required_keys.append("LANGCHAIN_API_KEY")

    missing = [k for k in required_keys if not os.environ.get(k)]
    if missing:
        print(f"❌ Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    target_pairs = args.sample * args.per_chunk
    print(f"\n🔬 Generating ~{target_pairs} Q&A pairs "
          f"({args.sample} chunks × {args.per_chunk}/chunk) → dataset '{args.dataset}'")
    print(f"   Concurrency: {args.concurrency}  |  "
          f"Mode: {'dry-run (no LangSmith upload)' if args.dry_run else 'live'}\n")

    anthropic = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # 1. Sample chunks
    print("📦 Sampling chunks from Qdrant...")
    chunks = await sample_chunks(n=args.sample)
    print(f"   Sampled {len(chunks)} chunks.\n")

    # 2. Generate Q&A pairs (concurrent, with a semaphore to respect rate limits)
    print("🤖 Generating Q&A pairs with Claude...")
    records: list[dict] = []
    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    total = len(chunks)

    async def worker(idx: int, chunk: dict):
        nonlocal done
        async with sem:
            if args.per_chunk == 1:
                qa = await generate_qa(anthropic, chunk)
                result = [qa] if qa else []
            else:
                result = await generate_qa_multi(anthropic, chunk, args.per_chunk)
        done += 1
        label = f"{chunk['source']} p.{chunk['page']}"
        print(f"  [{done:>3}/{total}] {label[:55]:<55} → {len(result)} pair(s)")
        return result

    results = await asyncio.gather(*(worker(i, c) for i, c in enumerate(chunks)))
    for r in results:
        records.extend(r)

    if not records:
        print("\n❌ No valid Q&A pairs generated. Check that chunks contain enough text.")
        sys.exit(1)

    print(f"\n  Generated {len(records)} valid pairs from {total} chunks.\n")

    # 3. Save locally
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"golden_{args.dataset}_{ts}.jsonl"

    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"💾 Saved locally → {out}")

    # 4. Upload to LangSmith (unless dry-run)
    if args.dry_run:
        print("🚫 Dry-run — skipping LangSmith upload.")
    else:
        print("📡 Uploading to LangSmith...")
        url = upload_to_langsmith(args.dataset, records)
        print(f"✅ Done! View dataset at:\n   {url}")

    print(f"\n📊 Summary: {len(records)} examples ready in '{args.dataset}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic golden Q&A dataset for clinical RAG offline evals."
    )
    parser.add_argument(
        "--sample", type=int, default=30,
        help="Number of Qdrant chunks to sample (default: 30).",
    )
    parser.add_argument(
        "--per-chunk", type=int, default=1,
        help="Distinct Q&A pairs to generate per chunk (default: 1). "
             "Total pairs ≈ sample × per-chunk.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent Claude calls (default: 5). Lower if you hit rate limits.",
    )
    parser.add_argument(
        "--dataset", type=str, default="clinical-rag-golden-set",
        help="LangSmith dataset name to create / append to.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate Q&A pairs locally but skip LangSmith upload.",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
