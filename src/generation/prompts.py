"""
prompts.py — All prompt templates in one place

To change how the LLM is instructed or how context is formatted,
edit this file only.  No other module contains prompt strings.
"""


SYSTEM_PROMPT = """You are a clinical trial regulatory compliance expert with deep knowledge \
of FDA regulations (21 CFR), ICH guidelines (E6, E8, E9), and GCP requirements.

RULES — follow these exactly:
1. Answer using ONLY the provided document excerpts. Never use outside knowledge.
2. If the excerpts do not contain enough information, respond with:
   "The available documents do not fully address this question. Please consult your regulatory affairs team."
3. Cite the source and page number for every factual claim, like this: [Source: filename.pdf, p.12]
4. Be precise and concise. Regulatory professionals need unambiguous answers."""


def build_user_message(question: str, contexts: list[dict]) -> str:
    """
    Format retrieved chunks into a grounded user message for Claude.

    Each chunk is labelled with its source and page so the model
    can cite them accurately.
    """
    if not contexts:
        return (
            "No relevant document excerpts were retrieved for this query.\n\n"
            f"Question: {question}"
        )

    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}, p.{c['page']} | Score: {c['score']}]\n{c['text']}"
        for c in contexts
    )

    return (
        f"Use the following document excerpts to answer the compliance question.\n\n"
        f"=== DOCUMENT EXCERPTS ===\n{context_block}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== ANSWER ==="
    )
