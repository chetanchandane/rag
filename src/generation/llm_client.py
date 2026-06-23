"""
llm_client.py — LLM wrapper

Stage 1: Anthropic Claude (claude-sonnet-4-6)
Stage 2 (planned): Add OpenAI fallback / model routing

To swap models, change config.generation_model in src/config.py.
To swap providers entirely, implement the same generate() interface here.
"""

import os

import anthropic
from langsmith import traceable

from src.config import config
from src.generation.prompts import SYSTEM_PROMPT, build_user_message


class ClaudeClient:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

    @traceable(name="generate", run_type="llm")
    async def generate(self, question: str, contexts: list[dict]) -> str:
        """
        Generate a grounded answer from Claude given a question and
        a list of retrieved context chunks.

        Returns the raw text response.
        """
        user_message = build_user_message(question, contexts)

        message = await self.client.messages.create(
            model=config.generation_model,
            max_tokens=config.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return message.content[0].text
