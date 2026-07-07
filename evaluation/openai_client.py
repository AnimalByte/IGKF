from __future__ import annotations

import os
from typing import Any, Dict


class OpenAIGenerator:
    def __init__(self, model: str, max_output_tokens: int):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for non-dry-run OpenAI generation.")
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.max_output_tokens = max_output_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=self.max_output_tokens,
        )
        raw = response.model_dump(mode="json")
        text = getattr(response, "output_text", "") or ""
        return {
            "raw_model_output": raw,
            "parsed_answer": text.strip(),
            "prompt_token_count": None,
            "output_token_count": None,
            "context_length": None,
            "requested_output_tokens": self.max_output_tokens,
        }
