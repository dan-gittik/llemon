from __future__ import annotations

import openai

import llemon


class Ollama(llemon.OpenAILLM):

    def __init__(self) -> None:
        super().__init__()
        self.client = openai.AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
