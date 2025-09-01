from __future__ import annotations

import openai

import llemon.sync as llemon


class Ollama(llemon.OpenAILLM):

    def __init__(self) -> None:
        self.client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
