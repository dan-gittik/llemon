from __future__ import annotations
import openai

import llemon


class Ollama(llemon.OpenAI):

    def __init__(self) -> None:
        self.client = openai.AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
