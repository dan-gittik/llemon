from __future__ import annotations

import openai

from llemon.sync.openai import OpenAI


class Ollama(OpenAI):

    def __init__(self) -> None:
        self.client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
