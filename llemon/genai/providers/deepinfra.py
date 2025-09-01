from __future__ import annotations

import openai

import llemon


class DeepInfra(llemon.OpenAILLM):

    llama31_70b = llemon.LLMProperty("meta-llama/Meta-Llama-3.1-70B-Instruct")
    llama31_8b = llemon.LLMProperty("meta-llama/Meta-Llama-3.1-8B-Instruct")

    def __init__(self, api_key: str) -> None:
        self.client = openai.AsyncOpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=api_key,
        )
