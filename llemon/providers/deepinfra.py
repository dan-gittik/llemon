from __future__ import annotations

import openai

from llemon.apis.llm.llm_model_property import LLMModelProperty
from llemon.providers.openai import OpenAI


class DeepInfra(OpenAI):

    llama31_70b = LLMModelProperty("meta-llama/Meta-Llama-3.1-70B-Instruct")
    llama31_8b = LLMModelProperty("meta-llama/Meta-Llama-3.1-8B-Instruct")

    def __init__(self, api_key: str) -> None:
        self.client = openai.AsyncOpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=api_key,
        )
