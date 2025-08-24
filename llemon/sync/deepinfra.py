from __future__ import annotations

import openai

from llemon.core.llm.llm_model_property import LLMModelProperty
from llemon.sync.huggingface import HuggingFaceTokenizer
from llemon.sync.llm_model import LLMModel
from llemon.sync.llm_tokenizer import LLMTokenizer
from llemon.sync.openai import OpenAI


class DeepInfra(OpenAI):

    llama31_70b = LLMModelProperty("meta-llama/Meta-Llama-3.1-70B-Instruct")
    llama31_8b = LLMModelProperty("meta-llama/Meta-Llama-3.1-8B-Instruct")

    def __init__(self, api_key: str) -> None:
        self.client = openai.OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=api_key,
        )

    def get_tokenizer(self, model: LLMModel) -> LLMTokenizer:
        return HuggingFaceTokenizer(model)
