from typing import cast

from llemon.genai.llm import LLM
from llemon.genai.llm_model import LLMModel


class LLMModelProperty:

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: object, owner: type) -> LLMModel:
        provider = cast(type[LLM], owner)
        return provider.model(self.name)
