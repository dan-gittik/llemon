from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from llemon.sync import LLM, LLMProvider


class LLMProperty:

    def __init__(self, model: str) -> None:
        self.model = model

    def __get__(self, instance: object, owner: type) -> LLM:
        provider = cast(type[LLMProvider], owner)
        return provider.llm(self.model)
