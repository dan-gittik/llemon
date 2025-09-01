from __future__ import annotations

from typing import TYPE_CHECKING, cast

import llemon.sync as llemon

if TYPE_CHECKING:
    from llemon.sync import LLM


class LLMProperty:

    def __init__(self, model: str) -> None:
        self.model = model

    def __get__(self, instance: object, owner: type) -> LLM:
        provider = cast(type[llemon.LLMProvider], owner)
        return provider.llm(self.model)
