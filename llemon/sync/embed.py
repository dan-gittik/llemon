from __future__ import annotations

from typing import TYPE_CHECKING

import llemon.sync as llemon
from llemon.sync.types import History
from llemon.utils import Emoji

if TYPE_CHECKING:
    from llemon.sync import LLM


class EmbedRequest(llemon.Request):

    def __init__(
        self,
        *,
        llm: LLM,
        text: str,
        history: History | None = None,
    ) -> None:
        super().__init__(history=history)
        self.llm = llm
        self.text = text

    def __str__(self) -> str:
        return f"{self.llm}.embed({self.text!r})"

    def format(self, emoji: bool = True) -> str:
        embed = Emoji.EMBED if emoji else "Embed: " + self.text
        return f"{embed}{self.text}"

    def check_supported(self) -> None:
        pass


class EmbedResponse(llemon.Response):

    request: EmbedRequest

    def __init__(self, request: EmbedRequest) -> None:
        super().__init__(request)
        self.embedding: list[float] | None = None
        self.input_tokens = 0

    def __str__(self) -> str:
        return f"{self.request.llm}: {self.embedding!r}"

    def complete_embedding(self, embedding: list[float]) -> None:
        self.embedding = embedding

    def format(self, emoji: bool = True) -> str:
        embed = Emoji.EMBED if emoji else "Embedding: "
        return f"{embed}{self.embedding!r}"
