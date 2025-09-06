from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Sequence

import tiktoken

import llemon.sync as llemon

if TYPE_CHECKING:
    from llemon.sync import LLM

ENCODINGS: dict[str, tiktoken.Encoding] = {}


class TikTokenizer(llemon.LLMTokenizer):

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        if llm.model not in ENCODINGS:
            ENCODINGS[llm.model] = tiktoken.encoding_for_model(llm.model)
        self.encoding = ENCODINGS[llm.model]

    def encode(self, *texts: str) -> list[int]:
        ids: list[int] = []
        for text in texts:
            ids.extend(self.encoding.encode(text))
        return ids

    def decode(self, *ids: int) -> str:
        return self.encoding.decode(ids)

    def parse(self, text: str) -> Sequence[TikToken]:
        tokens: list[TikToken] = []
        token: TikToken | None = None
        for token_id in self.encoding.encode(text):
            token = TikToken(token_id, self.encoding, token)
            tokens.append(token)
        return tokens

    def _count(self, text: str) -> int:
        return len(self.encoding.encode(text))


class TikToken(llemon.LLMToken):

    def __init__(self, id: int, encoding: tiktoken.Encoding, prev: TikToken | None) -> None:
        super().__init__(id)
        self._encoding = encoding
        self._prev = prev

    @cached_property
    def text(self) -> str:
        return self._encoding.decode([self.id])

    @cached_property
    def offset(self) -> int:
        if self._prev is None:
            return 0
        return self._prev.offset + len(self._prev.text)
