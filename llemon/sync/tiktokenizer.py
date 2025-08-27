from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Sequence

import tiktoken

from llemon.sync.llm_model import LLMModel
from llemon.sync.count_tokens import count_tokens
from llemon.sync.llm_tokenizer import LLMToken, LLMTokenizer
from llemon.sync.generate import GenerateRequest

ENCODINGS: dict[str, tiktoken.Encoding] = {}


class TikTokenizer(LLMTokenizer):

    label: ClassVar[str] = "tiktoken"

    def __init__(self, model: LLMModel) -> None:
        super().__init__(model)
        if model.name not in ENCODINGS:
            ENCODINGS[model.name] = tiktoken.encoding_for_model(model.name)
        self.encoding = ENCODINGS[model.name]

    def encode(self, *texts: str) -> list[int]:
        ids: list[int] = []
        for text in texts:
            ids.extend(self.encoding.encode(text))
        return ids

    def decode(self, *ids: int) -> str:
        return self.encoding.decode(ids)

    def parse(self, text: str) -> Sequence[OpenAIToken]:
        tokens: list[OpenAIToken] = []
        token: OpenAIToken | None = None
        for token_id in self.encoding.encode(text):
            token = OpenAIToken(token_id, self.encoding, token)
            tokens.append(token)
        return tokens

    def _count(self, request: GenerateRequest) -> int:
        return count_tokens(request, self.__count)

    def __count(self, text: str) -> int:
        return len(self.encoding.encode(text))


class OpenAIToken(LLMToken):

    def __init__(self, id: int, encoding: tiktoken.Encoding, prev: OpenAIToken | None) -> None:
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
