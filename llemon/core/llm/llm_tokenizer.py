from __future__ import annotations

from typing import Sequence


class LLMTokenizer:

    async def count(self, text: str) -> int:
        raise NotImplementedError()

    async def parse(self, text: str) -> Sequence[LLMToken]:
        raise NotImplementedError()

    async def encode(self, *texts: str) -> list[int]:
        raise NotImplementedError()

    async def decode(self, ids: list[int]) -> str:
        raise NotImplementedError()


class LLMToken:

    def __init__(self, id: int):
        self.id = id

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"<token {self.id}: {self.text}>"

    @property
    def text(self) -> str:
        raise NotImplementedError()

    @property
    def offset(self) -> int:
        raise NotImplementedError()
