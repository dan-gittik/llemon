from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import llemon
from llemon.types import History
from llemon.utils import Emoji

if TYPE_CHECKING:
    from llemon import Embedder


class EmbedRequest(llemon.Request):

    def __init__(
        self,
        *,
        embedder: Embedder,
        text: str,
        history: History | None = None,
    ) -> None:
        super().__init__(history=history)
        self.embedder = embedder
        self.text = text

    def __str__(self) -> str:
        return f"{self.embedder}.embed({self.text!r})"

    def format(self, emoji: bool = True) -> str:
        embed = Emoji.EMBED if emoji else "Embed: " + self.text
        return f"{embed}{self.text}"

    def check_supported(self) -> None:
        pass


class EmbedResponse(llemon.Response):

    request: EmbedRequest

    def __init__(self, request: EmbedRequest) -> None:
        super().__init__(request)
        self.embedding: NDArray[np.float32] | None = None
        self.input_tokens = 0

    def __str__(self) -> str:
        return f"{self.request.embedder}: {self.embedding!r}"

    def complete_embedding(self, embedding: bytes | list[float] | NDArray[np.float32]) -> None:
        if isinstance(embedding, bytes):
            self.embedding = np.frombuffer(embedding, dtype=np.float32)
        elif isinstance(embedding, list):
            self.embedding = np.array(embedding, dtype=np.float32)
        else:
            self.embedding = embedding

    def format(self, emoji: bool = True) -> str:
        embed = Emoji.EMBED if emoji else "Embedding: "
        return f"{embed}{self.embedding!r}"
