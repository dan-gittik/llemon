from __future__ import annotations
from typing import TYPE_CHECKING

import llemon

if TYPE_CHECKING:
    from llemon import EmbedderProvider, EmbedResponse


class Embedder:

    def __init__(self, provider: EmbedderProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"

    def __repr__(self) -> str:
        return f"<{self}>"

    async def embed(self, text: str) -> EmbedResponse:
        request = llemon.EmbedRequest(embedder=self, text=text)
        return await self.provider.embed(request)