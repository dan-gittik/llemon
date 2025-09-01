from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast

import llemon
from llemon.types import NS

if TYPE_CHECKING:
    from llemon import EmbedderProvider, EmbedResponse
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker


class Embedder(llemon.Serializeable):

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

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        provider = llemon.EmbedderProvider.get_subclass(unpacker.get("provider", str))
        return cast(Self, provider.embedder(model=unpacker.get("model", str)))

    def _dump(self, refs: DumpRefs) -> NS:
        return dict(
            provider=self.provider.__class__.__name__,
            model=self.model,
        )
