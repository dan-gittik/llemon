from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import llemon.sync as llemon
from llemon.sync.types import NS
from llemon.utils import filtered_dict

if TYPE_CHECKING:
    from llemon.sync import EmbedderConfig, EmbedderProvider, EmbedResponse
    from llemon.sync.serializeable import DumpRefs, LoadRefs, Unpacker


class Embedder(llemon.Serializeable):

    def __init__(self, provider: EmbedderProvider, model: str, config: EmbedderConfig) -> None:
        self.provider = provider
        self.model = model
        self.config = config

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"

    def __repr__(self) -> str:
        return f"<{self}>"

    def embed(self, text: str, **provider_options: Any) -> EmbedResponse:
        request = llemon.EmbedRequest(embedder=self, text=text, **provider_options)
        return self.provider.embed(request)

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        provider = llemon.EmbedderProvider.get_subclass(unpacker.get("provider", str))
        embedder = provider.embedder(model=unpacker.get("model", str), **unpacker.get("config", dict, {}))
        return cast(Self, embedder)

    def _dump(self, refs: DumpRefs) -> NS:
        config = self.config._dump(refs)
        del config["model"]
        return filtered_dict(
            provider=self.provider.__class__.__name__,
            model=self.model,
            config=config,
        )


EmbedderModel = llemon.Model[Embedder]
