from __future__ import annotations

from decimal import Decimal
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

import llemon.sync as llemon
from llemon.sync.types import NS
from llemon.utils import Emoji, filtered_dict

if TYPE_CHECKING:
    from llemon.sync import Embedder
    from llemon.sync.serializeable import DumpRefs, LoadRefs, Unpacker


class EmbedRequest(llemon.Request):

    def __init__(
        self,
        *,
        embedder: Embedder,
        text: str,
        **provider_options: Any,
    ) -> None:
        super().__init__(self._overrides(embedder.provider, provider_options))
        self.embedder = embedder
        self.text = text

    def __str__(self) -> str:
        return f"{self.embedder}.embed({self.text!r})"

    def format(self, emoji: bool = True) -> str:
        embed = Emoji.EMBED if emoji else "Embed: " + self.text
        return f"{embed}{self.text}"

    def check_supported(self) -> None:
        pass

    @classmethod
    def _restore(cls, unpacker: Unpacker, refs: LoadRefs) -> tuple[NS, NS]:
        args, attrs = super()._restore(unpacker, refs)
        args.update(
            embedder=refs.get_embedder(unpacker.get("embedder", str)),
            text=unpacker.get("text", str),
        )
        return args, attrs

    def _dump(self, refs: DumpRefs) -> NS:
        refs.add_embedder(self.embedder)
        data = super()._dump(refs)
        data.update(
            filtered_dict(
                embedder=self.embedder.model,
                text=self.text,
            )
        )
        return data


class EmbedResponse(llemon.Response):

    request: EmbedRequest

    def __init__(self, request: EmbedRequest) -> None:
        super().__init__(request)
        self.embedding: NDArray[np.float32] | None = None
        self.input_tokens = 0

    def __str__(self) -> str:
        return f"{self.request.embedder}: {self.embedding!r}"

    @cached_property
    def cost(self) -> Decimal:
        return (Decimal(self.input_tokens) * Decimal(self.request.embedder.config.cost_per_1m_tokens or 0)) / 1_000_000

    def complete_embedding(self, embedding: bytes | list[float] | NDArray[np.float32]) -> None:
        if isinstance(embedding, bytes):
            self.embedding = np.frombuffer(embedding, dtype=np.float32)
        elif isinstance(embedding, list):
            self.embedding = np.array(embedding, dtype=np.float32)
        else:
            self.embedding = embedding
        self.complete()

    def format(self, emoji: bool = True) -> str:
        embed = Emoji.EMBED if emoji else "Embedding: "
        return f"{embed}{self.embedding!r}"

    def _restore(self, unpacker: Unpacker, refs: LoadRefs) -> None:
        super()._restore(unpacker, refs)
        self.embedding = np.frombuffer(unpacker.get("embedding", bytes), dtype=np.float32)
        self.input_tokens = unpacker.get("input_tokens", int)

    def _dump(self, refs: DumpRefs) -> NS:
        data = super()._dump(refs)
        data.update(
            filtered_dict(
                embedding=self.embedding.tobytes() if self.embedding is not None else None,
                input_tokens=self.input_tokens,
            )
        )
        data.update(super()._dump(refs))
        return data
