from __future__ import annotations

from typing import TYPE_CHECKING, cast

import llemon

if TYPE_CHECKING:
    from llemon import Embedder


class EmbedderProperty:

    def __init__(self, model: str) -> None:
        self.model = model

    def __get__(self, instance: object, owner: type) -> Embedder:
        provider = cast(type[llemon.EmbedderProvider], owner)
        return provider.embedder(self.model)
