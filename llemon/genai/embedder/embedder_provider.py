from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import llemon

if TYPE_CHECKING:
    from llemon import (
        Embedder,
        EmbedderProperty,
        EmbedRequest,
        EmbedResponse,
    )

log = logging.getLogger(__name__)


class EmbedderProvider(ABC, llemon.Provider):

    embedder_models: ClassVar[dict[str, Embedder]] = {}
    default_embedder: ClassVar[EmbedderProperty | None] = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.embedder_models = {}

    @classmethod
    def embedder(cls, model: str) -> Embedder:
        self = cls.get()
        if model not in self.embedder_models:
            log.debug("creating model %s", model)
            config = llemon.EmbedderConfig(
                model=model,
            )
            self.embedder_models[model] = llemon.Embedder(self, model, config)
        return self.embedder_models[model]

    @abstractmethod
    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        raise NotImplementedError()
