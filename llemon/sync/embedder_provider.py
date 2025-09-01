from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import llemon.sync as llemon

if TYPE_CHECKING:
    from llemon.sync import (
        Embedder,
        EmbedderProperty,
        EmbedRequest,
        EmbedResponse,
    )

log = logging.getLogger(__name__)


class EmbedderProvider(ABC, llemon.Provider):

    embedders: ClassVar[dict[str, Embedder]] = {}
    default_embedder: ClassVar[EmbedderProperty | None] = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.embedders = {}

    @classmethod
    def embedder(cls, model: str) -> Embedder:
        self = cls.get()
        if model not in self.embedders:
            log.debug("creating model %s", model)
            self.embedders[model] = llemon.Embedder(self, model)
        return self.embedders[model]

    @abstractmethod
    def embed(self, request: EmbedRequest) -> EmbedResponse:
        raise NotImplementedError()
