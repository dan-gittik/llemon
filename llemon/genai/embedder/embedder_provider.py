from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import llemon

if TYPE_CHECKING:
    from llemon import (
        Embedder,
        EmbedderModel,
        EmbedRequest,
        EmbedResponse,
    )

log = logging.getLogger(__name__)


class EmbedderProvider(ABC, llemon.Provider):

    embedder_models: ClassVar[dict[str, Embedder]] = {}
    default_embedder: ClassVar[EmbedderModel | None] = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.embedder_models = {}

    @classmethod
    def embedder(
        cls,
        model: str,
        *,
        cost_per_1m_tokens: float | None = None,
    ) -> Embedder:
        self = cls.get()
        if model not in self.embedder_models:
            log.debug("creating model %s", model)
            config = llemon.EmbedderConfig(
                model=model,
                cost_per_1m_tokens=cost_per_1m_tokens,
            )
            self.embedder_models[model] = llemon.Embedder(self, model, config)
        return self.embedder_models[model]

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        request.check_supported()
        response = llemon.EmbedResponse(request)
        await self._embed(request, response)
        return response

    @abstractmethod
    async def _embed(self, request: EmbedRequest, response: EmbedResponse) -> None:
        raise NotImplementedError()
