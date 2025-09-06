from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import llemon

if TYPE_CHECKING:
    from llemon import TTS, SynthesizeRequest, SynthesizeResponse, TTSModel

log = logging.getLogger(__name__)


class TTSProvider(ABC, llemon.Provider):

    tts_models: ClassVar[dict[str, TTS]] = {}
    default_tts: ClassVar[TTSModel | None] = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.tts_models = {}

    @classmethod
    def tts(
        cls,
        model: str,
        *,
        supports_timestamps: bool | None = None,
        supports_formats: list[str] | None = None,
        supports_streaming: bool | None = None,
        cost_per_1m_characters: float | None = None,
        cost_per_1m_tokens: float | None = None,
    ) -> TTS:
        self = cls.get()
        if model not in self.tts_models:
            log.debug("creating model %s", model)
            config = llemon.TTSConfig(
                model=model,
                supports_timestamps=supports_timestamps,
                supports_formats=supports_formats,
                supports_streaming=supports_streaming,
                cost_per_1m_characters=cost_per_1m_characters,
                cost_per_1m_tokens=cost_per_1m_tokens,
            )
            self.tts_models[model] = llemon.TTS(self, model, config)
        return self.tts_models[model]

    async def synthesize(self, request: SynthesizeRequest) -> SynthesizeResponse:
        request.check_supported()
        response = llemon.SynthesizeResponse(request)
        await self._synthesize(request, response)
        return response

    @abstractmethod
    async def _synthesize(self, request: SynthesizeRequest, response: SynthesizeResponse) -> None:
        raise NotImplementedError()
