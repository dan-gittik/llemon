from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import llemon

if TYPE_CHECKING:
    from llemon import STT, STTProperty, TranscribeRequest, TranscribeResponse

log = logging.getLogger(__name__)


class STTProvider(ABC, llemon.Provider):

    stt_models: ClassVar[dict[str, STT]] = {}
    default_stt: ClassVar[STTProperty | None] = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.stt_models = {}

    @classmethod
    def stt(
        cls,
        model: str,
        supports_timestamps: bool | None = None,
        cost_per_1m_input_tokens: float | None = None,
        cost_per_minute: float | None = None,
    ) -> STT:
        self = cls.get()
        if model not in self.stt_models:
            log.debug("creating model %s", model)
            config = llemon.STTConfig(
                model=model,
                supports_timestamps=supports_timestamps,
                cost_per_1m_input_tokens=cost_per_1m_input_tokens,
                cost_per_minute=cost_per_minute,
            )
            self.stt_models[model] = llemon.STT(self, model, config)
        return self.stt_models[model]

    @abstractmethod
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        raise NotImplementedError()
