from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import llemon.sync as llemon

if TYPE_CHECKING:
    from llemon.sync import (
        STT,
        STTProperty,
        TranscribeRequest,
        TranscribeResponse,
    )

log = logging.getLogger(__name__)


class STTProvider(ABC, llemon.Provider):

    stts: ClassVar[dict[str, STT]] = {}
    default_stt: ClassVar[STTProperty | None] = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.stts = {}

    @classmethod
    def stt(cls, model: str, supports_timestamps: bool | None = None) -> STT:
        self = cls.get()
        if model not in self.stts:
            log.debug("creating model %s", model)
            config = llemon.STTConfig(
                model=model,
                supports_timestamps=supports_timestamps,
            )
            self.stts[model] = llemon.STT(self, model, config)
        return self.stts[model]

    @abstractmethod
    def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        raise NotImplementedError()
