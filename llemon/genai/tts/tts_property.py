from __future__ import annotations

from typing import TYPE_CHECKING, cast

import llemon

if TYPE_CHECKING:
    from llemon import TTS


class TTSProperty:

    def __init__(self, model: str) -> None:
        self.model = model

    def __get__(self, instance: object, owner: type) -> TTS:
        provider = cast(type[llemon.TTSProvider], owner)
        return provider.tts(self.model)
