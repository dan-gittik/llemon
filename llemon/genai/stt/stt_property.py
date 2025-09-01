from __future__ import annotations

from typing import TYPE_CHECKING, cast

import llemon

if TYPE_CHECKING:
    from llemon import STT


class STTProperty:

    def __init__(self, model: str) -> None:
        self.model = model

    def __get__(self, instance: object, owner: type) -> STT:
        provider = cast(type[llemon.STTProvider], owner)
        return provider.stt(self.model)
